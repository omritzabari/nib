"""Confidence intervals for the three metrics, by resampling what was measured.

Every number this project has reported so far is a single point estimate with no
sense of its own noise. Two evaluation runs that differed only in the generator's
token budget returned FID 63.92 and 69.46 -- and nothing in either number says
whether 5.5 points is a real difference or the spread you would get by running
the same configuration twice. Without that, no comparison between two models,
two checkpoints or two settings can be decided.

**Resampling, not repeated generation.** Running the whole evaluation ten times
would cost ten hours of GPU. The bootstrap gets the same answer from one run:
draw the same number of samples *with replacement* from the ones you have,
recompute the metric, and repeat. The spread of those recomputed values
estimates the spread you would have seen across real runs.

What that does and does not cover matters, and is stated here rather than
implied. It captures the variance from **which samples happened to be drawn** --
which writers, which lines, which lengths. It does not capture the variance from
the generator's own randomness on a fixed sample, because the images are fixed
once generated. So these intervals are a floor on the true run-to-run spread,
and two results whose intervals overlap are certainly not distinguishable.

Each metric needs a different quantity saved, and the choice is about cost:

* FID resamples **Inception features**, not images -- 2048 floats per sample
  rather than a picture, so a whole run's worth is a few megabytes and the
  recomputation is a covariance, not a forward pass.
* Retrieval resamples per-query **hits**, which are booleans.
* CER resamples per-sample **edit distances and lengths**, and re-aggregates the
  corpus ratio rather than averaging per-sample rates -- the two differ, and the
  corpus ratio is what the metric reports.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 2000
"""Enough for a stable 95% interval without being slow. The interval's own
error falls as 1/sqrt(resamples), so 2000 puts it well inside the last digit
this project reports."""

DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate and the range resampling puts around it."""

    value: float
    low: float
    high: float
    resamples: int
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def half_width(self) -> float:
        """The +/- most people actually want to read."""
        return (self.high - self.low) / 2.0

    def separates_from(self, other: Interval) -> bool:
        """Whether these two results can be told apart at all.

        Non-overlapping intervals are a conservative test -- two results *can*
        differ significantly while their intervals overlap slightly -- so this
        answers the question worth asking of a project's own numbers: is the
        difference large enough that nobody has to argue about it?
        """
        return self.high < other.low or other.high < self.low

    def format(self, as_percent: bool = False, digits: int = 2) -> str:
        if as_percent:
            return f"{self.value:.1%}  [{self.low:.1%}, {self.high:.1%}]"
        return f"{self.value:.{digits}f}  [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def _percentile_interval(
    point: float,
    resampled: Sequence[float],
    resamples: int,
    confidence: float,
) -> Interval:
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(resampled, [100 * tail, 100 * (1 - tail)])
    return Interval(
        value=float(point),
        low=float(low),
        high=float(high),
        resamples=resamples,
        confidence=confidence,
    )


def resample_indices(count: int, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap draw: ``count`` indices, with replacement."""
    return rng.integers(0, count, size=count)


def bootstrap_statistic(
    statistic: Callable[[np.ndarray], float],
    count: int,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 1337,
) -> Interval:
    """Interval for any statistic that is a function of a sample's indices.

    Taking indices rather than values is what lets FID -- which depends on the
    whole set at once and not on each sample independently -- use the same
    machinery as a simple mean.
    """
    if count < 2:
        raise ValueError(f"need at least two samples to resample, got {count}")

    rng = np.random.default_rng(seed)
    point = statistic(np.arange(count))
    draws = [statistic(resample_indices(count, rng)) for _ in range(resamples)]
    return _percentile_interval(point, draws, resamples, confidence)


def fid_interval(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 1337,
) -> Interval:
    """Interval for FID, resampling the generated set.

    The real set is held fixed. It is the reference the model is being measured
    against, and resampling both would widen the interval with uncertainty about
    the ruler rather than about the thing being measured.

    Note that FID is biased by sample count -- it falls as the count rises -- and
    a bootstrap draw has the same count as the original by construction, so the
    bias is held constant and the interval describes spread rather than
    correcting the bias. Two FID figures remain comparable only at equal counts,
    intervals or no intervals.
    """
    from nib.engine.metrics.fid import frechet_distance, gaussian_statistics

    mu_real, sigma_real = gaussian_statistics(real_features)

    def statistic(indices: np.ndarray) -> float:
        mu_gen, sigma_gen = gaussian_statistics(generated_features[indices])
        value, _ = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return value

    return bootstrap_statistic(statistic, len(generated_features), resamples, confidence, seed)


def rate_interval(
    hits: Sequence[bool],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 1337,
) -> Interval:
    """Interval for a proportion: writer retrieval top-1 or top-5."""
    outcomes = np.asarray(hits, dtype=float)
    return bootstrap_statistic(
        lambda indices: float(outcomes[indices].mean()),
        len(outcomes),
        resamples,
        confidence,
        seed,
    )


def cer_interval(
    errors: Sequence[int],
    lengths: Sequence[int],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 1337,
) -> Interval:
    """Interval for corpus CER: total edits over total reference characters.

    Re-aggregated per draw rather than averaged over per-sample rates. A short
    line and a long one contribute equally to a mean of rates and unequally to
    the corpus ratio, and the corpus ratio is what this project reports.
    """
    edits = np.asarray(errors, dtype=float)
    chars = np.asarray(lengths, dtype=float)
    if len(edits) != len(chars):
        raise ValueError(f"{len(edits)} error counts for {len(chars)} lengths")

    def statistic(indices: np.ndarray) -> float:
        total = chars[indices].sum()
        return float(edits[indices].sum() / total) if total else 0.0

    return bootstrap_statistic(statistic, len(edits), resamples, confidence, seed)
