"""Character Error Rate: is the generated text actually readable?

A generative model can produce beautiful handwriting that says something other
than what was asked. FID will not notice -- it compares distributions, not
content -- and neither will a style metric. CER is the only one of the three that
checks the *words*.

How it works: run a handwriting recogniser over the generated image, compare what
it read to what was requested, and count character edits. The recogniser is a
judge, not part of the system being built; it is a separate pre-trained model.

**A number without a baseline is meaningless.** A recogniser makes mistakes on
genuine handwriting too, so 8% CER on generated images could be excellent or
terrible depending on what it scores on real ones. Every reported figure must sit
next to the same recogniser's score on real images from the same writers, and
:func:`relative_cer` exists so that comparison is hard to skip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Recogniser(Protocol):
    """Anything that can read handwriting. Kept behind a Protocol so the metric
    can be built and tested without downloading a 1.4 GB model."""

    def read(self, images: Sequence[np.ndarray]) -> list[str]: ...


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: insertions, deletions and substitutions.

    Two rows rather than a full matrix -- the strings here are words, but the same
    function is used on whole lines later, and the full matrix is wasteful.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def cer(prediction: str, target: str) -> float:
    """Character error rate for one pair.

    Normalised by the *target* length, which is the convention in the handwriting
    literature. Normalising by the prediction would let a model score well by
    emitting nothing.
    """
    if not target:
        return 0.0 if not prediction else 1.0
    return edit_distance(prediction, target) / len(target)


def corpus_cer(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """CER over a whole set: total edits divided by total target characters.

    Not the mean of per-sample rates. Averaging rates gives a one-character word
    the same weight as a twelve-character one, so a handful of short words can
    swing the figure -- and short words are exactly where a recogniser errs.
    """
    if len(predictions) != len(targets):
        raise ValueError(f"{len(predictions)} predictions for {len(targets)} targets")
    if not targets:
        raise ValueError("cannot compute CER over an empty set")

    edits = sum(edit_distance(p, t) for p, t in zip(predictions, targets, strict=True))
    characters = sum(len(t) for t in targets)
    if characters == 0:
        return 0.0
    return edits / characters


@dataclass
class CerResult:
    """A CER figure and the baseline that makes it interpretable."""

    generated: float
    real: float | None = None
    num_samples: int = 0

    @property
    def gap(self) -> float | None:
        """How much worse the generated images read than real ones.

        This, not the raw figure, is the number worth reporting. A recogniser
        scoring 6% on real handwriting and 9% on ours means a gap of 3 points --
        which reads very differently from a bare "9% CER".
        """
        return None if self.real is None else self.generated - self.real

    def summary(self) -> str:
        lines = [f"CER (generated)  {self.generated:.2%}   over {self.num_samples} samples"]
        if self.real is not None:
            lines.append(f"CER (real)       {self.real:.2%}   <- the recogniser's own error rate")
            lines.append(f"gap              {self.gap:+.2%}")
        else:
            lines.append("CER (real)       not measured -- this figure has no baseline")
        return "\n".join(lines)


def evaluate(
    recogniser: Recogniser,
    generated_images: Sequence[np.ndarray],
    targets: Sequence[str],
    real_images: Sequence[np.ndarray] | None = None,
    real_targets: Sequence[str] | None = None,
    batch_size: int = 32,
) -> CerResult:
    """Read the generated images and score them, with a baseline where possible."""
    if len(generated_images) != len(targets):
        raise ValueError(f"{len(generated_images)} images for {len(targets)} targets")
    if not generated_images:
        raise ValueError("nothing to evaluate")

    predictions = _read_in_batches(recogniser, generated_images, batch_size)
    generated = corpus_cer(predictions, list(targets))

    real = None
    if real_images is not None:
        if real_targets is None or len(real_images) != len(real_targets):
            raise ValueError("real_images and real_targets must be the same length")
        if real_images:
            real = corpus_cer(
                _read_in_batches(recogniser, real_images, batch_size), list(real_targets)
            )

    return CerResult(generated=generated, real=real, num_samples=len(generated_images))


def _read_in_batches(
    recogniser: Recogniser, images: Sequence[np.ndarray], batch_size: int
) -> list[str]:
    out: list[str] = []
    for start in range(0, len(images), batch_size):
        out.extend(recogniser.read(list(images[start : start + batch_size])))
    if len(out) != len(images):
        raise RuntimeError(f"the recogniser returned {len(out)} readings for {len(images)} images")
    return out
