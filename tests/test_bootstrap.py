"""Tests for the confidence intervals.

An interval is a claim about how much a number would move if the run were
repeated, and a wrong one is worse than none: it turns "we cannot tell these
apart" into "we measured a difference". So these tests check the properties that
make an interval trustworthy rather than merely present -- that it contains the
point estimate, that it narrows as evidence accumulates, that it is reproducible,
and that it declines to separate two results that are genuinely the same.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.engine.metrics.bootstrap import (
    Interval,
    bootstrap_statistic,
    cer_interval,
    fid_interval,
    rate_interval,
)

# ---------------------------------------------------------------------------
# the shape of an interval
# ---------------------------------------------------------------------------


def test_the_interval_contains_the_point_estimate():
    hits = [True] * 30 + [False] * 70

    interval = rate_interval(hits, resamples=500)

    assert interval.low <= interval.value <= interval.high
    assert interval.value == pytest.approx(0.30)


def test_more_evidence_narrows_the_interval():
    """The property that makes the interval worth reporting: it should shrink
    with sample size, or it is not measuring uncertainty."""
    small = rate_interval([True, False] * 25, resamples=500)
    large = rate_interval([True, False] * 500, resamples=500)

    assert large.half_width < small.half_width / 2


def test_the_same_data_gives_the_same_interval():
    """A confidence interval that moves between runs cannot be quoted."""
    hits = [True] * 40 + [False] * 60

    first = rate_interval(hits, resamples=400, seed=7)
    second = rate_interval(hits, resamples=400, seed=7)

    assert (first.low, first.high) == (second.low, second.high)


def test_a_unanimous_sample_has_no_spread():
    interval = rate_interval([True] * 50, resamples=300)

    assert interval.value == 1.0
    assert interval.half_width == 0.0


def test_resampling_needs_more_than_one_sample():
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_statistic(lambda idx: 0.0, count=1)


# ---------------------------------------------------------------------------
# telling two results apart -- the reason any of this exists
# ---------------------------------------------------------------------------


def test_two_runs_of_the_same_thing_do_not_separate():
    """The guard against reading noise as improvement. Two samples from one
    distribution must not be declared different."""
    rng = np.random.default_rng(0)
    truth = 0.25
    first = rate_interval(rng.random(400) < truth, resamples=500, seed=1)
    second = rate_interval(rng.random(400) < truth, resamples=500, seed=2)

    assert not first.separates_from(second)


def test_a_large_real_difference_does_separate():
    """And the interval must not be so wide that nothing is ever decidable."""
    poor = rate_interval([True] * 60 + [False] * 340, resamples=500)
    good = rate_interval([True] * 240 + [False] * 160, resamples=500)

    assert poor.separates_from(good)
    assert good.separates_from(poor)


# ---------------------------------------------------------------------------
# CER: a ratio of totals, not a mean of ratios
# ---------------------------------------------------------------------------


def test_cer_aggregates_the_corpus_rather_than_averaging_rates():
    """One short line wrong and one long line right is not a 50% error rate.

    Averaging per-sample rates would say 50%; the corpus ratio says 2 errors in
    22 characters, which is 9%. The metric reports the corpus ratio, so the
    interval has to be built on the same quantity or it describes a different
    number from the one beside it.
    """
    interval = cer_interval(errors=[2, 0], lengths=[2, 20], resamples=200)

    assert interval.value == pytest.approx(2 / 22)


def test_cer_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="error counts for"):
        cer_interval(errors=[1, 2, 3], lengths=[10, 20], resamples=100)


def test_cer_of_a_perfect_transcription_is_zero():
    interval = cer_interval(errors=[0] * 20, lengths=[30] * 20, resamples=200)

    assert interval.value == 0.0
    assert interval.high == 0.0


# ---------------------------------------------------------------------------
# FID: a statistic of the whole set, through the same machinery
# ---------------------------------------------------------------------------


def test_fid_of_a_set_against_itself_is_near_zero_with_a_spread():
    """The interval must not collapse to nothing here. FID against the same
    distribution is ~0 in expectation, and a resampled subset still wobbles --
    an interval of exactly zero would mean the resampling is not happening."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, 16))

    interval = fid_interval(features, features, resamples=100)

    assert interval.value == pytest.approx(0.0, abs=1e-6)
    assert interval.high > 0.0


def test_fid_separates_two_genuinely_different_distributions():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(150, 16))
    close = rng.normal(size=(150, 16))
    far = rng.normal(loc=4.0, size=(150, 16))

    assert (
        fid_interval(real, far, resamples=100).value
        > fid_interval(real, close, resamples=100).value
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_a_rate_is_formatted_as_a_percentage():
    assert "%" in Interval(0.221, 0.18, 0.26, 2000).format(as_percent=True)


def test_a_distance_is_formatted_as_a_number():
    text = Interval(69.46, 64.1, 74.9, 2000).format()

    assert "69.46" in text
    assert "%" not in text


# ---------------------------------------------------------------------------
# the subspace projection: exactness bought back as speed
# ---------------------------------------------------------------------------


def test_projecting_to_the_span_does_not_change_fid():
    """The property the whole optimisation rests on.

    If this drifts, every FID the project has reported since is a different
    quantity from the ones before it, and no comparison across runs holds.
    """
    from nib.engine.metrics.bootstrap import project_to_span
    from nib.engine.metrics.fid import compute_fid

    rng = np.random.default_rng(0)
    real = rng.normal(size=(60, 256))
    generated = rng.normal(loc=0.3, size=(60, 256))

    before = compute_fid(real, generated).value
    after = compute_fid(*project_to_span(real, generated)).value

    assert after == pytest.approx(before, rel=1e-6)


def test_the_projection_actually_reduces_the_dimension():
    """40 + 40 samples cannot span 256 dimensions, and computing there is why
    one bootstrap draw took 38 seconds at 2048."""
    from nib.engine.metrics.bootstrap import project_to_span

    rng = np.random.default_rng(1)
    real, generated = project_to_span(rng.normal(size=(40, 256)), rng.normal(size=(40, 256)))

    assert real.shape[1] <= 79, "40 + 40 samples span at most 79 dimensions"
    assert generated.shape[1] == real.shape[1]


def test_the_projection_leaves_already_small_features_alone():
    """More samples than dimensions: nothing to remove, and no basis change
    that would perturb the value for no gain."""
    from nib.engine.metrics.bootstrap import project_to_span

    rng = np.random.default_rng(2)
    real, _ = project_to_span(rng.normal(size=(80, 8)), rng.normal(size=(80, 8)))

    assert real.shape[1] == 8


def test_the_fid_interval_is_quick_enough_to_run_inline():
    """A guard on the mistake, not just the fix. The first version of this asked
    a Colab session for 21 hours of matrix square roots."""
    import time

    from nib.engine.metrics.bootstrap import fid_interval

    rng = np.random.default_rng(3)
    started = time.perf_counter()
    fid_interval(rng.normal(size=(60, 512)), rng.normal(size=(60, 512)), resamples=25)
    elapsed = time.perf_counter() - started

    assert elapsed < 20, f"25 draws took {elapsed:.0f}s; 200 would be unusable"


def test_the_fid_interval_contains_its_own_estimate():
    """The bug this caught, kept as a guard.

    FID rises as the number of distinct samples falls, and a bootstrap draw is
    only ~63% distinct, so every resampled value is biased upward together. The
    raw percentile interval came out at [3068, 3124] around an estimate of 2881
    -- entirely above the number it claimed to describe.
    """
    from nib.engine.metrics.bootstrap import fid_interval

    rng = np.random.default_rng(4)
    interval = fid_interval(
        rng.normal(size=(80, 128)), rng.normal(loc=0.2, size=(80, 128)), resamples=40
    )

    assert interval.low <= interval.value <= interval.high
    assert interval.half_width > 0, "a spread of zero means the resampling did nothing"
