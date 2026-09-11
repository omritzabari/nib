"""Tests for HWD's arithmetic and its reference set.

None of these load the VGG or need the `hwd` package: the network sits behind an
extractor protocol, and a stand-in that summarises an image by two numbers is
enough to check that the right things are compared with the right things.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from nib.engine.metrics import hwd


class TwoNumberExtractor:
    """Stands in for the VGG: one feature row per image, brightness and width."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, images, writer_ids):
        self.calls += 1
        rows = np.array(
            [[float(np.mean(image)) / 255.0, image.shape[1] / 100.0] for image in images]
        ).reshape(len(images), 2)
        return hwd.writer_means(rows, writer_ids)


def _line(level: int, width: int = 50) -> np.ndarray:
    return np.full((64, width), level, dtype=np.uint8)


# ---------------------------------------------------------------------------
# The reference set
# ---------------------------------------------------------------------------


def _pack(**sizes: int) -> dict[str, list[str]]:
    return {writer: [f"{writer}-{i:02d}" for i in range(n)] for writer, n in sizes.items()}


def test_the_reference_never_holds_a_target_or_a_style_line():
    """A target in the reference is compared with itself, and a style line in it
    favours the model, which was shown that very image."""
    by_writer = _pack(a=20, b=20)
    consumed = {"a-00", "a-01", "b-05"}

    reference = hwd.select_reference(by_writer, consumed, ["a", "b", "a"], depth=12, minimum=6)

    assert not set(reference.keys) & consumed
    assert Counter(reference.writer_ids) == {"a": 12, "b": 12}
    assert len(set(reference.keys)) == len(reference.keys)
    assert all(
        key.startswith(writer)
        for key, writer in zip(reference.keys, reference.writer_ids, strict=True)
    )


def test_a_writer_without_enough_spare_lines_is_named_not_skipped():
    by_writer = _pack(a=20, thin=7)
    consumed = {"thin-00", "thin-01"}  # five spare, one short of the minimum

    reference = hwd.select_reference(by_writer, consumed, ["a", "thin"], minimum=6)

    assert reference.withheld == ["thin"]
    assert "thin" not in reference.writer_ids


def test_the_reference_is_fixed_by_its_seed():
    by_writer = _pack(a=20, b=20)
    first = hwd.select_reference(by_writer, set(), ["a", "b"], seed=3)
    again = hwd.select_reference(by_writer, set(), ["a", "b"], seed=3)
    other = hwd.select_reference(by_writer, set(), ["a", "b"], seed=4)

    assert first == again
    assert first.keys != other.keys


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_a_writer_is_the_mean_of_every_row_attributed_to_them():
    means = hwd.writer_means(np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]]), ["a", "a", "b"])

    np.testing.assert_allclose(means["a"], [2.0, 0.0])
    np.testing.assert_allclose(means["b"], [0.0, 5.0])


def test_hwd_is_the_euclidean_distance_between_means_averaged_over_writers():
    side = {"a": np.array([3.0, 4.0]), "b": np.array([0.0, 0.0])}
    reference = {"a": np.zeros(2), "b": np.array([0.0, 1.0])}

    result = hwd.distance(side, reference)

    assert result.writers == ["a", "b"]
    np.testing.assert_allclose(result.per_writer, [5.0, 1.0])
    assert result.value == pytest.approx(3.0)


def test_a_writer_the_reference_does_not_hold_is_an_error():
    with pytest.raises(ValueError, match="no reference lines"):
        hwd.distance({"a": np.zeros(2)}, {"b": np.zeros(2)})


def test_the_interval_resamples_writers_and_brackets_the_figure():
    spread = hwd.Distance(writers=list("abcd"), per_writer=np.array([0.5, 0.7, 0.9, 1.1]))
    flat = hwd.Distance(writers=list("abc"), per_writer=np.array([0.8, 0.8, 0.8]))

    interval = spread.interval()
    assert interval.low <= spread.value <= interval.high
    assert interval.low < interval.high
    assert flat.interval().half_width == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The three sets
# ---------------------------------------------------------------------------


def test_three_sets_are_scored_against_one_reference_extracted_once():
    ids = ["a", "a", "b"]
    real = [_line(100), _line(100), _line(60)]
    generated = [_line(120), _line(120), _line(80)]
    typeface = [_line(250), _line(250), _line(250)]
    extractor = TwoNumberExtractor()

    result = hwd.measure(
        generated, real, typeface, ids, [_line(100), _line(60)], ["a", "b"], extractor=extractor
    )

    assert extractor.calls == 4, "the reference once, then one pass per set"
    assert result.real.value == pytest.approx(0.0)
    assert 0 < result.generated.value < result.typeface.value
    assert result.position == pytest.approx(result.generated.value / result.typeface.value)
    assert (result.scored, result.withheld_samples, result.reference_lines) == (3, 0, 2)


def test_a_writer_outside_the_reference_leaves_all_three_sets_alike():
    """Dropping a writer from one set and not the others would compare different
    people and call the difference style."""
    ids = ["a", "b", "c"]
    images = [_line(100), _line(60), _line(30)]

    result = hwd.measure(
        images, images, images, ids, [_line(100), _line(60)], ["a", "b"], TwoNumberExtractor()
    )

    assert (result.scored, result.withheld_samples) == (2, 1)
    assert result.generated.writers == result.real.writers == result.typeface.writers == ["a", "b"]


def test_misaligned_sets_are_refused():
    with pytest.raises(ValueError, match="generated images"):
        hwd.measure(
            [_line(1)], [_line(1)] * 2, [_line(1)] * 2, ["a", "a"], [], [], TwoNumberExtractor()
        )


def test_without_the_package_it_reports_not_measured(monkeypatch):
    monkeypatch.setattr(hwd, "available", lambda: False)

    assert hwd.measure([], [], [], [], [], []) is None
    assert "not measured" in hwd.describe(None)


def test_position_is_undefined_when_the_typeface_does_not_score_worse_than_real():
    same = hwd.Distance(writers=["a"], per_writer=np.array([1.0]))
    result = hwd.HwdResult(same, same, same, reference_lines=1, scored=1, withheld_samples=0)

    assert math.isnan(result.position)
