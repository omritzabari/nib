"""Tests for HWD's arithmetic and its reference set.

None of these load the VGG or need the `hwd` package: the network sits behind an
extractor protocol, and a stand-in that summarises an image by two numbers is
enough to check that the right things are compared with the right things.
"""

from __future__ import annotations

import importlib.util
import math
import os
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


def test_every_writer_is_also_measured_against_every_other_writers_reference():
    side = {"a": np.array([0.0, 0.0]), "b": np.array([10.0, 0.0])}
    reference = {"a": np.array([0.0, 0.0]), "b": np.array([10.0, 0.0]), "c": np.array([0.0, 10.0])}

    result = hwd.distance(side, reference)

    np.testing.assert_allclose(result.per_writer, [0.0, 0.0])
    np.testing.assert_allclose(result.wrong, [10.0, (10.0 + math.hypot(10.0, 10.0)) / 2])
    np.testing.assert_allclose(result.gap, result.wrong)
    assert result.nearest_is_right.tolist() == [True, True]


def test_a_writer_the_reference_does_not_hold_is_an_error():
    with pytest.raises(ValueError, match="no reference lines"):
        hwd.distance({"a": np.zeros(2)}, {"b": np.zeros(2)})


def _distance(per_writer: list[float]) -> hwd.Distance:
    values = np.array(per_writer)
    return hwd.Distance(
        writers=[str(i) for i in range(len(values))],
        per_writer=values,
        wrong=values + 1.0,
        nearest_is_right=np.ones(len(values), dtype=bool),
    )


def test_the_interval_resamples_writers_and_brackets_the_figure():
    spread = _distance([0.5, 0.7, 0.9, 1.1])
    flat = _distance([0.8, 0.8, 0.8])

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


def test_where_the_package_is_installed_it_loads_even_without_editdistance():
    """editdistance has no Windows wheel for Python 3.13; the score never uses it."""
    if importlib.util.find_spec("hwd") is None:
        pytest.skip("the hwd extra is not installed")

    assert hwd.available()
    if os.name == "nt":
        import hwd.metrics.backbones as backbones

        # The package asks for a worker; on Windows it would re-run the caller.
        assert backbones.DataLoader([1, 2], num_workers=1).num_workers == 0


def test_position_is_undefined_when_the_typeface_does_not_score_worse_than_real():
    same = _distance([1.0])
    result = hwd.HwdResult(
        same, same, same, reference_lines=1, scored=1, withheld_samples=0, candidates=1
    )

    assert math.isnan(result.position)


# ---------------------------------------------------------------------------
# Identity: the right writer against everyone else
# ---------------------------------------------------------------------------


def test_identity_is_one_for_the_writers_own_hand_and_zero_for_nobodys():
    """A copy of the real lines carries all the identity they do. Output that is
    the same for every writer -- the typeface is -- is exactly as far from the
    right writer as from the wrong one on average, so it carries none."""
    ids = ["a", "b"]
    real = [_line(100), _line(60)]
    nobody = [_line(250), _line(250)]
    extractor = TwoNumberExtractor()

    copy = hwd.measure(real, real, nobody, ids, real, ids, extractor=extractor)
    generic = hwd.measure(nobody, real, nobody, ids, real, ids, extractor=extractor)

    assert copy.identity == pytest.approx(1.0)
    assert generic.identity == pytest.approx(0.0, abs=1e-12)
    assert float(np.mean(copy.typeface.gap)) == pytest.approx(0.0, abs=1e-12)
    assert copy.real.nearest_is_right.all()
    assert copy.chance == pytest.approx(0.5)
    assert "writer identity" in hwd.describe(copy)


def test_identity_is_undefined_when_real_lines_are_not_closer_to_their_own_writer():
    flat = hwd.Distance(
        writers=["a", "b"],
        per_writer=np.array([1.0, 1.0]),
        wrong=np.array([1.0, 1.0]),
        nearest_is_right=np.array([False, False]),
    )
    result = hwd.HwdResult(
        flat, flat, flat, reference_lines=2, scored=2, withheld_samples=0, candidates=2
    )

    assert math.isnan(result.identity)


def test_identity_refuses_sets_scored_over_different_writers():
    first = _distance([1.0, 2.0])
    other = hwd.Distance(
        writers=["x", "y"],
        per_writer=first.per_writer,
        wrong=first.wrong,
        nearest_is_right=first.nearest_is_right,
    )
    result = hwd.HwdResult(
        first, other, first, reference_lines=2, scored=2, withheld_samples=0, candidates=2
    )

    with pytest.raises(ValueError, match="different writers"):
        _ = result.identity
