"""Tests for the judge that asks whether generated lines can be told from real ones.

No model and no data: the lines are drawn here, so each test knows exactly what
separates its two sets -- or that nothing does.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.engine.metrics import detectability


def _line(rng, ink=40, spread=40, height=(12, 52), width=600, words=6):
    """A line of blocky "words": ink of a chosen darkness and tonal spread."""
    image = np.full((64, width), 255, dtype=np.uint8)
    x = 10
    for _ in range(words):
        w = int(rng.integers(40, 80))
        top = height[0] + int(rng.integers(-2, 3))
        bottom = height[1] + int(rng.integers(-2, 3))
        tone = rng.normal(ink, spread / 2, size=(bottom - top, w))
        image[top:bottom, x : x + w] = np.clip(tone, 0, 150).astype(np.uint8)
        x += w + int(rng.integers(15, 30))
    return image


def test_the_ten_statistics_of_a_known_line():
    image = np.full((64, 200), 255, dtype=np.uint8)
    image[16:48, 50:150] = 55  # one solid block: 32 rows by 100 columns, darkness 200

    values = dict(zip(detectability.NAMES, detectability.features(image), strict=True))

    assert values["ink_height"] == pytest.approx(32 / 64)
    assert values["ink_per_column"] == pytest.approx(32)
    assert values["pieces_per_100_columns"] == pytest.approx(1.0)
    assert values["darkness"] == pytest.approx(200)
    assert values["darkness_spread"] == pytest.approx(0)
    assert values["blank_column_share"] == pytest.approx(0)
    assert values["baseline_wobble"] == pytest.approx(0)


def test_a_line_is_measured_at_the_same_height_whatever_its_size():
    """Most statistics are in pixels, so a line at twice the height must not read
    as having twice the ink."""
    small = np.full((64, 200), 255, dtype=np.uint8)
    small[16:48, 50:150] = 55
    large = np.full((128, 400), 255, dtype=np.uint8)
    large[32:96, 100:300] = 55

    assert detectability.features(large) == pytest.approx(detectability.features(small), rel=0.05)


def test_a_line_with_no_ink_cannot_be_measured():
    assert detectability.features(np.full((64, 300), 255, dtype=np.uint8)) is None


def test_two_sets_drawn_alike_cannot_be_told_apart():
    rng = np.random.default_rng(0)
    real = [_line(rng) for _ in range(60)]
    generated = [_line(rng) for _ in range(60)]

    result = detectability.measure(real, generated)

    assert 0.3 < result.accuracy.value < 0.7


def test_flatter_ink_gives_itself_away_and_is_named():
    """The measured failure: generated ink is flatter in tone than a pen's."""
    rng = np.random.default_rng(1)
    real = [_line(rng, spread=60) for _ in range(60)]
    generated = [_line(rng, spread=10) for _ in range(60)]

    result = detectability.measure(real, generated)

    assert result.accuracy.value > 0.9
    assert max(result.alone, key=result.alone.get) == "darkness_spread"
    assert result.real["darkness_spread"] > result.generated["darkness_spread"]


def test_lines_without_ink_are_skipped_and_counted():
    rng = np.random.default_rng(2)
    blank = np.full((64, 300), 255, dtype=np.uint8)
    real = [_line(rng) for _ in range(20)]
    generated = [_line(rng) for _ in range(20)] + [blank, blank]

    result = detectability.measure(real, generated)

    assert result.skipped == 2
    assert result.lines == 40


def test_too_few_lines_on_a_side_is_refused():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="at least"):
        detectability.measure([_line(rng) for _ in range(20)], [_line(rng) for _ in range(3)])


def test_the_report_names_the_goal():
    rng = np.random.default_rng(4)
    result = detectability.measure(
        [_line(rng) for _ in range(20)], [_line(rng, spread=5) for _ in range(20)]
    )

    assert "50% = cannot be told apart" in result.describe()
