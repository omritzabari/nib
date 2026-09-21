"""Tests for fitting a generated line to the hand it copies, from that hand's lines.

No model: lines are drawn here, with a known size, spacing and ink, so each test
knows what the calibrated line should come out as.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.inference.calibrate import INK, WIDTH_LIMITS, calibrate, measure_hand


def _line(ink_top, ink_bottom, width=400, chars=20, char_width=None, tone=(40, 80), seed=0):
    """A line whose ink is a band of blocks, one per character, of a given tone range."""
    rng = np.random.default_rng(seed)
    image = np.full((64, width), 255, dtype=np.uint8)
    step = char_width or (width - 20) // chars
    for k in range(chars):
        left = 10 + k * step
        block = rng.uniform(*tone, size=(ink_bottom - ink_top, max(1, step - 4)))
        image[ink_top:ink_bottom, left : left + max(1, step - 4)] = block.astype(np.uint8)
    return image


def _box(image):
    ink = image < INK
    rows = np.nonzero(ink.any(axis=1))[0]
    cols = np.nonzero(ink.any(axis=0))[0]
    return rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1


def _hand(**kwargs):
    lines = [_line(1, 63, seed=s, **kwargs) for s in range(4)]
    return measure_hand(lines, ["x" * 20] * 4)


def test_a_hand_is_measured_from_its_lines():
    hand = _hand()

    assert hand.ink_height == pytest.approx(62)
    assert hand.top == pytest.approx(1)
    assert hand.bottom == pytest.approx(1)
    assert hand.width_per_character > 0
    assert hand.darkness[0] <= hand.darkness[-1]


def test_size_brings_the_ink_to_the_hands_height():
    """The model writes inside a band with margins; the hand's lines fill theirs."""
    written = _line(12, 52)  # 40 px of ink where the hand has 62

    out = calibrate(written, "x" * 20, _hand(), width=False, tone=False)

    assert out.shape[0] == 64
    assert _box(out)[0] == pytest.approx(62, abs=1)


def test_width_takes_the_hands_cost_per_character():
    hand = _hand()
    wide = _line(1, 63, width=600, char_width=29)  # the same 20 characters, half as wide again

    out = calibrate(wide, "x" * 20, hand, size=False, tone=False)

    assert _box(out)[1] < _box(wide)[1]


def test_width_never_squeezes_past_its_limit():
    """A prediction is an average; a line far off it is more likely unusual than wrong."""
    hand = _hand()
    very_wide = _line(1, 63, width=1500, char_width=74)

    out = calibrate(very_wide, "x" * 20, hand, size=False, tone=False)

    assert _box(out)[1] >= _box(very_wide)[1] * WIDTH_LIMITS[0] - 2


def test_tone_takes_the_hands_darkness():
    hand = _hand(tone=(0, 30))  # a dark pen
    pale = _line(1, 63, tone=(100, 150))

    out = calibrate(pale, "x" * 20, hand, size=False, width=False)

    ink = out[out < INK]
    assert 255 - ink.mean() == pytest.approx(255 - 15, abs=6)


def test_the_tone_map_leaves_paper_as_paper():
    """Darkening the ink must not grey the page: every white pixel stays white."""
    written, hand = _line(10, 50, tone=(100, 150)), _hand(tone=(0, 30))

    shaped = calibrate(written, "x" * 20, hand, tone=False)
    toned = calibrate(written, "x" * 20, hand, tone=True)

    assert np.array_equal(shaped == 255, toned == 255)


def test_a_line_with_no_ink_comes_back_as_it_was():
    blank = np.full((64, 300), 255, dtype=np.uint8)

    assert np.array_equal(calibrate(blank, "anything", _hand()), blank)


def test_a_hand_needs_something_to_measure():
    blank = np.full((64, 300), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="no style line"):
        measure_hand([blank], ["text"])
