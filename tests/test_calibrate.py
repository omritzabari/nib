"""Tests for fitting a generated line's ink to the hand it copies.

No model: lines are drawn here with a known ink, so each test knows what the
calibrated line should come out as.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.inference.calibrate import INK, calibrate, measure_hand


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


def _hand(**kwargs):
    lines = [_line(1, 63, seed=s, **kwargs) for s in range(4)]
    return measure_hand(lines)


def test_a_hand_is_measured_from_its_lines():
    hand = _hand(tone=(40, 80))

    assert 255 - 80 <= hand.darkness[0] <= hand.darkness[-1] <= 255 - 40


def test_the_line_keeps_its_shape():
    """Size and width were measured to cost identity and were removed: only ink moves."""
    written = _line(12, 52, tone=(100, 150))

    out = calibrate(written, _hand(tone=(0, 30)))

    assert out.shape == written.shape
    assert np.array_equal(out < 255, written < 255)


def test_tone_takes_the_hands_darkness():
    hand = _hand(tone=(0, 30))  # a dark pen
    pale = _line(1, 63, tone=(100, 150))

    out = calibrate(pale, hand)

    ink = out[out < INK]
    assert 255 - ink.mean() == pytest.approx(255 - 15, abs=6)


def test_the_tone_map_leaves_paper_as_paper():
    """Darkening the ink must not grey the page: every white pixel stays white."""
    written, hand = _line(10, 50, tone=(100, 150)), _hand(tone=(0, 30))

    toned = calibrate(written, hand)

    assert np.array_equal(written == 255, toned == 255)


def test_a_line_with_no_ink_comes_back_as_it_was():
    blank = np.full((64, 300), 255, dtype=np.uint8)

    assert np.array_equal(calibrate(blank, _hand()), blank)


def test_a_hand_needs_something_to_measure():
    blank = np.full((64, 300), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="no style line"):
        measure_hand([blank])


def test_haze_on_the_paper_is_made_white_and_the_stroke_edges_kept():
    written = _line(20, 44, tone=(100, 150))
    written[5, 300:320] = 235  # a faint grey mark, far from any stroke
    written[19, 12:20] = 200  # a stroke's soft edge, touching it

    out = calibrate(written, _hand())

    assert (out[5, 300:320] == 255).all()
    assert (out[19, 12:20] < 255).all()
