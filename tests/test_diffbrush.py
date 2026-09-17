"""Tests for the DiffBrush adapter, without DiffBrush.

The model is 1.2 GB and 42 seconds a line on CPU, so it is exercised by a real
run, not here. What is tested is everything the adapter decides for itself:
how text becomes glyphs, how a style line is fitted to the model's canvas, and
how the fixed 1024px output is cut back to the line that was written.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from nib.models.diffbrush import (
    CANVAS_WIDTH,
    CHARSET,
    crop_output,
    encode_text,
    load_glyphs,
    prepare_style,
)
from nib.models.generator import GeneratorError


def fake_glyphs() -> np.ndarray:
    """One 16x16 glyph per charset character, each filled with its own index."""
    return np.stack([np.full((16, 16), i / 100, np.float32) for i in range(len(CHARSET))])


# ---------------------------------------------------------------------------
# text to glyphs
# ---------------------------------------------------------------------------


def test_each_character_becomes_its_inverted_glyph_in_order():
    glyphs = fake_glyphs()

    encoded = encode_text("ab", glyphs)

    assert encoded.shape == (2, 16, 16)
    np.testing.assert_allclose(encoded[0], 1.0 - glyphs[CHARSET.index("a")])
    np.testing.assert_allclose(encoded[1], 1.0 - glyphs[CHARSET.index("b")])


def test_a_character_outside_the_charset_is_refused_by_name():
    with pytest.raises(GeneratorError, match="'@'"):
        encode_text("mail @ home", fake_glyphs())


def test_glyphs_are_loaded_in_charset_order(tmp_path):
    table = [
        {"idx": [ord(char)], "mat": np.full((16, 16), ord(char), np.float64)} for char in CHARSET
    ]
    path = tmp_path / "unifont.pickle"
    path.write_bytes(pickle.dumps(list(reversed(table))))

    glyphs = load_glyphs(path)

    assert glyphs.shape == (len(CHARSET), 16, 16)
    assert glyphs.dtype == np.float32
    assert glyphs[CHARSET.index("Q"), 0, 0] == ord("Q")


# ---------------------------------------------------------------------------
# the style line
# ---------------------------------------------------------------------------


def test_a_style_line_at_the_native_height_is_scaled_to_zero_one():
    image = np.full((64, 800), 255, np.uint8)
    image[20:40, 100:200] = 0

    style = prepare_style(image)

    assert style.shape == (64, 800)
    assert style.dtype == np.float32
    assert style.min() == 0.0 and style.max() == 1.0


def test_a_style_line_wider_than_the_canvas_keeps_its_first_1024_pixels():
    image = np.full((64, 1500), 255, np.uint8)
    image[:, 1000:1010] = 0

    style = prepare_style(image)

    assert style.shape == (64, CANVAS_WIDTH)
    assert style[:, 1000:1010].max() == 0.0


def test_a_taller_style_line_is_resized_to_the_native_height_keeping_its_aspect():
    image = np.full((128, 1000), 255, np.uint8)

    assert prepare_style(image).shape == (64, 500)


def test_a_colour_style_line_is_refused():
    with pytest.raises(GeneratorError, match="grayscale"):
        prepare_style(np.full((64, 800, 3), 255, np.uint8))


# ---------------------------------------------------------------------------
# the output canvas
# ---------------------------------------------------------------------------


def test_the_canvas_is_cut_a_margin_after_the_last_ink():
    canvas = np.full((64, CANVAS_WIDTH), 255, np.uint8)
    canvas[20:40, 10:600] = 0

    image, truncated = crop_output(canvas, margin=8)

    assert image.shape == (64, 608)
    assert not truncated


def test_ink_reaching_the_edge_of_the_canvas_is_reported_as_truncated():
    canvas = np.full((64, CANVAS_WIDTH), 255, np.uint8)
    canvas[20:40, 10:CANVAS_WIDTH] = 0

    image, truncated = crop_output(canvas)

    assert image.shape == (64, CANVAS_WIDTH)
    assert truncated


def test_faint_grey_is_not_mistaken_for_ink():
    canvas = np.full((64, CANVAS_WIDTH), 255, np.uint8)
    canvas[20:40, 10:300] = 0
    canvas[30, 900] = 200  # decoder noise, not a stroke

    image, _ = crop_output(canvas, margin=8)

    assert image.shape == (64, 308)


def test_a_canvas_with_no_ink_is_no_line():
    assert crop_output(np.full((64, CANVAS_WIDTH), 255, np.uint8)) is None
