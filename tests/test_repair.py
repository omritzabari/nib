"""Tests for the stroke repair: it keeps the line's shape and can only add ink."""

from __future__ import annotations

import numpy as np
import pytest

from nib.models import repair

torch = pytest.importorskip("torch")


def _line(width=101):
    rng = np.random.default_rng(0)
    image = np.full((64, width), 255, dtype=np.uint8)
    image[20:44, 10 : width - 10] = rng.integers(0, 200, (24, width - 20))
    return image


def test_a_line_of_any_width_comes_back_the_same_shape():
    torch.manual_seed(0)
    network = repair.build_network()

    out = repair.repair(network, _line(101))

    assert out.shape == (64, 101) and out.dtype == np.uint8


def test_it_can_only_add_ink_never_erase_it():
    """Letterforms stay the model's: nothing may turn lighter."""
    torch.manual_seed(0)
    network = repair.build_network()
    line = _line(96)

    out = repair.repair(network, line)

    assert (out.astype(int) <= line.astype(int) + 1).all()  # +1: rounding


def test_the_breaking_cuts_pieces_out_and_leaves_the_rest_of_the_line():
    """Emuru drops pieces of a stroke; it does not fade the whole line."""
    field = repair.weakening((1, 1, 8, 400), np.random.default_rng(3))

    assert field.shape == (1, 1, 8, 400) and field.dtype == np.float32
    assert field.min() >= 0.0 and field.max() <= 1.0
    assert (field >= repair.LEVEL[0]).mean() > 0.5, "most of the line survives"
    assert (field <= repair.DEPTH[1]).any(), "and some of it is cut"
