"""Tests for splitting a page into lines.

Synthetic pages first, each isolating one thing the splitter must get right, and
then Amri's own page photographed five ways: the same page under different
conditions must give the same lines.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from nib.config import find_repo_root
from nib.data.preprocessing import normalise_page
from nib.data.segmentation import DEFAULT, split_lines

PHOTOS = find_repo_root() / "data" / "raw" / "personal"
_REAL = [p for p in sorted(PHOTOS.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
needs_photos = pytest.mark.skipif(len(_REAL) < 3, reason=f"fewer than 3 photographs under {PHOTOS}")

PAGE1_LINES = 13
"""Counted on the photographs by eye: ten lines of sentences, one of digits, and
the alphabet over two."""


def written_page(texts, pitch=60, width=900, top=80):
    """Text on squared paper whose grid is lighter than the ink threshold, as a
    normalised photograph's is. Returns the page and each line's baseline."""
    height = top + pitch * len(texts) + 160
    page = np.full((height, width), 255, dtype=np.uint8)
    for y in range(0, height, 20):
        cv2.line(page, (0, y), (width, y), 170, 1)
    for x in range(0, width, 20):
        cv2.line(page, (x, 0), (x, height), 170, 1)
    baselines = []
    for number, text in enumerate(texts):
        baseline = top + number * pitch
        cv2.putText(page, text, (40, baseline), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
        baselines.append(baseline)
    return page, baselines


TEXTS = ["hello world", "the quick brown", "jumping gypsy", "last line here"]


def test_each_line_of_text_becomes_one_crop_in_reading_order():
    page, baselines = written_page(TEXTS)

    result = split_lines(page)

    assert len(result.lines) == len(TEXTS)
    for line, baseline in zip(result.lines, baselines, strict=True):
        _, y, _, h = line.box
        assert y <= baseline <= y + h
    tops = [line.box[1] for line in result.lines]
    assert tops == sorted(tops)


def test_margin_line_punch_hole_and_specks_leave_no_trace():
    """The margin line is a straight run and is cut from the ink before anything
    else; the hole is a blob; the specks are specks. None of them becomes a line."""
    page, _ = written_page(TEXTS)
    cv2.line(page, (820, 0), (820, page.shape[0]), 0, 2)
    cv2.circle(page, (700, 300), 18, 0, -1)
    for x in (550, 600, 650):
        page[400:402, x : x + 2] = 0

    result = split_lines(page)

    assert len(result.lines) == len(TEXTS)
    assert result.dropped["blob"] >= 1
    assert result.dropped["speck"] >= 3
    assert all(x + w < 690 for x, _, w, _ in (line.box for line in result.lines))


def test_a_long_rule_that_escapes_the_straight_run_cut_is_dropped_as_ruling():
    """Tilted just enough that no row holds a run long enough to cut, as a
    photographed rule often is."""
    page, _ = written_page(TEXTS)
    cv2.line(page, (380, 420), (880, 455), 0, 2)

    result = split_lines(page)

    assert len(result.lines) == len(TEXTS)
    assert result.dropped["ruling"] >= 1


def test_a_mark_with_nothing_beside_it_is_not_a_line():
    page, _ = written_page(TEXTS)
    cv2.putText(page, "x x", (560, 400), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 3, cv2.LINE_AA)

    result = split_lines(page)

    assert len(result.lines) == len(TEXTS)
    assert result.dropped["sparse"] >= 1


def test_a_crop_holds_only_its_own_strokes_when_lines_nearly_touch():
    """Descenders hang into the line below; a crop copies its own components only,
    so the border it was padded with stays blank."""
    page, _ = written_page(["gggyyyjjj", "fffllkkhh", "gyjgyjgyj"], pitch=40)

    result = split_lines(page)

    assert len(result.lines) == 3
    clear = DEFAULT.padding - DEFAULT.stroke_margin
    for line in result.lines:
        assert (line.image[:clear] == 255).all()
        assert (line.image[-clear:] == 255).all()


def test_a_page_with_nothing_written_has_no_lines():
    page, _ = written_page([])

    result = split_lines(page)

    assert result.lines == []


def test_a_colour_image_is_refused():
    with pytest.raises(ValueError, match="grayscale"):
        split_lines(np.full((50, 50, 3), 255, dtype=np.uint8))


def _photos():
    for path in _REAL:
        marks = []
        if path.stem == "dim":
            marks.append(
                pytest.mark.xfail(
                    reason="find_page returns the whole frame for this photo, so the black "
                    "cloth above the sheet reaches segmentation -- a page-detection fault",
                    strict=True,
                )
            )
        yield pytest.param(path, marks=marks, id=path.stem[:12])


@needs_photos
@pytest.mark.parametrize("path", list(_photos()))
def test_amris_page_gives_the_same_lines_under_every_condition(path):
    page = normalise_page(cv2.imread(str(path)))

    result = split_lines(page)

    assert len(result.lines) == PAGE1_LINES, result.dropped
