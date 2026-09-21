"""Fit what the model wrote to the hand it was copying, from that hand's own page.

A user's page teaches the system more than the model takes in. Emuru is shown one
line and writes in its manner, but it writes on its own scale and in its own ink:
on the held-out CVL writers its lines stand 52 px of ink tall where the writer's
stand 62.5, it spends 14% more width per character relative to that height, and
its ink is flatter in tone than a pen's. A classifier told real from generated on
99.3% of cell 7e's lines, and those three were most of how.

So the output is adapted to each user, from their page and nothing else: what is
measured on their style lines -- whose transcriptions enrolment always has -- is
imposed on what the model wrote for them. Three operations, each switchable so
each can be judged alone:

    size    crop to the ink, and scale it to the hand's ink height
    width   scale across, so a character costs the width it costs in the hand
    tone    map the ink's darkness onto the hand's, quantile by quantile

**This is calibration, not a better model.** It changes nothing about letterforms,
joins or rhythm, which are what the model got right or wrong. Size is largely
framing -- the pack's lines are cropped to their ink and the model writes inside a
band -- so matching it is what a page needs, not evidence of a better hand. What
the model gets wrong in the strokes themselves -- broken strokes, a line too even
-- is untouched here.

**Fitted on the style lines, judged against other lines.** Everything here is
measured from the lines the model was shown; the judge
(:mod:`nib.engine.metrics.detectability`) compares against the writer's *target*
lines, which neither the model nor this module ever sees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

INK = 160
"""A pixel darker than this is ink -- the threshold the judge uses."""

PAPER = 255

QUANTILES = np.linspace(0.0, 1.0, 21)
"""Where the ink's darkness is sampled, for the tone map."""

WIDTH_LIMITS = (0.8, 1.25)
"""How far the width operation may squeeze or stretch beyond a uniform scale.

The prediction is a hand's average cost per character, and a line of narrow
letters -- "illicit" -- honestly costs less than one of wide ones. Measured on 7e,
the model spends a median 1.14 times the writer's width per character, so the
typical correction is 0.88. Past these limits the line is more likely to be
unusual than wrong."""


@dataclass(frozen=True)
class Hand:
    """What a writer's own lines say about size, spacing and ink."""

    height: int
    """The line height everything was measured at."""

    ink_height: float
    """Median height of the ink, in pixels."""

    top: float
    bottom: float
    left: float
    right: float
    """Median white margins around the ink, in pixels."""

    width_per_character: float
    """Width of the ink per character of the transcription, in units of the ink's
    own height, so it does not depend on the scale a line was measured at."""

    darkness: np.ndarray
    """The ink's darkness (255 minus the pixel) at :data:`QUANTILES`."""


def measure_hand(
    style_images: Sequence[np.ndarray], style_texts: Sequence[str], height: int = 64
) -> Hand:
    """Measure a hand from its style lines and their transcriptions."""
    boxes, darkness, widths = [], [], []
    for image, text in zip(style_images, style_texts, strict=True):
        gray = _at_height(image, height)
        box = _ink_box(gray)
        if box is None or not text.strip():
            continue
        top, bottom, left, right = box
        boxes.append((bottom - top, top, height - bottom, left, gray.shape[1] - right))
        widths.append((right - left) / len(text) / (bottom - top))
        darkness.append(PAPER - gray[gray < INK].astype(np.float64))
    if not boxes:
        raise ValueError("no style line has ink and a transcription to measure")
    ink_height, top, bottom, left, right = np.median(np.asarray(boxes, dtype=np.float64), axis=0)
    return Hand(
        height=height,
        ink_height=float(ink_height),
        top=float(top),
        bottom=float(bottom),
        left=float(left),
        right=float(right),
        width_per_character=float(np.median(widths)),
        darkness=np.quantile(np.concatenate(darkness), QUANTILES),
    )


def calibrate(
    image: np.ndarray,
    text: str,
    hand: Hand,
    size: bool = True,
    width: bool = True,
    tone: bool = True,
) -> np.ndarray:
    """``image`` -- a line the model wrote of ``text`` -- fitted to ``hand``.

    Geometry first, then tone: resampling blends pixels, and the darkness the hand
    is matched to should be the darkness that is finally shown. A line with no ink
    is returned as it came.
    """
    gray = _at_height(image, hand.height)
    box = _ink_box(gray)
    if box is None:
        return gray
    if size or width:
        gray = _fit_geometry(gray, box, text, hand, size=size, width=width)
    if tone:
        gray = _fit_tone(gray, hand)
    return gray


def _fit_geometry(
    gray: np.ndarray, box: tuple[int, int, int, int], text: str, hand: Hand, size: bool, width: bool
) -> np.ndarray:
    top, bottom, left, right = box
    ink = gray[top:bottom, left:right]
    scale = hand.ink_height / ink.shape[0] if size else 1.0
    new_height = max(1, round(ink.shape[0] * scale))
    new_width = ink.shape[1] * scale
    if width and text.strip():
        wanted = hand.width_per_character * len(text) * new_height
        low, high = WIDTH_LIMITS
        new_width = float(np.clip(wanted, new_width * low, new_width * high))
    new_width = max(1, round(new_width))
    grow = new_height > ink.shape[0] or new_width > ink.shape[1]
    ink = cv2.resize(
        ink, (new_width, new_height), interpolation=cv2.INTER_CUBIC if grow else cv2.INTER_AREA
    )

    # Back into a line of the hand's height, with the hand's margins; if the scaled
    # ink and those margins do not fit exactly, the vertical slack is shared in the
    # hand's own proportion.
    slack = hand.height - new_height
    share = hand.top / max(hand.top + hand.bottom, 1e-9)
    above = int(np.clip(round(slack * share), 0, max(slack, 0)))
    below = max(slack - above, 0)
    if slack < 0:  # taller than the line: keep the centre
        cut = -slack // 2
        ink = ink[cut : cut + hand.height]
        above = below = 0
    return cv2.copyMakeBorder(
        ink,
        above,
        below,
        round(hand.left),
        round(hand.right),
        cv2.BORDER_CONSTANT,
        value=PAPER,
    )


def _fit_tone(gray: np.ndarray, hand: Hand) -> np.ndarray:
    """Map the ink's darkness onto the hand's, quantile by quantile.

    Paper stays paper and the pale fringe of a stroke stays pale: the map runs
    from no darkness at all, through the ink's own quantiles, to the hand's.
    """
    darkness = PAPER - gray.astype(np.float64)
    ink = darkness[gray < INK]
    if ink.size == 0:
        return gray
    source = np.quantile(ink, QUANTILES)
    target = hand.darkness
    # Strictly increasing, or interpolation is undefined where the ink is flat.
    source = np.maximum.accumulate(source) + np.arange(len(source)) * 1e-6
    mapped = np.interp(
        darkness,
        np.concatenate([[0.0], source]),
        np.concatenate([[0.0], target]),
        right=target[-1],
    )
    return np.clip(PAPER - mapped, 0, PAPER).astype(np.uint8)


def _ink_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Rows and columns the ink occupies, as [top, bottom) and [left, right)."""
    ink = gray < INK
    rows = np.nonzero(ink.any(axis=1))[0]
    columns = np.nonzero(ink.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1


def _at_height(image: np.ndarray, height: int) -> np.ndarray:
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if gray.shape[0] == height:
        return gray
    scale = height / gray.shape[0]
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(
        gray, (max(1, round(gray.shape[1] * scale)), height), interpolation=interpolation
    )
