"""Fit the ink of what the model wrote to the hand it copies, from that hand's own page.

Emuru writes in its own ink, flatter in tone than a pen's: the spread of ink
darkness alone told its lines from real ones 93% of the time on cell 7e. So the
darkness of the writer's style lines -- measured on their page and nothing else --
is imposed on what the model wrote for them, quantile by quantile. Measured on
7e's 150 lines, judged against target lines neither the model nor this module saw:
the detectability judge falls from 99.3% to 91.6%, and HWD identity is unmoved
(+0.3 points [-0.2, +0.8], paired by writer).

**Tone only, on purpose.** This module also matched size (crop to the ink, scale
to the hand's ink height) and width (squeeze to the hand's cost per character).
Together they took the judge to 74.9% and **cost 6.8 identity points [-9.3,
-4.2]**; size alone 5.6, width with tone 4.4 (PROGRESS.md, 2026-09-22). They were
removed. Size on a real page belongs to the page engine, in page units.
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


@dataclass(frozen=True)
class Hand:
    """How dark a writer's ink is, from their own lines."""

    height: int
    """The line height the darkness was measured at."""

    darkness: np.ndarray
    """The ink's darkness (255 minus the pixel) at :data:`QUANTILES`."""


def measure_hand(style_images: Sequence[np.ndarray], height: int = 64) -> Hand:
    """Measure a hand's ink from its style lines."""
    darkness = []
    for image in style_images:
        gray = _at_height(image, height)
        if (gray < INK).any():
            darkness.append(PAPER - gray[gray < INK].astype(np.float64))
    if not darkness:
        raise ValueError("no style line has ink to measure")
    return Hand(height=height, darkness=np.quantile(np.concatenate(darkness), QUANTILES))


def calibrate(image: np.ndarray, hand: Hand) -> np.ndarray:
    """``image`` -- a line the model wrote -- in the ink of ``hand``.

    Paper stays paper and the pale fringe of a stroke stays pale: the map runs
    from no darkness at all, through the ink's own quantiles, to the hand's. A
    line with no ink comes back as it was.
    """
    gray = _at_height(image, hand.height)
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
