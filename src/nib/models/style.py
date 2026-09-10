"""Several real lines as one style reference.

The project's claim is that a hand can be learned from a page. What it has been
doing is showing the model a single line and hoping. Writer retrieval sits at
22.1% against a ceiling of 85.8%, and this is the largest thing never tried
against that number: a person asked to imitate handwriting from one line would
do worse than one given four, and there is no reason a model should differ.

Neither Emuru nor Eruku takes a list of style images. Both take one -- and one
*wide* image holding several lines side by side is still one image. Nothing in
either model constrains the width, and Emuru's own encoder measures the prefix
in eight-pixel slices however many there are.

Two details decide whether this helps or hurts.

**The gap.** Butting two lines together makes the last word of one and the first
of the next read as a single word, and teaches the model a letter join that the
writer never made. The separator is paper-coloured and about as wide as the
space between words at this height.

**The text follows the image.** Both models are told what the style sample says.
If the images are joined and the transcriptions are not, the model is reading
one line while looking at four, which is worse than showing it one line.

What this costs: a longer prefix for the model to encode, and -- for Emuru
specifically -- more canvas for its stopping heuristic to scan, which is where
that heuristic misfires. Both are measured rather than assumed; see the
`--style-refs` results in `PROGRESS.md`.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

GAP_RATIO = 0.35
"""Separator width as a fraction of image height. At 64px that is 22px, which is
about one word space in CVL's hand -- wide enough to read as a break, narrow
enough not to look like the end of a paragraph."""

PAPER = 255
"""The separator's colour. Style images arrive normalised, with paper at the top
of the range, so anything darker would read as ink the writer never put down."""


def join_style(
    images: Sequence[np.ndarray],
    texts: Sequence[str] | None = None,
    gap_ratio: float = GAP_RATIO,
) -> tuple[np.ndarray, str | None]:
    """Lay several style samples side by side, and join their texts to match.

    Returns the combined image and the combined transcription, or None for the
    transcription when none was given -- Eruku does not require one.

    A single image passes through untouched rather than being copied through the
    joining path, so the one-reference case stays exactly what it was and any
    change measured against it is a change in the references and not in the
    plumbing.
    """
    if not images:
        raise ValueError("no style images to join")

    if texts is not None and len(texts) != len(images):
        raise ValueError(f"{len(texts)} style texts for {len(images)} style images")

    if len(images) == 1:
        return np.asarray(images[0]), (texts[0] if texts else None)

    height = max(int(np.asarray(image).shape[0]) for image in images)
    gap = np.full((height, max(1, round(height * gap_ratio))), PAPER, dtype=np.uint8)

    pieces: list[np.ndarray] = []
    for index, image in enumerate(images):
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim != 2:
            raise ValueError(f"style image {index} is not grayscale: {array.shape}")
        if array.shape[0] != height:
            scale = height / array.shape[0]
            array = cv2.resize(
                array,
                (max(1, round(array.shape[1] * scale)), height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
        if pieces:
            pieces.append(gap)
        pieces.append(array)

    joined = np.concatenate(pieces, axis=1)
    return joined, (" ".join(texts) if texts else None)
