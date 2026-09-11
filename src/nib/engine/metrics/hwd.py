"""Handwriting Distance: the field's style metric, and the one ours failed to be.

This project's own style measure -- writer retrieval against a trained embedding
-- turned out to be measuring image quality. A *real* line, by unquestionably
the right writer, blurred by 0.8 pixels, scores 12.2% where the untouched line
scores 96.8%. Blur does not change whose handwriting something is, and every
generative model's decoder produces exactly that softness, so the number the
project treated as its headline was substantially a sharpness score.

HWD (Pippi et al., BMVC 2023) is a VGG16 trained on 100 million rendered text
lines to classify calligraphic fonts, used as a feature extractor for a distance
between two sets of handwriting. Measured here on the same damage:

    reference against         ours          HWD
    other real lines          96.8%        0.641
    the same lines, blurred   12.2%        0.721
    the same lines, resampled 10.1%        0.636
    the same text in a font      --        2.931

It barely moves under damage that keeps the hand, and it separates handwriting
from a typeface by a factor of four and a half. That is a metric of style.

It also comes from the group that built Emuru and Eruku and is what their papers
report, so a figure produced here can be set beside a published one -- which the
retrieval percentage never could.

**It is per-writer.** ``FolderDataset`` reads each image's author from its parent
directory's name, so both sides must be filed by writer under the same names,
and the comparison is then each writer's samples against that writer's own hand.
That is the right question, and it is why :func:`compute_hwd` takes writer ids
rather than two flat lists.

**Installing it.** `hwd` is a research package that imports every score it owns
at package level, so it drags in gudhi, matplotlib, tiktoken and more; and it
depends on `editdistance`, which has no wheel for Python 3.13 on Windows. It is
therefore an optional extra, and everything here degrades to "not measured"
rather than failing when it is absent.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

DEFAULT_HEIGHT = 32
"""The height HWD's own examples use, and its default. Ours are 64px; the
package resizes, and matching its default keeps our figures on the same scale as
published ones."""

REAL_FLOOR = 0.641
"""Two disjoint sets of real CVL lines by the same 20 writers, measured here.

The floor, in the same sense as the FID floor: what the metric gives when both
sides genuinely are that person's hand. A generated set is read as a distance
from this, not from zero."""

TYPEFACE_CEILING = 2.931
"""The same texts drawn in a Hershey font. No style at all, and the far end of
the scale -- without it, a bare HWD figure has no size."""


def available() -> bool:
    """Whether the optional package is installed and importable."""
    try:
        import hwd.scores  # noqa: F401
    except Exception:
        return False
    return True


def _write_by_writer(
    images: Sequence[np.ndarray], writer_ids: Sequence[str], directory: Path
) -> None:
    """One subdirectory per writer, because that is where HWD reads identity."""
    for index, (image, writer) in enumerate(zip(images, writer_ids, strict=True)):
        folder = directory / str(writer)
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{index:05d}.png"), np.asarray(image, dtype=np.uint8))


def compute_hwd(
    generated: Sequence[np.ndarray],
    real: Sequence[np.ndarray],
    writer_ids: Sequence[str],
    height: int = DEFAULT_HEIGHT,
) -> float | None:
    """Style distance between generated and real handwriting, per writer.

    ``writer_ids`` labels both sides: entry *i* of ``generated`` and entry *i* of
    ``real`` are both attributed to ``writer_ids[i]``, which is exactly the
    pairing the evaluation already maintains -- each generated line has a real
    line of the same text by the same hand.

    Returns None when the package is not installed, so a run without it reports
    one metric fewer rather than failing.
    """
    if not available():
        return None

    from hwd.datasets import FolderDataset
    from hwd.scores import HWDScore

    root = Path(tempfile.mkdtemp(prefix="nib_hwd_"))
    try:
        _write_by_writer(generated, writer_ids, root / "generated")
        _write_by_writer(real, writer_ids, root / "real")
        score = HWDScore(height=height)
        return float(score(FolderDataset(root / "generated"), FolderDataset(root / "real")))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def describe(value: float | None) -> str:
    """The figure with the two anchors that give it a size."""
    if value is None:
        return "HWD            not measured -- pip install the 'hwd' extra"
    position = (value - REAL_FLOOR) / (TYPEFACE_CEILING - REAL_FLOOR)
    return (
        f"HWD            {value:8.3f}   vs {REAL_FLOOR:.3f} real, "
        f"{TYPEFACE_CEILING:.3f} for a typeface\n"
        f"               {position:8.0%} of the way from real handwriting to no style at all"
    )
