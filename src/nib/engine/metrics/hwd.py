"""Handwriting Distance: the field's style metric, and the one ours failed to be.

This project's own style measure -- writer retrieval against a trained embedding
-- turned out to be measuring image quality. A *real* line, by unquestionably
the right writer, blurred by 0.8 pixels, scores 12.2% where the untouched line
scores 96.8%. Blur does not change whose handwriting something is, and every
generative model's decoder produces exactly that softness, so the number the
project treated as its headline was substantially a sharpness score.

HWD (Pippi et al., BMVC 2023) is a VGG16 trained on 100 million rendered text
lines to classify calligraphic fonts, used as a feature extractor. Each image is
resized to 32px high and cut into 32px columns; each column becomes a feature
vector; a writer is summarised by the mean of all their columns. HWD is the
Euclidean distance between a writer's mean on either side, averaged over
writers. Under the blur that costs retrieval 87% of its accuracy it moved by
12%, and it separated handwriting from a typeface by a factor of four and a
half. It is also what Emuru's and Eruku's own papers report.

**The floor is measured inside every run, never carried in as a constant.** A
mean over few lines is noisy, and that noise adds to the distance even when both
sides are the same person's real hand. What real handwriting scores therefore
depends on how many lines each writer contributes, and a run of 300 targets over
94 writers gives each about three. The first floor this project quoted, 0.641,
came from a script that was never committed, with a count per writer nobody
recorded -- which is exactly why it cannot be the yardstick for a run.

Measured on 90 held-out writers, real lines against other real lines by the same
writers, k lines per writer on each side:

    k = 1     1.760
    k = 2     1.255
    k = 3     1.005
    k = 6     0.704
    k = 10    0.539

Under the exact sampling of a 300-sample run, real against real read 1.06 over
three seeds, where 0.641 would have been quoted as the floor: a perfect
generator would have been reported as nearly a fifth of the way to having no
hand at all. Content, by contrast, barely registers -- a typeface scored 3.080
against the target's own text and 3.075 against a different one.

So a run scores three sets, aligned sample for sample, against one shared
reference of real lines that are neither a target nor a style line:

    generated   what the model wrote
    real        the target lines themselves: same writers, same texts, same counts
    typeface    the same texts drawn in a font: no hand at all

Generated and real differ in nothing but being generated, so neither the count
per writer nor the content can tilt the comparison between them, and the
typeface gives the figure its far end on the same terms.

**Distance is not identity.** The typeface anchors *no handwriting*, not
*someone else's handwriting*. A set can sit far from real because it does not
look like real handwriting, or because it does not look like this writer, and
the three figures above cannot say which. So every set is also measured against
every *other* writer's reference. The gap between the two -- how much farther
everyone else is than the right writer -- is zero for output that is nobody's
hand in particular, however blurred or however clean, and positive for output
that is closer to its own writer. The generated gap as a share of the real
lines' gap is the identity figure: 0 for none, 1 for as much as the writer's own
lines carry. Emuru's first measured run, 53% of the way to a typeface on
distance, predates it.

**Installing it.** `hwd` is a research package that imports every score it owns
at package level, so it drags in gudhi, matplotlib, tiktoken and more; and it
depends on `editdistance`, which has no wheel for Python 3.13 on Windows. It is
therefore an optional extra, and everything here degrades to "not measured"
rather than failing when it is absent. Only its IAM loader uses `editdistance`,
so :func:`_load_hwd` stands a blank module in for it, and the score runs on a
Windows laptop too. The network sits behind
:class:`Extractor`, so the arithmetic is tested without it.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import types
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from nib.engine.metrics import bootstrap

DEFAULT_HEIGHT = 32
"""The height HWD's own examples use, and its default. Ours are 64px; the
package resizes, and matching its default keeps our figures on the same scale as
published ones."""

REFERENCE_DEPTH = 12
"""Real lines per writer in the shared reference, where the writer has that many
to spare. A writer's mean steadies as this grows; CVL writers hold about thirty
lines and a run's targets and style lines take a handful. The same depth as the
retrieval gallery."""

REFERENCE_MINIMUM = 6
"""Below this many spare lines a writer is left out of HWD -- from all three sets
alike, so the comparison between them is untouched -- and counted. A mean over
two or three reference lines adds more noise than the writer adds evidence."""


def _load_hwd() -> None:
    """Import the package, past the two things that stop it off Colab.

    ``editdistance`` has no Windows wheel for Python 3.13 and is used only by the
    package's IAM loader, never by the score, so a blank module stands in. And on
    Windows the package's DataLoader worker re-runs the calling script -- the first
    local attempt hung for 13 minutes -- so it loads in the calling process.
    """
    try:
        import editdistance  # noqa: F401
    except ImportError:
        sys.modules["editdistance"] = types.ModuleType("editdistance")
    import hwd.scores  # noqa: F401

    if os.name == "nt":
        import functools

        import hwd.metrics.backbones as backbones
        import torch.utils.data

        backbones.DataLoader = functools.partial(torch.utils.data.DataLoader, num_workers=0)


def available() -> bool:
    """Whether the optional package is installed and importable."""
    try:
        _load_hwd()
    except Exception:
        return False
    return True


class Extractor(Protocol):
    """Anything that summarises each writer's images as one mean feature vector."""

    def __call__(
        self, images: Sequence[np.ndarray], writer_ids: Sequence[str]
    ) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class Reference:
    """The real lines every set is scored against."""

    keys: list[str]
    writer_ids: list[str]
    withheld: list[str]
    """Writers with too few spare lines to be scored, named rather than skipped."""


def select_reference(
    by_writer: Mapping[str, Sequence[str]],
    consumed: Collection[str],
    writers: Iterable[str],
    depth: int = REFERENCE_DEPTH,
    minimum: int = REFERENCE_MINIMUM,
    seed: int = 0,
) -> Reference:
    """Up to ``depth`` real lines per writer, none of them in ``consumed``.

    ``consumed`` is every key the run used as a target or as a style line. A
    target in the reference would be compared with itself; a style line in it
    would favour the model, which was shown that very image.
    """
    rng = random.Random(seed)
    keys: list[str] = []
    ids: list[str] = []
    withheld: list[str] = []
    for writer in sorted(set(writers)):
        spare = sorted(key for key in by_writer.get(writer, ()) if key not in consumed)
        if len(spare) < minimum:
            withheld.append(writer)
            continue
        for key in rng.sample(spare, min(depth, len(spare))):
            keys.append(key)
            ids.append(writer)
    return Reference(keys=keys, writer_ids=ids, withheld=withheld)


def writer_means(features: np.ndarray, authors: Sequence[str]) -> dict[str, np.ndarray]:
    """Each writer's mean over every feature row attributed to them."""
    rows = np.asarray(features, dtype=np.float64)
    labels = np.asarray([str(author) for author in authors])
    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} feature rows for {len(labels)} author labels")
    return {writer: rows[labels == writer].mean(axis=0) for writer in sorted(set(labels.tolist()))}


def _writer_interval(
    statistic: Callable[[np.ndarray], float], count: int, seed: int
) -> bootstrap.Interval:
    """Resampling writers, because every figure here is an average over writers.
    One writer has no spread to resample, so it gets a point."""
    if count < 2:
        value = statistic(np.arange(count))
        return bootstrap.Interval(value, value, value, resamples=0)
    return bootstrap.bootstrap_statistic(statistic, count, seed=seed)


@dataclass(frozen=True)
class Distance:
    """One set's HWD, kept per writer so every figure can carry an interval.

    ``per_writer`` is each writer's distance to their own reference -- the terms
    HWD averages. ``wrong`` is the same writer's mean distance to every *other*
    writer's reference, and ``nearest_is_right`` whether their own reference was
    the closest of all. Those two say whether the set looks like its writer in
    particular, rather than merely like handwriting.
    """

    writers: list[str]
    per_writer: np.ndarray
    wrong: np.ndarray
    nearest_is_right: np.ndarray

    @property
    def value(self) -> float:
        return float(np.mean(self.per_writer))

    @property
    def gap(self) -> np.ndarray:
        """Per writer, how much farther everyone else's reference is than their own."""
        return np.asarray(self.wrong, np.float64) - np.asarray(self.per_writer, np.float64)

    def interval(self, seed: int = 1337) -> bootstrap.Interval:
        """How far HWD moves when a different set of people is drawn."""
        values = np.asarray(self.per_writer, dtype=np.float64)
        return _writer_interval(lambda indices: float(values[indices].mean()), len(values), seed)

    def gap_interval(self, seed: int = 1337) -> bootstrap.Interval:
        gap = self.gap
        return _writer_interval(lambda indices: float(gap[indices].mean()), len(gap), seed)

    def nearest_interval(self, seed: int = 1337) -> bootstrap.Interval:
        hits = np.asarray(self.nearest_is_right, dtype=np.float64)
        return _writer_interval(lambda indices: float(hits[indices].mean()), len(hits), seed)


def distance(side: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]) -> Distance:
    """Each writer's distance to their own reference, and to everyone else's.

    The own-writer distances are the terms HWD averages. The rest cost nothing
    extra -- every mean is already computed -- and are what separates "not this
    writer" from "not real handwriting".
    """
    writers = sorted(side)
    absent = [writer for writer in writers if writer not in reference]
    if absent:
        raise ValueError(f"{len(absent)} writers have no reference lines, e.g. {absent[:3]}")

    candidates = sorted(reference)
    own = np.array([candidates.index(writer) for writer in writers])
    sides = np.stack([np.asarray(side[writer], np.float64) for writer in writers])
    references = np.stack([np.asarray(reference[writer], np.float64) for writer in candidates])
    # Rows are this set's writers, columns every writer in the reference.
    matrix = np.linalg.norm(sides[:, None, :] - references[None, :, :], axis=2)

    right = matrix[np.arange(len(writers)), own]
    others = len(candidates) - 1
    wrong = (matrix.sum(axis=1) - right) / others if others else np.full(len(writers), np.nan)
    return Distance(
        writers=writers,
        per_writer=right,
        wrong=wrong,
        nearest_is_right=matrix.argmin(axis=1) == own,
    )


@dataclass(frozen=True)
class HwdResult:
    generated: Distance
    real: Distance
    typeface: Distance
    reference_lines: int
    scored: int
    withheld_samples: int
    candidates: int
    """Writers in the reference: chance for nearest-is-right is one in this many."""

    @property
    def position(self) -> float:
        """Where the model sits between real handwriting (0) and a typeface (1)."""
        span = self.typeface.value - self.real.value
        if span <= 0:
            return float("nan")
        return (self.generated.value - self.real.value) / span

    @property
    def chance(self) -> float:
        return 1.0 / self.candidates if self.candidates else float("nan")

    @property
    def identity(self) -> float:
        """The generated lines' identity gap as a share of the real lines'.

        0: no closer to their own writer than to anyone else -- handwriting, but
        nobody's in particular. 1: as distinctly their writer's as that writer's
        own lines. Both distances come from the same set, so output cannot score
        here merely by being blurred, clean, or far from everyone.
        """
        generated, real = self._gaps()
        return _identity(generated, real, np.arange(len(real)))

    def identity_interval(self, seed: int = 1337) -> bootstrap.Interval:
        """Generated and real resampled together: the two gaps belong to the same
        people, and drawing them apart would add noise the comparison lacks."""
        generated, real = self._gaps()
        return _writer_interval(
            lambda indices: _identity(generated, real, indices), len(real), seed
        )

    def _gaps(self) -> tuple[np.ndarray, np.ndarray]:
        if self.generated.writers != self.real.writers:
            raise ValueError("generated and real were scored over different writers")
        return self.generated.gap, self.real.gap


def _identity(generated_gap: np.ndarray, real_gap: np.ndarray, indices: np.ndarray) -> float:
    real = float(np.mean(real_gap[indices]))
    if real <= 0:
        return float("nan")
    return float(np.mean(generated_gap[indices])) / real


def measure(
    generated: Sequence[np.ndarray],
    real: Sequence[np.ndarray],
    typeface: Sequence[np.ndarray],
    writer_ids: Sequence[str],
    reference_images: Sequence[np.ndarray],
    reference_ids: Sequence[str],
    extractor: Extractor | None = None,
) -> HwdResult | None:
    """Score the three aligned sets against one shared reference.

    Entry *i* of ``generated``, ``real`` and ``typeface`` all belong to
    ``writer_ids[i]``. Samples whose writer the reference does not hold are left
    out of all three and counted in ``withheld_samples``.

    Returns None when the package is not installed, so a run without it reports
    one metric fewer rather than failing.
    """
    for name, images in (("generated", generated), ("real", real), ("typeface", typeface)):
        if len(images) != len(writer_ids):
            raise ValueError(f"{len(images)} {name} images for {len(writer_ids)} writer ids")

    if extractor is None:
        if not available():
            return None
        extractor = VggExtractor()

    reference = extractor(reference_images, reference_ids)
    keep = [index for index, writer in enumerate(writer_ids) if writer in reference]
    if not keep:
        raise ValueError("no sample's writer is in the reference")
    ids = [writer_ids[index] for index in keep]

    def score(images: Sequence[np.ndarray]) -> Distance:
        return distance(extractor([images[index] for index in keep], ids), reference)

    return HwdResult(
        generated=score(generated),
        real=score(real),
        typeface=score(typeface),
        reference_lines=len(reference_images),
        scored=len(keep),
        withheld_samples=len(writer_ids) - len(keep),
        candidates=len(reference),
    )


class VggExtractor:
    """HWD's own backbone and transforms, loaded once and reused for every set.

    Through the package's FolderDataset rather than around it, so the resize, the
    padding and the column masking are exactly the published ones. FolderDataset
    reads each image's writer from its parent directory, hence the files.
    """

    def __init__(self, height: int = DEFAULT_HEIGHT) -> None:
        _load_hwd()
        from hwd.scores import HWDScore

        self._score = HWDScore(height=height)

    def __call__(
        self, images: Sequence[np.ndarray], writer_ids: Sequence[str]
    ) -> dict[str, np.ndarray]:
        from hwd.datasets import FolderDataset

        if len(images) == 0:
            return {}
        root = Path(tempfile.mkdtemp(prefix="nib_hwd_"))
        try:
            _write_by_writer(images, writer_ids, root)
            processed = self._score.digest(FolderDataset(root))
            return writer_means(processed.features.cpu().numpy(), processed.authors)
        finally:
            shutil.rmtree(root, ignore_errors=True)


def _write_by_writer(
    images: Sequence[np.ndarray], writer_ids: Sequence[str], directory: Path
) -> None:
    """One subdirectory per writer, because that is where HWD reads identity."""
    for index, (image, writer) in enumerate(zip(images, writer_ids, strict=True)):
        folder = directory / str(writer)
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{index:05d}.png"), np.asarray(image, dtype=np.uint8))


def describe(result: HwdResult | None) -> str:
    """The figure with the two anchors that give it a size, all from this run."""
    if result is None:
        return "not measured -- pip install the 'hwd' extra"
    lines = [
        f"reference  {result.reference_lines} real lines over {len(result.real.writers)} "
        "writers, none a target or a style line",
        f"generated  {result.generated.interval().format()}",
        f"real       {result.real.interval().format()}   "
        "the target lines themselves: same writers, same texts",
        f"typeface   {result.typeface.interval().format()}   "
        "the same texts in a font: no hand at all",
        f"-> {result.position:.0%} of the way from real handwriting to no style at all",
    ]
    if result.withheld_samples:
        lines.append(
            f"withheld   {result.withheld_samples} samples whose writers have fewer than "
            f"{REFERENCE_MINIMUM} spare lines, from all three sets alike"
        )
    lines += [
        "",
        f"identity   its own writer against the other {result.candidates - 1}",
        f"           {'gap to the others':<26}own writer nearest (chance {result.chance:.1%})",
    ]
    for name, side in (
        ("generated", result.generated),
        ("real", result.real),
        ("typeface", result.typeface),
    ):
        gap = side.gap_interval().format()
        nearest = side.nearest_interval().format(as_percent=True)
        lines.append(f"{name:<10} {gap:<26}{nearest}")
    lines.append(
        f"-> the generated lines carry {result.identity_interval().format(as_percent=True)} "
        "of the writer identity real lines do"
    )
    return "\n".join(lines)
