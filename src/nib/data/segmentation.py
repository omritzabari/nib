"""Split a normalised page into lines of handwriting.

The generator works in lines and a user brings a page, so this is the first step
from one to the other: the style lines it is shown, and the real lines a result
is judged against, both come out of here.

**Components, not a row profile.** The textbook split sums ink along each row
and cuts at the valleys. Handwriting defeats it: a ``g`` or a ``y`` hangs into
the line below and a capital reaches into the line above, so the valleys fill in
and the cuts slice letters in half. Here the ink is broken into connected
components -- roughly letters, or runs of joined letters -- and each is assigned
*whole* to the line its centre of mass sits in. A descender then travels with its
letter, and a crop is drawn from its own components only, so a neighbour's
descender cannot leak into it.

**A low ink threshold, measured.** After :func:`~nib.data.preprocessing.normalise_page`
the handwriting sits at 0 -- ``normalise_ink`` puts the darkest 2% there -- while
the squared paper's grid survives ruling suppression at roughly 34 to 150. On
Amri's five photographs, below 40 the text band is 3.6-8.2% ink and an empty band
of grid is 0.00-0.17%, or 0.75% on the WhatsApp copy, whose compression darkens
the grid most. At 100 that copy's grid reaches 6.2%, nearly as much as its
writing. So a strict threshold separates hand from paper where geometry alone
would struggle.

**Hysteresis is built in, and off by default.** The first "The" on Amri's page
was written lightly: only 63 of its pixels fall below 40, so at that threshold
it breaks into specks and is lost. Building components from a looser threshold
and keeping those with enough truly dark pixels brings it back -- and brings the
grid back with it. Lines found on the five photographs, where 13 is right:

    weak  min_strong   angle  dim  normal  shadow  WhatsApp
     40        3          13   19      13      13        13
     80       15          13   19      13      13        14
    120       15          13   14      13      13        16

A lost light word costs less than a false line, so the looser pass stays off.

**The dim photograph is a page-detection failure, not a segmentation one.**
``find_page`` returns the whole frame for it, so the black cloth above the sheet
reaches this module and splits into false lines.

**What is not handwriting is dropped and counted, never silently.** Specks;
ruling remnants and the margin line, which are long and thin; blobs, which are
large and filled -- punch holes, and crossed-out words, which make poor style
samples anyway; and oversized components, which are grid that survived as a
lattice. Components within a thin band along the page's border are edge: a
cropped photograph keeps a strip of desk or a dark border there, and handwriting
rarely comes that close. A component too far from every line is a stray, and a
"line" of too few components -- a punch hole with nothing beside it -- is sparse.

**Grid lines are cut out of the ink first.** Some grid survives the threshold
and touches letters, and a letter joined to a grid line becomes one long thin
component that is dropped as ruling -- which is how "The" and "written" vanished
from the first run on Amri's photographs. Straight horizontal and vertical runs
far longer than any stroke are removed from the ink mask before it is split.

**Small marks follow the letter nearest them, not the nearest line.** Assigning
every component by its centre sent an asterisk written high before "She" to the
line above, and the bowl of a "P" -- written as two strokes -- to one line and
its stem to the next. Dots, commas and full stops were lost outright, as specks.
Now only components of at least ``anchor_share`` of the median letter height
place the lines; every smaller mark -- a dot, a comma, a bowl, and a speck that
passed the darkness test -- joins the line of the nearest anchor, if one is
within ``attach_reach``. A speck with no letter near it is still dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentConfig:
    """Every tunable in one place, so an experiment is a config change."""

    ink_threshold: int = 40
    """Darker than this, on a normalised page, is certainly ink. See the module docstring."""

    weak_threshold: int = 40
    """Darker than this may be ink. Components are built from these pixels and kept
    only when ``min_strong`` of them are also darker than ``ink_threshold``. Equal
    to it by default, which turns the looser pass off; the module docstring has
    the measurement behind that."""

    min_strong: int = 3
    """Certainly-ink pixels a component needs to count as writing at all."""

    min_area: int = 15
    """Components with fewer pixels are specks: surviving grid crossings, noise."""

    straight_run: float = 0.05
    """Horizontal or vertical ink runs at least this share of the page width long
    are grid, and are cut from the ink mask before components are found. The bar
    of a capital T on these pages is under half of it."""

    rule_elongation: float = 10.0
    """A bounding box this many times longer than it is thick is ruling..."""

    rule_min_length: float = 0.15
    """...provided it also spans this share of the page along its long side, so an
    underline or a long dash inside a line is kept."""

    blob_fill: float = 0.55
    """Filling more than this share of its bounding box makes a large component a
    blob. Strokes are thin and leave most of their box empty."""

    blob_min_side: float = 0.02
    """A blob's shorter side, as a share of page width. Below it a filled component
    is a full stop or an i's dot."""

    oversized: float = 4.0
    """A component taller than this many median component heights is not a letter."""

    line_min_share: float = 0.08
    """A profile peak weaker than this share of the strongest line is not a line."""

    edge_band: float = 0.03
    """Share of the page's shorter side, along every border, inside which a
    component is the page's edge rather than writing."""

    min_line_components: int = 3
    """A line of fewer components is a punch hole or a smudge. The shortest real
    line on Amri's page, the end of the alphabet, has five to seven."""

    anchor_share: float = 0.6
    """Components at least this share of the median component height place the
    lines. Smaller ones join the line of the nearest of these."""

    attach_reach: float = 0.75
    """How near a small mark must be to a letter to join its line, in median
    component heights, measured between their bounding boxes."""

    padding: int = 6
    """White border around each crop, in pixels."""

    stroke_margin: int = 2
    """Pixels grown around each component when copying it out, so the soft edge of
    a stroke -- lighter than the ink threshold -- comes with it."""


DEFAULT = SegmentConfig()

REASONS = ("faint", "speck", "edge", "ruling", "blob", "oversized", "stray", "sparse")

_NO_MIN_AREA = SegmentConfig(min_area=0)
"""The same rules without the size test: what a speck would be if it were bigger."""


@dataclass(frozen=True)
class Line:
    image: np.ndarray
    """Grayscale crop on white, holding this line's strokes and nothing else."""

    box: tuple[int, int, int, int]
    """x, y, width, height of the crop on the page."""

    components: int


@dataclass(frozen=True)
class Segmentation:
    lines: list[Line]
    """Top to bottom."""

    dropped: dict[str, int]
    """Components left out, by reason."""


def split_lines(page: np.ndarray, config: SegmentConfig = DEFAULT) -> Segmentation:
    """Lines of handwriting on a page from ``normalise_page``, in reading order."""
    if page.ndim != 2:
        raise ValueError("split_lines expects the grayscale page normalise_page returns")

    height, width = page.shape
    ink = (page < config.weak_threshold).astype(np.uint8)
    ink[_straight_runs(ink, width, config) > 0] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    strong = np.bincount(labels[page < config.ink_threshold], minlength=count)
    dropped = dict.fromkeys(REASONS, 0)

    kept: list[int] = []
    specks: list[int] = []
    for index in range(1, count):
        if strong[index] < config.min_strong:
            dropped["faint"] += 1
            continue
        reason = _reject(stats[index], width, height, config)
        if reason is None:
            kept.append(index)
        elif reason == "speck" and _reject(stats[index], width, height, _NO_MIN_AREA) is None:
            specks.append(index)
        else:
            dropped[reason] += 1
    if not kept:
        dropped["speck"] += len(specks)
        return Segmentation(lines=[], dropped=dropped)

    # Oversized is relative to the writing itself, so it needs the others first.
    scale = float(np.median(stats[kept, cv2.CC_STAT_HEIGHT]))
    letters = [i for i in kept if stats[i, cv2.CC_STAT_HEIGHT] <= config.oversized * scale]
    dropped["oversized"] += len(kept) - len(letters)
    anchors = [i for i in letters if stats[i, cv2.CC_STAT_HEIGHT] >= config.anchor_share * scale]
    anchor_set = set(anchors)
    small = [i for i in letters if i not in anchor_set] + specks
    if not anchors:
        dropped["speck"] += len(specks)
        return Segmentation(lines=[], dropped=dropped)

    rows = centroids[anchors, 1]
    centres = _line_centres(rows, stats[anchors, cv2.CC_STAT_AREA], height, scale, config)
    pitch = float(np.median(np.diff(centres))) if len(centres) > 1 else 3.0 * scale
    nearest = np.abs(rows[:, None] - centres[None, :]).argmin(axis=1)
    far = np.abs(rows - centres[nearest]) > 0.75 * pitch
    dropped["stray"] += int(far.sum())
    line_of = {anchor: int(nearest[i]) for i, anchor in enumerate(anchors) if not far[i]}

    attached = _attach(stats, small, list(line_of), config.attach_reach * scale)
    speck_set = set(specks)
    for mark, anchor in attached.items():
        if anchor is None:
            dropped["speck" if mark in speck_set else "stray"] += 1
        else:
            line_of[mark] = line_of[anchor]

    lines = []
    for number in range(len(centres)):
        members = [c for c, n in line_of.items() if n == number]
        placed = sum(1 for c in members if c in anchor_set)
        if placed < config.min_line_components:
            dropped["sparse"] += len(members)
            continue
        lines.append(_crop(page, labels, stats, members, config))
    return Segmentation(lines=lines, dropped=dropped)


def _attach(stats: np.ndarray, marks: list[int], anchors: list[int], reach: float) -> dict:
    """For each small mark, the nearest anchor within ``reach`` -- by the gap
    between their bounding boxes -- or None."""
    if not marks:
        return {}
    if not anchors:
        return dict.fromkeys(marks)

    def boxes(indices):
        x = stats[indices, cv2.CC_STAT_LEFT].astype(float)
        y = stats[indices, cv2.CC_STAT_TOP].astype(float)
        return x, y, x + stats[indices, cv2.CC_STAT_WIDTH], y + stats[indices, cv2.CC_STAT_HEIGHT]

    mx0, my0, mx1, my1 = boxes(marks)
    ax0, ay0, ax1, ay1 = boxes(anchors)
    dx = np.maximum(0, np.maximum(ax0[None, :] - mx1[:, None], mx0[:, None] - ax1[None, :]))
    dy = np.maximum(0, np.maximum(ay0[None, :] - my1[:, None], my0[:, None] - ay1[None, :]))
    gap = np.hypot(dx, dy)
    best = gap.argmin(axis=1)
    return {
        mark: (anchors[best[row]] if gap[row, best[row]] <= reach else None)
        for row, mark in enumerate(marks)
    }


def _straight_runs(ink: np.ndarray, width: int, config: SegmentConfig) -> np.ndarray:
    """Pixels on horizontal or vertical ink runs too long to be part of a letter."""
    length = max(15, int(config.straight_run * width))
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    )
    return cv2.bitwise_or(horizontal, vertical)


def _reject(stat: np.ndarray, width: int, height: int, config: SegmentConfig) -> str | None:
    x = int(stat[cv2.CC_STAT_LEFT])
    y = int(stat[cv2.CC_STAT_TOP])
    w = int(stat[cv2.CC_STAT_WIDTH])
    h = int(stat[cv2.CC_STAT_HEIGHT])
    area = int(stat[cv2.CC_STAT_AREA])
    if area < config.min_area:
        return "speck"
    band = config.edge_band * min(width, height)
    if x < band or y < band or x + w > width - band or y + h > height - band:
        return "edge"
    long_side, short_side = max(w, h), max(1, min(w, h))
    span = width if w >= h else height
    if (
        long_side / short_side >= config.rule_elongation
        and long_side >= config.rule_min_length * span
    ):
        return "ruling"
    if min(w, h) >= config.blob_min_side * width and area / (w * h) >= config.blob_fill:
        return "blob"
    return None


def _line_centres(
    rows: np.ndarray, areas: np.ndarray, height: int, scale: float, config: SegmentConfig
) -> np.ndarray:
    """Rows where lines sit: peaks of the components' centres, weighted by ink.

    Centres rather than every inked row, so ascenders and descenders do not smear
    a line into its neighbours; smoothed at half a letter's height, so a line's
    tall and short letters merge into one peak.
    """
    profile = np.bincount(
        np.clip(rows.round().astype(int), 0, height - 1), weights=areas, minlength=height
    )
    sigma = max(2.0, scale / 2.0)
    radius = int(3 * sigma)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    smooth = np.convolve(profile, kernel / kernel.sum(), mode="same")

    inner = smooth[1:-1]
    peaks = np.flatnonzero((inner > smooth[:-2]) & (inner >= smooth[2:])) + 1
    peaks = peaks[smooth[peaks] >= config.line_min_share * smooth.max()]

    # Strongest first, and nothing within a line's height and a half of a peak
    # already taken: two peaks that close are one line's top and bottom halves.
    chosen: list[int] = []
    for peak in peaks[np.argsort(-smooth[peaks])]:
        if all(abs(int(peak) - other) >= 1.5 * scale for other in chosen):
            chosen.append(int(peak))
    return np.sort(np.array(chosen, dtype=float))


def _crop(
    page: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    members: list[int],
    config: SegmentConfig,
) -> Line:
    height, width = page.shape
    x0 = int(stats[members, cv2.CC_STAT_LEFT].min())
    y0 = int(stats[members, cv2.CC_STAT_TOP].min())
    x1 = int((stats[members, cv2.CC_STAT_LEFT] + stats[members, cv2.CC_STAT_WIDTH]).max())
    y1 = int((stats[members, cv2.CC_STAT_TOP] + stats[members, cv2.CC_STAT_HEIGHT]).max())
    x0, y0 = max(0, x0 - config.padding), max(0, y0 - config.padding)
    x1, y1 = min(width, x1 + config.padding), min(height, y1 + config.padding)

    mask = np.isin(labels[y0:y1, x0:x1], members).astype(np.uint8)
    if config.stroke_margin:
        size = 2 * config.stroke_margin + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    image = np.full(mask.shape, 255, dtype=np.uint8)
    region = page[y0:y1, x0:x1]
    image[mask > 0] = region[mask > 0]
    return Line(image=image, box=(x0, y0, x1 - x0, y1 - y0), components=len(members))
