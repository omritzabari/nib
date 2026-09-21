"""Can a trivial model tell the system's lines from real handwriting?

The product's question is whether someone who knows a hand can tell the system's
writing from the person's own. The machine form of that is a *classifier
two-sample test*: train a classifier to separate real lines from generated ones,
score it on lines it never saw, and read its accuracy. **50% is the goal** -- the
two sets cannot be told apart. 100% means every line gives itself away.

The classifier here is deliberately weak: a logistic regression over ten plain
statistics of the ink. A strong one would also find differences, but could not
say what they are. These ten each name a defect that can be fixed, and the
report gives each one's accuracy alone beside the whole. Measured first on cell
7e's saved pairs, before this module existed: 96.9% overall, and the spread of
ink darkness alone told real from generated 93.8% of the time -- generated ink
is flatter in tone than a pen's.

It judges *the ink*, not whose hand it is; HWD identity answers that. A change
that brings this toward 50% while identity falls has made the lines look more
like handwriting and less like the writer's.

**Fit and judge on different lines.** Anything calibrated from a writer's style
lines must be judged against *other* real lines of theirs -- the evaluation
compares against the target lines, which the generator never sees -- or a
calibration that copies the statistics measured here would pass by construction.

No network and no optional dependency: the regression is a few lines of numpy,
so this runs in the test suite and on any machine in seconds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from nib.engine.metrics import bootstrap

HEIGHT = 64
"""Every image is scaled to this height first. Most of the statistics are in
pixels, and a line compared at another scale would differ for that alone."""

INK = 160
"""A pixel darker than this is ink. The measurements this module was built from
were made at this threshold; the ink deficit they found held at every threshold
from 64 to 200."""

FEATURES: tuple[tuple[str, str], ...] = (
    ("ink_per_column", "ink pixels per column of the written span"),
    ("ink_height", "height of the ink, as a share of the image"),
    ("pieces_per_100_columns", "separate pieces of ink per 100 columns"),
    ("median_piece_area", "the median piece of ink, in pixels"),
    ("piece_area_spread", "how much piece sizes vary, as a share of their mean"),
    ("blank_column_share", "empty columns inside the written span"),
    ("darkness", "mean darkness of the ink"),
    ("darkness_spread", "how much the darkness of the ink varies"),
    ("baseline_wobble", "how far the ink's centre line moves up and down"),
    ("stroke_width", "twice the median distance from ink to paper"),
)
"""The ten statistics, in the order :func:`features` returns them."""

NAMES = tuple(name for name, _ in FEATURES)

FOLDS = 10
PENALTY = 1.0
"""L2 penalty on the standardised weights. The ten statistics are strongly
correlated -- darkness and its spread, pieces and their size -- and an unpenalised
fit on a few hundred lines chases that noise."""


def features(image: np.ndarray) -> np.ndarray | None:
    """The ten statistics of one line, or None for a line with no ink to measure."""
    gray = _at_height(np.asarray(image))
    ink = (gray < INK).astype(np.uint8)
    columns = ink.sum(axis=0)
    written = np.nonzero(columns)[0]
    if len(written) < 10:
        return None
    left, right = written[0], written[-1] + 1
    span = right - left
    rows = np.nonzero(ink.sum(axis=1))[0]

    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)

    occupied = columns[left:right] > 0
    heights = np.arange(gray.shape[0], dtype=np.float64)[:, None]
    centre = (ink[:, left:right] * heights).sum(axis=0)[occupied] / columns[left:right][occupied]
    darkness = 255.0 - gray[ink > 0].astype(np.float64)

    return np.array(
        [
            ink.sum() / span,
            (rows[-1] - rows[0] + 1) / gray.shape[0],
            100.0 * (count - 1) / span,
            float(np.median(areas)),
            float(areas.std() / max(areas.mean(), 1e-9)),
            float((columns[left:right] == 0).mean()),
            float(darkness.mean()),
            float(darkness.std()),
            float(centre.std()),
            float(2.0 * np.median(distance[ink > 0])),
        ]
    )


@dataclass(frozen=True)
class Detectability:
    """How well the ten statistics separate generated lines from real ones."""

    accuracy: bootstrap.Interval
    """Cross-validated: every line is judged by a classifier that never saw it."""

    alone: dict[str, float]
    """Each statistic's accuracy on its own, cross-validated the same way."""

    real: dict[str, float]
    generated: dict[str, float]
    """Each statistic's mean on either side."""

    lines: int
    """Lines judged, both sides together."""

    skipped: int
    """Lines with too little ink to measure, left out of both the fit and the score."""

    def describe(self) -> str:
        lines = [
            f"a classifier tells generated from real  {self.accuracy.format(as_percent=True)}"
            "   (50% = cannot be told apart)",
            f"over {self.lines} lines"
            + (f", {self.skipped} without ink skipped" if self.skipped else ""),
            "",
            f"  {'what gives it away':<26}{'alone':>7}{'real':>10}{'generated':>11}",
        ]
        for name in sorted(NAMES, key=lambda n: -self.alone[n]):
            lines.append(
                f"  {name:<26}{self.alone[name]:>7.1%}"
                f"{self.real[name]:>10.2f}{self.generated[name]:>11.2f}"
            )
        return "\n".join(lines)


def measure(
    real: Sequence[np.ndarray],
    generated: Sequence[np.ndarray],
    folds: int = FOLDS,
    seed: int = 0,
) -> Detectability:
    """Separate the two sets as well as ten statistics allow, and report how well.

    The folds are stratified -- each holds the same share of either side -- so a
    fold can never be scored against a classifier that saw only one class.
    """
    rows, labels, skipped = [], [], 0
    for label, images in ((0, real), (1, generated)):
        for image in images:
            values = features(image)
            if values is None:
                skipped += 1
                continue
            rows.append(values)
            labels.append(label)
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(set(labels)) < 2 or min(np.bincount(y.astype(int))) < folds:
        raise ValueError(f"need at least {folds} measurable lines on either side")

    fold = _stratified_folds(y, folds, seed)
    correct = _cross_validate(x, y, fold)
    alone = {
        name: float(_cross_validate(x[:, [i]], y, fold).mean()) for i, name in enumerate(NAMES)
    }
    return Detectability(
        accuracy=bootstrap.rate_interval(correct),
        alone=alone,
        real=dict(zip(NAMES, x[y == 0].mean(axis=0).tolist(), strict=True)),
        generated=dict(zip(NAMES, x[y == 1].mean(axis=0).tolist(), strict=True)),
        lines=len(y),
        skipped=skipped,
    )


def _at_height(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if gray.shape[0] == HEIGHT:
        return gray
    scale = HEIGHT / gray.shape[0]
    width = max(1, round(gray.shape[1] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (width, HEIGHT), interpolation=interpolation)


def _stratified_folds(y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for label in (0.0, 1.0):
        members = rng.permutation(np.nonzero(y == label)[0])
        fold[members] = np.arange(len(members)) % folds
    return fold


def _cross_validate(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Whether each line was classified correctly by a model fitted without it."""
    correct = np.empty(len(y), dtype=bool)
    for k in np.unique(fold):
        test = fold == k
        mean = x[~test].mean(axis=0)
        scale = x[~test].std(axis=0)
        scale[scale == 0] = 1.0
        weights = _fit((x[~test] - mean) / scale, y[~test])
        predicted = _probability((x[test] - mean) / scale, weights) > 0.5
        correct[test] = predicted == (y[test] == 1)
    return correct


def _fit(x: np.ndarray, y: np.ndarray, iterations: int = 50) -> np.ndarray:
    """L2-penalised logistic regression by Newton's method; the intercept is free."""
    design = np.hstack([np.ones((len(x), 1)), x])
    weights = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], PENALTY)
    penalty[0] = 0.0
    for _ in range(iterations):
        p = _sigmoid(design @ weights)
        gradient = design.T @ (p - y) + penalty * weights
        hessian = (
            (design * (p * (1 - p))[:, None]).T @ design
            + np.diag(penalty)
            + 1e-9 * np.eye(design.shape[1])
        )
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.abs(step).max() < 1e-8:
            break
    return weights


def _probability(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _sigmoid(weights[0] + x @ weights[1:])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
