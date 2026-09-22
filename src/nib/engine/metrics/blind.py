"""The blind test that decides whether nib works (CLAUDE.md, "How success is measured").

A judge sees three real lines by a writer, then the same text twice -- once in
the writer's hand, once generated -- and picks the real one. The same text on
both sides, so the words cannot give it away. 50% means the judges cannot tell;
nib works at :data:`PASS` or below.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np

from nib.engine.metrics import bootstrap

PASS = 0.60
"""The share of pairs in which judges may pick the real line, at most, for nib to count as working."""

INK = 160
MARGIN = 4

CODE_PREFIX = "NIB1"
"""Each judge's answers come back as one line of text they copy from the page:
``NIB1;<judge>;<trial><A|B> <trial><A|B> ...`` -- anyone can send it, no account."""


def crop_to_ink(image: np.ndarray, margin: int = MARGIN) -> np.ndarray:
    """The ink and a thin white margin.

    Pack lines are cropped to their ink and generated ones carry white above and
    below, so shown as they are the margins alone would say which is real. On a
    page the page engine sets the size; here both sides are shown alike.
    """
    ink = np.asarray(image) < INK
    rows, columns = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if rows.size == 0:
        return image
    crop = image[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    return np.pad(crop, margin, constant_values=255)


def assign_sides(count: int, seed: int) -> list[str]:
    """Which side holds the real line in each trial: half A, half B, shuffled --
    so a judge who always answers A scores exactly 50%, not whatever a coin gave."""
    sides = ["A", "B"] * (count // 2) + ["A"] * (count % 2)
    random.Random(seed).shuffle(sides)
    return sides


def parse_codes(text: str) -> list[dict]:
    """The answers in the codes judges sent back, one code a line; other lines ignored."""
    answers = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(CODE_PREFIX + ";"):
            continue
        _, judge, choices = line.split(";", 2)
        for item in choices.split():
            answers.append({"judge": judge, "trial": int(item[:-1]), "choice": item[-1]})
    return answers


def score(answers: Sequence[Mapping], key: Mapping[str, str]) -> dict:
    """How often the judges picked the real line.

    ``answers`` holds ``{"judge", "trial", "choice"}`` per judgment; ``key`` maps a
    trial to the side the real line was on. The interval resamples judgments.
    """
    if len(answers) < 2:
        raise ValueError(f"need at least two answers, got {len(answers)}")
    hits = np.array([a["choice"] == key[str(a["trial"])] for a in answers], dtype=float)
    interval = bootstrap.bootstrap_statistic(lambda i: float(hits[i].mean()), len(hits))
    names = np.array([str(a["judge"]) for a in answers])
    per_judge = {judge: float(hits[names == judge].mean()) for judge in sorted(set(names))}
    return {
        "accuracy": interval.value,
        "interval": [interval.low, interval.high],
        "answers": len(hits),
        "per_judge": per_judge,
        "passes": interval.value <= PASS,
    }
