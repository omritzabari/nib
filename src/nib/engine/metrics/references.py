"""Where a pack's reference numbers live, and how they are read back.

Three numbers decide whether a generated result is any good: the FID floor
between two disjoint halves of real handwriting, the recogniser's own error rate,
and the writer embedding's accuracy on real images. None of them is a property of
the project -- each is a property of *a particular pack*. A floor measured on word
crops says nothing about generated lines: lines are five times wider, hold far
more paper per image, and land somewhere else entirely in Inception's feature
space.

So the numbers are stored per pack and named after it. ``scripts/check_metrics.py``
writes them and ``scripts/evaluate_generator.py`` reads them, which removes the
step where three figures are copied by hand from one Colab session into the next.
A baseline that is wrong by transcription is worse than no baseline at all,
because nothing about it looks wrong.

The file is small, human-readable, and **committed**, all on purpose. Committed
for the same reason the writer split is: it has to survive the trip from the
machine that measured it to the machine that generates, it has to survive
re-downloading the dataset, and it is the artefact someone else needs in order to
reproduce a number you published. A figure nobody can trace back to a run is a
figure nobody should trust.
"""

from __future__ import annotations

import json
from pathlib import Path

FILENAME_PREFIX = "references_"

REQUIRED = ("fid_floor", "cer_real", "retrieval_real")
"""The three a generated result is scored against. A file missing any of them is
usable but incomplete -- a run with ``--skip-cer``, most likely."""


def path_for(outputs: Path | str, pack_name: str) -> Path:
    """The reference file belonging to a pack.

    Keyed on the pack's filename, so ``cvl_lines_64.lmdb`` and
    ``cvl_words_64.lmdb`` cannot overwrite each other's numbers -- which is the
    whole failure this module exists to prevent.
    """
    return Path(outputs) / f"{FILENAME_PREFIX}{Path(pack_name).stem}.json"


def save(outputs: Path | str, pack_name: str, values: dict) -> Path:
    """Write a pack's reference numbers exactly as given, replacing the file.

    Prefer :func:`update` unless you mean to discard what is already there.
    """
    path = path_for(outputs, pack_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def update(outputs: Path | str, pack_name: str, values: dict) -> tuple[Path, list[str]]:
    """Write what this run measured, keeping what it did not.

    Returns the path and the names carried over from the previous file.

    Replacing the file wholesale looked obviously right until a run measured two
    of the three. ``check_metrics.py`` reads CER from the *raw* CVL line images
    rather than from the pack, so a machine that has the pack but not the 5 GB of
    source -- a Colab session, exactly the case this was built for -- skips CER
    and would have deleted a figure someone had already measured. Losing a real
    number to a run that did not produce one is precisely the quiet damage this
    module exists to prevent.

    Carried-over names are returned rather than absorbed, so the caller can say
    which figures in the file came from some earlier run.
    """
    previous = load(outputs, pack_name) or {}
    carried = sorted(name for name in previous if name not in values)
    return save(outputs, pack_name, {**previous, **values}), carried


def load(outputs: Path | str, pack_name: str) -> dict | None:
    """A pack's reference numbers, or None if they have not been measured.

    None rather than a default. A caller that silently substituted some other
    pack's numbers would report a comparison it never made, and the point of
    every number in this project is that it can be traced to a run.
    """
    path = path_for(outputs, pack_name)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def missing(values: dict) -> list[str]:
    """Which of the three a reference file does not carry."""
    return [name for name in REQUIRED if name not in values]
