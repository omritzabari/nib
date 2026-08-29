"""Writer-disjoint splits.

The single most important correctness property in this project's data layer.

The standard splits shipped with handwriting datasets were built for *recognition*,
where the same writer appearing in train and test is harmless -- you are testing
whether the model reads text, not whether it generalises across people. For us it
is fatal: if the model has seen a writer, reproducing their hand is memory, not
few-shot generalisation, and the headline claim of the project is false.

So: a writer belongs to exactly one side. This module builds such a split, asserts
the property, and writes it to a JSON file that is committed to the repository.

Committing the split matters for two reasons. It survives re-downloading the
dataset, so numbers stay comparable across time. And it is the artefact someone
else needs in order to reproduce a reported result.

Balancing note: writers contribute different numbers of samples, so splitting
*writers* 70/30 does not split *samples* 70/30. The assignment below shuffles
writers deterministically and then repeatedly gives the next writer to whichever
side is furthest below its target share of samples.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RATIOS = {"train": 0.7, "test": 0.3}


class SplitError(RuntimeError):
    pass


@dataclass(frozen=True)
class WriterSplit:
    """An assignment of writer ids to named splits."""

    name: str
    seed: int
    ratios: dict[str, float]
    writers: dict[str, list[str]]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for split_name, ids in self.writers.items():
            overlap = seen & set(ids)
            if overlap:
                raise SplitError(
                    f"writers appear in more than one split: {sorted(overlap)[:10]}. "
                    "This is the data leak the whole module exists to prevent."
                )
            seen |= set(ids)
            if len(set(ids)) != len(ids):
                raise SplitError(f"split {split_name!r} lists a writer twice")

    @property
    def all_writers(self) -> set[str]:
        return {w for ids in self.writers.values() for w in ids}

    def split_of(self, writer_id: str) -> str | None:
        for split_name, ids in self.writers.items():
            if writer_id in ids:
                return split_name
        return None

    def partition(
        self,
        records: Sequence[Any],
        key: Callable[[Any], str] = lambda r: r.writer_id,
        strict: bool = True,
    ) -> dict[str, list[Any]]:
        """Divide ``records`` according to this split.

        With ``strict``, a record whose writer is not in the split raises rather
        than being dropped. Silently discarding samples because a split file is
        stale is exactly the kind of quiet loss this codebase refuses.
        """
        out: dict[str, list[Any]] = {name: [] for name in self.writers}
        unknown: set[str] = set()

        for record in records:
            writer_id = key(record)
            split_name = self.split_of(writer_id)
            if split_name is None:
                unknown.add(writer_id)
                continue
            out[split_name].append(record)

        if unknown and strict:
            raise SplitError(
                f"{len(unknown)} writers are not in split {self.name!r}: "
                f"{sorted(unknown)[:10]}. The split file is stale, or the records "
                "come from a different dataset. Rebuild the split or pass strict=False."
            )
        return out

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "seed": self.seed,
            "ratios": self.ratios,
            "counts": {k: len(v) for k, v in self.writers.items()},
            "writers": {k: sorted(v) for k, v in self.writers.items()},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> WriterSplit:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            seed=data["seed"],
            ratios=data["ratios"],
            writers={k: list(v) for k, v in data["writers"].items()},
        )

    def summary(self, sample_counts: dict[str, int] | None = None) -> str:
        total_writers = len(self.all_writers)
        lines = [f"split {self.name!r}  seed={self.seed}  writers={total_writers}"]
        total_samples = sum(sample_counts.values()) if sample_counts else 0
        for split_name, ids in self.writers.items():
            writer_share = len(ids) / total_writers if total_writers else 0
            row = f"  {split_name:<8} writers {len(ids):>5} ({writer_share:6.1%})"
            if sample_counts:
                n = sum(sample_counts.get(w, 0) for w in ids)
                sample_share = n / total_samples if total_samples else 0
                row += f"   samples {n:>7} ({sample_share:6.1%})"
            lines.append(row)
        return "\n".join(lines)


def make_split(
    sample_counts: dict[str, int],
    ratios: dict[str, float] | None = None,
    seed: int = 1337,
    name: str = "default",
) -> WriterSplit:
    """Build a writer-disjoint split, balanced by sample count.

    Args:
        sample_counts: writer id -> how many samples that writer contributes.
            Counts rather than a bare id list, because balancing on writers alone
            gives lopsided sample totals when writers differ in productivity.
        ratios: split name -> target share of *samples*. Must sum to 1.
        seed: shuffling seed. The same inputs and seed always give the same split.
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    if not sample_counts:
        raise SplitError("no writers to split")
    if len(sample_counts) < len(ratios):
        raise SplitError(
            f"cannot divide {len(sample_counts)} writers into {len(ratios)} non-empty splits"
        )

    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise SplitError(f"ratios must sum to 1, got {total_ratio}")
    if any(v <= 0 for v in ratios.values()):
        raise SplitError(f"every ratio must be positive, got {ratios}")

    # Deterministic order first, then shuffle, so the result never depends on the
    # iteration order of the incoming dict.
    writer_ids = sorted(sample_counts)
    random.Random(seed).shuffle(writer_ids)

    total_samples = sum(sample_counts.values())
    targets = {name_: ratio * total_samples for name_, ratio in ratios.items()}
    assigned: dict[str, list[str]] = {name_: [] for name_ in ratios}
    running: dict[str, float] = dict.fromkeys(ratios, 0.0)

    for writer_id in writer_ids:
        # Give this writer to whoever is furthest below target. Ties break on the
        # split name so the outcome stays deterministic.
        deficits = {n: targets[n] - running[n] for n in ratios}
        chosen = max(sorted(deficits), key=lambda n: deficits[n])
        assigned[chosen].append(writer_id)
        running[chosen] += sample_counts[writer_id]

    empty = [n for n, ids in assigned.items() if not ids]
    if empty:
        raise SplitError(
            f"splits {empty} ended up empty. Too few writers ({len(writer_ids)}) "
            f"for ratios {ratios}."
        )

    return WriterSplit(
        name=name,
        seed=seed,
        ratios=ratios,
        writers={n: sorted(ids) for n, ids in assigned.items()},
    )


def counts_from_records(
    records: Sequence[Any],
    key: Callable[[Any], str] = lambda r: r.writer_id,
) -> dict[str, int]:
    """writer id -> number of records, ready to hand to :func:`make_split`."""
    counts: dict[str, int] = {}
    for record in records:
        counts[key(record)] = counts.get(key(record), 0) + 1
    return counts
