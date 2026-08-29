"""Report what data is present, what is missing, and what can be computed with it.

    python scripts/check_data.py

The point is to turn "is everything in place?" from a feeling into a command.
It reads only; it never downloads, extracts, moves or deletes anything.

Exit code is 0 when every required item is present, 1 otherwise, so it can gate a
run later without anyone having to remember to look.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from nib.config import get_path, load_config
from nib.data import cvl
from nib.data.cvl_words import scan_words
from nib.data.split import WriterSplit

OK = "  ok  "
MISSING = " miss "
PARTIAL = "partial"


@dataclass
class Check:
    label: str
    status: str
    detail: str
    required: bool = True

    @property
    def blocking(self) -> bool:
        return self.required and self.status != OK


def check_cvl(root: Path) -> list[Check]:
    if not root.is_dir() or not any(root.rglob("*.tif")):
        return [
            Check(
                "CVL images",
                MISSING,
                f"nothing under {root}. Get cvl-database-cropped-1-1.zip from "
                "zenodo.org/records/1492267 and extract it there.",
            )
        ]

    inv = cvl.scan(root)
    checks = [
        Check(
            "CVL images",
            OK if len(inv.pages) == 1604 else PARTIAL,
            f"{len(inv.pages)} pages, {len(inv.writers)} writers "
            f"(published: 1604 pages, 310 writers)",
        )
    ]

    # The XML in the full release holds only line geometry -- no text at all.
    # What actually carries the transcriptions is the cropped word images, whose
    # filenames encode them. So that is what gets checked.
    full = root / "cvl-database-1-1"
    if full.is_dir():
        words, report = scan_words(root)
        checks.append(
            Check(
                "CVL words",
                OK if len(words) > 90000 else PARTIAL,
                f"{report.kept} cropped words with transcriptions, "
                f"{len(report.writers)} writers "
                f"({report.total_seen - report.kept} excluded, see the report)",
            )
        )
    else:
        checks.append(
            Check(
                "CVL words",
                MISSING,
                "the full release is not extracted. Needs cvl-database-1-1.zip "
                "(4.2 GB) -- it carries the cropped word images whose filenames "
                "hold the transcriptions. Without it there is no CER baseline.",
            )
        )

    if inv.unparsed:
        checks.append(
            Check(
                "CVL filenames",
                PARTIAL,
                f"{len(inv.unparsed)} files did not parse, e.g. {inv.unparsed[0].name}",
                required=False,
            )
        )
    return checks


def check_split(split_path: Path, root: Path) -> Check:
    if not split_path.is_file():
        return Check("writer split", MISSING, f"not built yet: {split_path}")

    split = WriterSplit.load(split_path)
    counts = {name: len(ids) for name, ids in split.writers.items()}
    detail = f"{split.name!r} seed={split.seed} {counts}"

    if root.is_dir() and any(root.rglob("*.tif")):
        on_disk = cvl.scan(root).writers
        if split.all_writers != on_disk:
            only_split = len(split.all_writers - on_disk)
            only_disk = len(on_disk - split.all_writers)
            return Check(
                "writer split",
                PARTIAL,
                f"{detail} -- does NOT match the data on disk "
                f"({only_split} writers only in the split, {only_disk} only on disk). "
                "Rebuild it, or reported numbers stop being comparable.",
            )
    return Check("writer split", OK, detail)


def check_personal(root: Path) -> Check:
    images = [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}]
    if not images:
        return Check(
            "your handwriting",
            MISSING,
            f"no photos under {root}. Needs 4-5 phone photos of your own English "
            "handwriting, deliberately imperfect -- shadow, slight angle, room light.",
        )
    status = OK if len(images) >= 4 else PARTIAL
    return Check("your handwriting", status, f"{len(images)} photos")


def check_fixture(root: Path) -> Check:
    pngs = list(root.rglob("*.png")) if root.is_dir() else []
    if not pngs:
        return Check(
            "synthetic fixture",
            MISSING,
            "not built. Run: python scripts/make_fixture.py",
            required=False,
        )
    return Check("synthetic fixture", OK, f"{len(pngs)} images", required=False)


def check_iam(root: Path) -> Check:
    xml = list((root / "xml").glob("*.xml")) if (root / "xml").is_dir() else []
    if xml:
        return Check("IAM (optional)", OK, f"{len(xml)} XML files", required=False)
    archives = [p.name for p in root.glob("*.tgz")] if root.is_dir() else []
    if archives:
        return Check(
            "IAM (optional)", PARTIAL, f"archives present, not extracted: {archives}", False
        )
    return Check(
        "IAM (optional)",
        MISSING,
        "only needed to report numbers comparable with published papers",
        required=False,
    )


CAPABILITIES = [
    ("writer-disjoint split", ["CVL images"]),
    ("writer retrieval metric", ["CVL images"]),
    ("deception study controls", ["CVL images"]),
    ("normalisation work (T6)", ["CVL images", "your handwriting"]),
    ("FID reference set", ["CVL words"]),
    ("CER baseline", ["CVL words"]),
]


def main() -> int:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    root = get_path(cfg, "root")
    raw = get_path(cfg, "raw")

    checks: list[Check] = []
    checks += check_cvl(raw / "cvl")
    checks.append(
        check_split(root / "configs" / "splits" / "cvl-writer-disjoint.json", raw / "cvl")
    )
    checks.append(check_personal(get_path(cfg, "personal")))
    checks.append(check_fixture(get_path(cfg, "fixture")))
    checks.append(check_iam(raw / "iam"))

    print(f"data root: {raw}\n")
    width = max(len(c.label) for c in checks)
    for check in checks:
        print(f"[{check.status}] {check.label:<{width}}  {check.detail}")

    status_of = {c.label: c.status for c in checks}
    print("\nwhat this data supports:")
    for capability, needed in CAPABILITIES:
        ready = all(status_of.get(n) == OK for n in needed)
        mark = "yes" if ready else "no "
        blockers = (
            "" if ready else f"   <- needs {', '.join(n for n in needed if status_of.get(n) != OK)}"
        )
        print(f"  {mark}  {capability}{blockers}")

    blocking = [c for c in checks if c.blocking]
    if blocking:
        print(f"\n{len(blocking)} required item(s) missing.")
        return 1
    print("\nall required data present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
