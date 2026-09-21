"""Can a trivial classifier tell a finished run's lines from real handwriting?

    python scripts/detectability.py outputs/eval_emuru_lines_refs4_cand4_byhand
    python scripts/detectability.py outputs/<before> outputs/<after>

Reads what a run saved -- ``per_sample.json`` and ``generated/`` -- and the real
line of each sample from the pack, and reports how well ten statistics of the
ink separate the two: 50% means they cannot be told apart, which is the goal.
No GPU, a second or two a run. See :mod:`nib.engine.metrics.detectability`.

Given two runs, it also says whether their figures separate at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from nib.config import get_path, load_config
from nib.data.pack import PackReader
from nib.engine.metrics import detectability


def judge(run: Path, pack: PackReader) -> detectability.Detectability:
    records = json.loads((run / "per_sample.json").read_text(encoding="utf-8"))
    real = [pack[record["key"]].image for record in records]
    generated = [
        cv2.imread(str(run / "generated" / f"{index:03d}.png"), cv2.IMREAD_GRAYSCALE)
        for index in range(len(records))
    ]
    return detectability.measure(real, generated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="evaluation output directories")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)
    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{height}.lmdb")

    results = []
    for run in args.runs:
        if not (run / "per_sample.json").is_file():
            print(f"no per_sample.json in {run}")
            return 1
        print("=" * 62)
        print(run.name)
        result = judge(run, pack)
        print(result.describe())
        results.append(result)

    if len(results) == 2:
        a, b = (r.accuracy for r in results)
        verdict = "SEPARATE" if a.separates_from(b) else "overlap -- cannot be told apart"
        print("=" * 62)
        print(f"{args.runs[0].name}  against  {args.runs[1].name}")
        print(f"  {a.format(as_percent=True)}  vs  {b.format(as_percent=True)}   {verdict}")
    pack.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
