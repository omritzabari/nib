"""Interrogate a finished evaluation without a GPU.

    python scripts/analyse_run.py outputs/eval_eruku_lines
    python scripts/analyse_run.py outputs/eval_eruku_lines outputs/eval_emuru_lines

Every run saves ``analysis.npz`` and ``per_sample.json``, which hold the terms
its metrics were computed from rather than only the conclusions. This reads them
back, so a question that occurs after the run does not cost another hour of GPU
to answer.

Three things it reports.

**Width.** How wide the model wrote, against how wide the writer really wrote
the same words. This is not a curiosity: scale is part of a hand, so a model
that systematically writes larger than its reference is failing at style in a
way that FID and writer retrieval both punish without either of them naming it.
It also explains truncation, since a line written twice as wide runs out of
budget at half the length.

**Intervals, recomputed.** Same figures the run printed, from the same terms,
which is a check that the saved bundle really does reconstruct the result.

**Comparison.** Given two runs, whether their intervals separate. Two numbers
whose intervals overlap are not a finding, and this is the question that made
the bundle worth saving in the first place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nib.engine.metrics import bootstrap


def load(directory: Path) -> tuple[dict, list[dict]]:
    arrays = dict(np.load(directory / "analysis.npz"))
    records = json.loads((directory / "per_sample.json").read_text(encoding="utf-8"))
    return arrays, records


def report_widths(records: list[dict]) -> None:
    real = np.array([r["real_width"] for r in records], dtype=float)
    generated = np.array([r["generated_width"] for r in records], dtype=float)
    ratio = generated / np.maximum(real, 1.0)

    print("\nwidth, generated against the real line of the same words")
    print(f"  real       mean {real.mean():6.0f}px   median {np.median(real):6.0f}px")
    print(f"  generated  mean {generated.mean():6.0f}px   median {np.median(generated):6.0f}px")
    print(
        f"  ratio      mean {ratio.mean():5.2f}x    median {np.median(ratio):5.2f}x"
        f"    p10 {np.percentile(ratio, 10):4.2f}   p90 {np.percentile(ratio, 90):4.2f}"
    )

    wider = float((ratio > 1.15).mean())
    narrower = float((ratio < 0.87).mean())
    print(f"  {wider:.0%} more than 15% too wide, {narrower:.0%} more than 15% too narrow")
    if abs(np.median(ratio) - 1.0) > 0.15:
        direction = "wider" if np.median(ratio) > 1 else "narrower"
        print(
            f"  -> systematically {direction} than the hand it is copying. Scale is part\n"
            f"     of a hand, so this costs writer retrieval and FID without either\n"
            f"     metric saying the word 'scale'."
        )


def report_metrics(arrays: dict, label: str) -> dict:
    print(f"\n{label}")
    out: dict = {}

    if "features_real" in arrays:
        out["fid"] = bootstrap.fid_interval(arrays["features_real"], arrays["features_generated"])
        print(f"  FID            {out['fid'].format()}")
    if "hits_top1" in arrays:
        out["top1"] = bootstrap.rate_interval(arrays["hits_top1"])
        out["top5"] = bootstrap.rate_interval(arrays["hits_topk"])
        print(f"  writer top-1   {out['top1'].format(as_percent=True)}")
        print(f"  writer top-5   {out['top5'].format(as_percent=True)}")
    if "cer_errors" in arrays:
        out["cer"] = bootstrap.cer_interval(arrays["cer_errors"], arrays["cer_lengths"])
        out["cer_real"] = bootstrap.cer_interval(arrays["cer_real_errors"], arrays["cer_lengths"])
        print(f"  CER            {out['cer'].format(as_percent=True)}")
        print(f"  CER on real    {out['cer_real'].format(as_percent=True)}")
    return out


def compare(first: dict, second: dict, names: tuple[str, str]) -> None:
    print(f"\n{'=' * 62}\n{names[0]} against {names[1]}")
    print("  a difference whose intervals overlap is not a difference\n")

    for key, label, percent in (
        ("fid", "FID", False),
        ("top1", "writer top-1", True),
        ("top5", "writer top-5", True),
        ("cer", "CER", True),
    ):
        if key not in first or key not in second:
            continue
        a, b = first[key], second[key]
        verdict = "SEPARATE" if a.separates_from(b) else "overlap -- cannot be told apart"
        left = a.format(as_percent=percent)
        right = b.format(as_percent=percent)
        print(f"  {label:<14} {left:>26}   vs {right:<26}  {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="evaluation output directories")
    args = parser.parse_args(argv)

    measured = []
    for directory in args.runs:
        if not (directory / "analysis.npz").is_file():
            print(f"no analysis.npz in {directory}. Runs before 2026-09-10 did not save one.")
            return 1
        arrays, records = load(directory)
        print("=" * 62)
        print(f"{directory.name}   {len(records)} samples")
        report_widths(records)
        measured.append(report_metrics(arrays, "metrics, recomputed from the saved terms"))

    if len(measured) == 2:
        compare(measured[0], measured[1], (args.runs[0].name, args.runs[1].name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
