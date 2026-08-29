"""Show and measure what normalisation does to real photographs.

    python scripts/check_normalisation.py
    python scripts/check_normalisation.py --no-ruling      # compare with ruling left in

Writes a before/after contact sheet to `outputs/normalisation/` and prints the
numbers underneath it.

The point of the numbers: "does it look better" is not a criterion anyone can
check twice. The same page photographed under different lighting should come out
*the same*, and that is measurable without any alignment between the images --
ink coverage, paper level, ink level and contrast should all converge. The
printed spread is max minus min across the conditions; smaller is better, and
zero means the condition made no difference at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data.preprocessing import _to_gray, normalise_page
from nib.engine.tracking import make_grid


def measure(gray: np.ndarray) -> dict[str, float]:
    """Statistics that should be identical across lighting conditions.

    Deliberately alignment-free. Comparing images pixel by pixel would mostly
    measure how well two crops happen to line up, which is not what normalisation
    is for.
    """
    paper = float(np.percentile(gray, 90))
    ink = float(np.percentile(gray, 2))
    return {
        "ink %": 100.0 * float((gray < 128).mean()),
        "paper": paper,
        "ink": ink,
        "contrast": paper - ink,
    }


def spread(rows: dict[str, dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows.values()]
    return max(values) - min(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ruling", action="store_true", help="leave the ruling in place")
    parser.add_argument("--height", type=int, default=560, help="contact sheet cell height")
    args = parser.parse_args(argv)

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    source = get_path(cfg, "personal")
    photos = sorted(p for p in source.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not photos:
        print(f"no photos under {source}")
        return 1

    ensure_dirs(cfg, "outputs")
    out_dir = get_path(cfg, "outputs") / "normalisation"
    out_dir.mkdir(parents=True, exist_ok=True)

    before_rows: dict[str, dict[str, float]] = {}
    after_rows: dict[str, dict[str, float]] = {}
    cells: list[np.ndarray] = []

    for path in photos:
        image = cv2.imread(str(path))
        if image is None:
            print(f"could not read {path.name}")
            continue
        name = path.stem

        raw = _to_gray(image)
        clean = normalise_page(image, remove_ruling=not args.no_ruling)

        before_rows[name] = measure(raw)
        after_rows[name] = measure(clean)
        cv2.imwrite(str(out_dir / f"{name}_normalised.png"), clean)

        cells.append(_fit(raw, args.height))
        cells.append(_fit(clean, args.height))

    sheet = make_grid(cells, columns=2, pad=8)
    sheet_path = out_dir / "before_after.png"
    cv2.imwrite(str(sheet_path), sheet)

    names = list(before_rows)
    keys = ["ink %", "paper", "ink", "contrast"]

    print(f"contact sheet: {sheet_path}")
    print("  left column is the original photo, right is the same page normalised\n")

    for title, rows in (("BEFORE", before_rows), ("AFTER", after_rows)):
        print(title)
        print(f"  {'':<12}" + "".join(f"{k:>11}" for k in keys))
        for name in names:
            print(f"  {name:<12}" + "".join(f"{rows[name][k]:11.1f}" for k in keys))
        print()

    print("spread across conditions (max - min). Lower means the lighting mattered less.")
    print(f"  {'metric':<12}{'before':>10}{'after':>10}")
    worse = []
    for key in keys:
        b, a = spread(before_rows, key), spread(after_rows, key)
        flag = "" if a <= b else "   <- WORSE"
        if a > b:
            worse.append(key)
        print(f"  {key:<12}{b:10.1f}{a:10.1f}{flag}")

    return 1 if worse else 0


def _fit(gray: np.ndarray, height: int) -> np.ndarray:
    scale = height / gray.shape[0]
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    sys.exit(main())
