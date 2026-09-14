"""Show and count what line segmentation does to real photographs.

    python scripts/check_segmentation.py

For every photo under the personal data directory: normalise the page, split it
into lines, and write two pictures to `outputs/segmentation/` -- the page with
each line's box drawn on it, and the lines themselves stacked at the generator's
height. Prints how many lines each photo gave and what was dropped, and why.

The same page photographed several ways must give the same lines, so a count
that differs between photos is the first thing to look at. Exits non-zero when
it does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data.preprocessing import normalise_line, normalise_page
from nib.data.segmentation import Segmentation, split_lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="line height in the stacked sheet (default: the generator's)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    height = args.height or int(cfg.data.image_height)
    source = get_path(cfg, "personal")
    photos = sorted(p for p in source.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not photos:
        print(f"no photos under {source}")
        return 1

    ensure_dirs(cfg, "outputs")
    out_dir = get_path(cfg, "outputs") / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for path in photos:
        image = cv2.imread(str(path))
        if image is None:
            print(f"could not read {path.name}")
            continue
        page = normalise_page(image)
        result = split_lines(page)
        counts[path.stem] = len(result.lines)

        dropped = ", ".join(f"{reason} {n}" for reason, n in result.dropped.items() if n)
        print(f"{path.stem:<40} {len(result.lines):>3} lines   dropped: {dropped or 'nothing'}")

        cv2.imwrite(str(out_dir / f"{path.stem}_boxes.png"), _boxes(page, result))
        stacked = _stack([normalise_line(line.image, height) for line in result.lines])
        cv2.imwrite(str(out_dir / f"{path.stem}_lines.png"), stacked)

    distinct = sorted(set(counts.values()))
    print(f"\nline counts across photos: {distinct}")
    print(f"pictures in {out_dir}")
    return 0 if len(distinct) == 1 else 1


def _boxes(page: np.ndarray, result: Segmentation) -> np.ndarray:
    canvas = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    for number, line in enumerate(result.lines, start=1):
        x, y, w, h = line.box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            str(number),
            (max(0, x - 30), y + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return canvas


def _stack(lines: list[np.ndarray], gap: int = 8) -> np.ndarray:
    if not lines:
        return np.full((10, 10), 255, dtype=np.uint8)
    width = max(line.shape[1] for line in lines)
    rows: list[np.ndarray] = []
    for line in lines:
        row = np.full((line.shape[0], width), 255, dtype=np.uint8)
        row[:, : line.shape[1]] = line
        rows += [row, np.full((gap, width), 200, dtype=np.uint8)]
    return np.vstack(rows[:-1])


if __name__ == "__main__":
    sys.exit(main())
