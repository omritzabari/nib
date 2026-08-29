"""Build the synthetic fixture dataset, and a preview image to eyeball it.

    python scripts/make_fixture.py
    python scripts/make_fixture.py fixture.num_writers=5 data.image_height=96

Every setting comes from configs/base.yaml and may be overridden on the command
line. The preview is a contact sheet: one row per writer, the same words in each
row, so that writer-to-writer difference and within-writer consistency are both
visible at a glance. If a row looks like a different writer from itself, the
fixture is broken.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data import fixture

PREVIEW_WORDS = ["the", "handwriting", "London", "1960", "would"]


def build_preview(root: Path, writers: list[str], height: int, seed: int) -> np.ndarray:
    """Contact sheet: one row per writer, the same words across every row."""
    rows = []
    for index, _ in enumerate(writers):
        style = fixture.style_for_writer(index, seed=seed)
        cells = [fixture.render_word(w, style, height=height, seed=seed) for w in PREVIEW_WORDS]
        gap = np.full((height, height // 3), 255, dtype=np.uint8)
        row = [x for cell in cells for x in (cell, gap)][:-1]
        rows.append(np.hstack(row))

    width = max(r.shape[1] for r in rows)
    padded = [
        cv2.copyMakeBorder(r, 6, 6, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=255)
        for r in rows
    ]
    return np.vstack(padded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--no-preview", action="store_true")
    args, overrides = parser.parse_known_args(argv)

    default_config = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    cfg = load_config(args.config or default_config, overrides=overrides)

    ensure_dirs(cfg, "fixture")
    root = get_path(cfg, "fixture")

    started = time.perf_counter()
    summary = fixture.build(
        root=root,
        num_writers=cfg.fixture.num_writers,
        words_per_writer=cfg.fixture.words_per_writer,
        height=cfg.data.image_height,
        seed=cfg.fixture.seed,
        charset_name=cfg.data.charset,
    )
    elapsed = time.perf_counter() - started

    print(f"fixture written to {summary.root}")
    print(f"  writers      {summary.num_writers}")
    print(f"  word images  {summary.num_words}")
    print(f"  height       {cfg.data.image_height}px")
    print(f"  seed         {cfg.fixture.seed}")
    print(f"  elapsed      {elapsed:.2f}s")

    if not args.no_preview:
        sheet = build_preview(root, summary.forms[:8], cfg.data.image_height, cfg.fixture.seed)
        preview_path = root / "preview.png"
        cv2.imwrite(str(preview_path), sheet)
        print(f"\npreview      {preview_path}")
        print("  one row per writer, same words in every row.")
        print("  rows should differ from each other and be consistent within themselves.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
