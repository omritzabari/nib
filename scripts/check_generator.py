"""Draw a few lines with a generator, before an hour of GPU is spent on it.

    python scripts/check_generator.py --generator eruku --device cuda

Loads the model, draws the first few of the evaluation's own requests, and says
how long a line takes and what 150 of them would cost. No metrics, no reference,
nothing that can fail for a reason other than the model: this exists so that a
two-hour run fails in its third minute instead of its ninetieth.

Checks each draw for the three ways a wrapper goes wrong quietly: an empty image,
the wrong height, and a line far too narrow for its text. Saves the drawings
beside the writer's real line, to be looked at.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.models.candidates import predicted_width

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_generator import build_requests, load_generator

NARROW = 0.3
"""A draw narrower than this share of its predicted width is not a line of text."""

FULL_RUN = 150
"""Lines in the evaluation this is a pre-flight for."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="eruku")
    parser.add_argument("--lines", type=int, default=3)
    parser.add_argument(
        "--style-refs",
        type=int,
        default=1,
        help="style lines per draw, as the run will make them. Above 1 they are "
        "JOINED into one wide image, which is what the quality control never does "
        "-- it hands the model one line per draw -- and what broke Emuru in T27.",
    )
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)
    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{height}.lmdb")
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    held_out = [w for w in split.writers["test"] if w in pack.writers()]
    requests, truths, _ = build_requests(pack, held_out, args.style_refs, args.lines, int(cfg.seed))

    generator = load_generator(
        args.generator, args.device, height, cfg_scale=args.cfg_scale, cfg=cfg
    )
    print(f"generator          {generator.name}, output height {generator.output_height}px")

    rows, seconds, faults = [], [], []
    for index, (request, truth) in enumerate(zip(requests, truths, strict=True)):
        started = time.perf_counter()
        image = generator.generate([request])[0]
        seconds.append(time.perf_counter() - started)
        expected = predicted_width(request.style_images, request.style_texts, request.text)
        share = image.shape[1] / expected if expected else float("nan")
        print(
            f"  line {index + 1}: {seconds[-1]:5.1f}s  {image.shape[1]:>5}px "
            f"({share:.2f} of predicted)  {request.text[:44]!r}",
            flush=True,
        )
        if image.size == 0 or not image.shape[1]:
            faults.append(f"line {index + 1} came back empty")
        elif image.shape[0] != height:
            faults.append(f"line {index + 1} is {image.shape[0]}px tall, not {height}")
        elif expected and share < NARROW:
            faults.append(f"line {index + 1} is {share:.2f} of its predicted width")
        rows += [truth.image, np.asarray(image, dtype=np.uint8), np.full((6, 10), 200, np.uint8)]

    ensure_dirs(cfg, "outputs")
    out = get_path(cfg, "outputs") / f"check_{generator.name.replace('/', '-')}"
    out.mkdir(parents=True, exist_ok=True)
    widest = max(row.shape[1] for row in rows)
    sheet = np.vstack(
        [np.pad(row, ((2, 2), (0, widest - row.shape[1])), constant_values=255) for row in rows]
    )
    cv2.imwrite(str(out / "lines.png"), sheet)
    pack.close()

    average = float(np.mean(seconds))
    print(
        f"\na line takes       {average:.1f}s -> {average * FULL_RUN / 60:.0f} min for {FULL_RUN}"
    )
    print(f"drawings           {out / 'lines.png'}  (real above, generated below)")
    if faults:
        print("\nFAULTS")
        for fault in faults:
            print(f"  {fault}")
        return 1
    print("\nready: nothing is obviously wrong with this generator here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
