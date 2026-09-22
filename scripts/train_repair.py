"""Train the stroke repair on real lines broken the way Emuru breaks them.

    python scripts/train_repair.py --device cuda

Lines by the training-split writers only -- none of the held-out writers every
evaluation is measured on. Each step takes random crops, breaks them through
Emuru's own VAE (:func:`nib.models.repair.weakening`), and teaches the network
to give back the real crop. A fixed 5% of the lines is kept aside, and at the end
their broken and mended versions are compared with the real ones on the stroke
statistics people see: pieces of ink per 100 columns, ink per column, stroke
width. Saves the weights for :func:`nib.models.repair.load`.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.engine.metrics import detectability
from nib.models import repair

INK_WEIGHT = 4.0
"""Extra weight on the real line's ink pixels: most of a line is paper, and a
loss dominated by paper would learn to leave everything alone."""

REPORTED = ("pieces_per_100_columns", "ink_per_column", "stroke_width")


def _crop(image: np.ndarray, width: int, rng: random.Random) -> np.ndarray:
    if image.shape[1] < width:
        image = np.pad(image, ((0, 0), (0, width - image.shape[1])), constant_values=255)
    start = rng.randrange(0, image.shape[1] - width + 1)
    return image[:, start : start + width]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--crop", type=int, default=256, help="pixels wide, a multiple of 8")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args, overrides = parser.parse_known_args(argv)

    import torch

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    seed = int(cfg.seed)
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    torch.manual_seed(seed)

    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{int(cfg.data.image_height)}.lmdb")
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    by_writer = pack.writers()
    keys = sorted(k for w in split.writers["train"] if w in by_writer for k in by_writer[w])
    rng.shuffle(keys)
    aside = max(8, len(keys) // 20)
    lines = [pack[k].image for k in keys]
    train, held = lines[aside:], lines[:aside]
    pack.close()
    print(
        f"lines     {len(train)} to train on, {len(held)} kept aside, training-split writers only"
    )

    vae = repair.load_vae(args.device)
    network = repair.build_network().to(args.device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    started, losses = time.perf_counter(), []
    for step in range(1, args.steps + 1):
        crops = np.stack([_crop(rng.choice(train), args.crop, rng) for _ in range(args.batch)])
        target = torch.from_numpy(crops.astype(np.float32) / 255.0)[:, None].to(args.device)
        broken = repair.break_like_emuru(vae, target, np_rng)
        mended = network(broken)
        weight = 1.0 + INK_WEIGHT * (target < detectability.INK / 255.0)
        loss = (weight * (mended - target).abs()).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(float(loss.detach()))
        if step % 250 == 0 or step == args.steps:
            print(
                f"  step {step:>5}  loss {np.mean(losses[-250:]):.4f}  "
                f"[{time.perf_counter() - started:.0f}s]",
                flush=True,
            )

    print("\nKEPT-ASIDE LINES -- real, broken like Emuru, and mended")
    columns = [detectability.NAMES.index(name) for name in REPORTED]
    table = {"real": [], "broken": [], "mended": []}
    for image in held:
        target = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(args.device)
        target = torch.nn.functional.pad(target, (0, -image.shape[1] % 8), value=1.0)
        broken = repair.break_like_emuru(vae, target, np_rng)
        broken = (broken[0, 0, :, : image.shape[1]].cpu().numpy() * 255).astype(np.uint8)
        for name, line in (
            ("real", image),
            ("broken", broken),
            ("mended", repair.repair(network, broken, args.device)),
        ):
            values = detectability.features(line)
            if values is not None:
                table[name].append(values[columns])
    for name, rows in table.items():
        medians = np.median(np.array(rows), axis=0)
        print(
            f"  {name:<7} "
            + "   ".join(f"{k} {v:.2f}" for k, v in zip(REPORTED, medians, strict=True))
        )

    ensure_dirs(cfg, "outputs")
    out = get_path(cfg, "outputs") / "repair" / "repair.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), out)
    print(f"\nweights   {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
