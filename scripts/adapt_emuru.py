"""Teach Emuru real handwriting once, for every user -- not one writer's hand.

    python scripts/adapt_emuru.py --steps 2000 --device cuda

Emuru was pre-trained on millions of lines rendered from fonts and has never seen
a pen. Its identity figure, 65% of what real lines carry, is what its imitation is
worth on a domain it never saw. This trains a single LoRA adapter on the **training
side** of the writer split -- 216 CVL writers, real ink -- so that the skill of
copying a hand is practised on hands.

It is not the per-writer fine-tune, which failed twice and for a structural reason:
Emuru copies the style line in front of it, so one writer's sixteen lines taught its
weights nothing. Here nothing about a particular writer is meant to be learned. The
adapter ships with the system; a user still brings their page and nothing is trained
at enrolment.

The held-out 94 writers are untouched, so every figure this project reports stays
honest. The adapter is written to ``paths.checkpoints`` and is loaded by
``evaluate_generator.py --adapter`` and ``probe_writer.py --adapter``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.models import adapters, finetune


def training_lines(pack, split, limit: int, seed: int):
    """Lines by writers on the training side, shuffled, at most ``limit`` of them."""
    by_writer = pack.writers()
    keys = sorted(k for w in split.writers["train"] for k in by_writer.get(w, ()))
    random.Random(seed).shuffle(keys)
    chosen = [pack[k] for k in keys[:limit]]
    writers = {sample.writer_id for sample in chosen}
    return chosen, len(keys), writers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--lines",
        type=int,
        default=4000,
        help="training lines to draw on, of the 6,000-odd the training writers have",
    )
    parser.add_argument("--noise", type=float, default=finetune.DEFAULT_CONFIG.noise)
    parser.add_argument("--name", default=None, help="adapter filename; default from the settings")
    parser.add_argument("--device", default="cuda")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)
    seed = int(cfg.seed)
    config = finetune.FinetuneConfig(
        rank=args.rank,
        alpha=2 * args.rank,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        noise=args.noise,
        seed=seed,
    )

    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{height}.lmdb")
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    lines, available, writers = training_lines(pack, split, args.lines, seed)
    if not lines:
        raise SystemExit("no training-split lines in this pack")
    print(f"lines     {len(lines)} of {available} by {len(writers)} training-split writers")
    print(f"held out  {len(split.writers['test'])} writers, untouched by this")
    print(f"adapter   {config}")
    print(f"passes    {args.steps * args.batch_size / len(lines):.1f} over the training lines")

    from nib.models.emuru import EmuruGenerator

    emuru = EmuruGenerator(device=args.device, output_height=height)
    trainable = finetune.attach_lora(emuru.model, config, device=args.device)
    total = sum(p.numel() for p in emuru.model.parameters())
    print(
        f"LoRA      {trainable / 1e6:.2f}M trainable of {total / 1e6:.0f}M ({trainable / total:.2%})"
    )

    started = time.perf_counter()
    report = finetune.train_writer(
        emuru.model,
        [line.image for line in lines],
        [line.text for line in lines],
        config,
        device=args.device,
    )
    print(f"\n{report.summary()}")
    tenth = max(1, len(report.losses) // 10)
    marks = [
        float(np.mean(report.losses[i : i + tenth])) for i in range(0, len(report.losses), tenth)
    ]
    print("  loss by tenth: " + "  ".join(f"{value:.4f}" for value in marks))

    ensure_dirs(cfg, "checkpoints")
    name = args.name or f"emuru_cvl_r{args.rank}_s{args.steps}.pt"
    path = adapters.save(emuru.model, get_path(cfg, "checkpoints") / name)
    log = path.with_suffix(".json")
    log.write_text(
        json.dumps(
            {
                "config": config.__dict__,
                "lines": len(lines),
                "writers": sorted(writers),
                "seconds": report.seconds,
                "losses": report.losses,
                "trainable_parameters": trainable,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    size = path.stat().st_size / 1e6
    print(f"\nadapter   {path}  ({size:.0f} MB, {time.perf_counter() - started:.0f}s in all)")
    print(f"log       {log}")
    print("\nMeasure it with:")
    print(
        f"  python scripts/evaluate_generator.py --generator emuru --samples 150 "
        f"--style-refs 4 --candidates 4 --adapter {path} --device cuda"
    )
    pack.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
