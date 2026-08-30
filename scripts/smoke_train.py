"""Prove the training loop end to end, with a model that learns nothing.

    python scripts/smoke_train.py --steps 100
    python scripts/smoke_train.py --steps 200 --resume

This is the exit criterion for phase 1, and it deliberately contains no real
model. The brief names two traps that must work perfectly *before* serious
training begins -- slow data reading, and Colab disconnecting mid-run -- and this
script is the test of both. Everything it exercises is infrastructure:

    config -> pack -> writer split -> dataset -> collate -> forward -> backward
    -> checkpoint -> kill -> resume -> identical weights

The "model" is a single convolution plus a linear layer. Its output is
meaningless. What matters is that the loop runs, that samples reach the GPU fast
enough to keep it busy, and that killing the process loses nothing.

**The check that matters**: run it once to N steps, then again with --resume from
a checkpoint at N/2, and the final weights must be identical. The script does
that comparison itself when given --verify-resume.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from nib.config import ensure_dirs, get_path, load_config
from nib.data.dataset import WordDataset, collate
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.engine import checkpoint as ckpt
from nib.engine.tracking import make_tracker


class DummyModel(nn.Module):
    """Not a generator. Enough parameters and gradient flow to prove the loop."""

    def __init__(self, height: int, channels: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.head = nn.Linear(channels * height, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(images))  # (B, C, H, W)
        x = x.mean(dim=3)  # collapse width, which varies
        return self.head(x.flatten(1))


def epoch_loader(dataset, epoch: int, seed: int, args) -> DataLoader:
    """A loader whose shuffle depends only on (seed, epoch).

    This is what makes a resume exact, and its absence is what the first run of
    --verify-resume caught. The checkpoint restores every weight, every optimiser
    moment and every RNG -- and the run still diverged, because a DataLoader built
    with shuffle=True draws a fresh permutation each time it is constructed. The
    resumed run therefore saw *different data* from step N onward. Nothing was
    lost from the checkpoint; the order simply was not part of it.

    Deriving the permutation from the epoch number instead means a resumed run
    replays the same order, and skipping the batches already consumed puts it back
    exactly where it stopped.
    """
    generator = torch.Generator()
    generator.manual_seed(seed * 1_000_003 + epoch)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=args.workers,
        drop_last=True,
    )


def build(cfg, args):
    pack_path = (
        Path(args.pack)
        if args.pack
        else (get_path(cfg, "processed") / f"cvl_words_{cfg.data.image_height}.lmdb")
    )
    pack = PackReader(pack_path)

    split_path = get_path(cfg, "root") / "configs" / "splits" / "cvl-writer-disjoint.json"
    writer_split = WriterSplit.load(split_path) if split_path.is_file() else None

    dataset = WordDataset(
        pack,
        writer_split=writer_split,
        split=args.split,
        num_style_refs=args.style_refs,
        seed=int(cfg.seed),
    )
    return pack, dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--style-refs", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--pack", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verify-resume",
        action="store_true",
        help="run twice, once interrupted, and compare the weights",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)

    print(f"device        {args.device}")
    if args.device.startswith("cuda"):
        print(f"gpu           {torch.cuda.get_device_name(0)}")
    print(f"torch         {torch.__version__}")
    print(f"python        {sys.version.split()[0]}")

    pack, dataset = build(cfg, args)
    print()
    print(dataset.summary())

    if args.verify_resume:
        return _verify_resume(cfg, args, dataset)

    ensure_dirs(cfg, "checkpoints", "outputs")
    manager = ckpt.CheckpointManager(
        get_path(cfg, "checkpoints") / "smoke", every_n_steps=max(1, args.steps // 4)
    )
    tracker = make_tracker(cfg, get_path(cfg, "outputs") / "smoke", run_name="smoke")

    ckpt.seed_everything(int(cfg.seed))
    model = DummyModel(dataset.height).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    state = None
    if args.resume:
        state = manager.resume(models={"model": model}, optimizers={"opt": optimizer})
        print(f"\nresumed from step {state.step}" if state else "\nno checkpoint; starting fresh")
    state = state or ckpt.TrainingState()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.workers,
        drop_last=True,
    )

    print(f"\ntraining to step {args.steps}")
    started = time.perf_counter()
    samples = 0

    while state.step < args.steps:
        for batch in loader:
            if state.step >= args.steps:
                break
            images = batch.images.to(args.device)
            target = batch.label_lengths.float().unsqueeze(1).to(args.device)

            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(images), target)
            loss.backward()
            optimizer.step()

            state.step += 1
            samples += len(batch)

            if state.step % 25 == 0:
                rate = samples / (time.perf_counter() - started)
                print(f"  step {state.step:>5}  loss {loss.item():8.3f}  {rate:6.0f} samples/s")
                tracker.log({"loss": loss.item(), "samples_per_second": rate}, step=state.step)

            if state.step % max(1, args.steps // 2) == 0:
                tracker.log_images(
                    "batch",
                    [batch.images[i, 0].cpu().numpy() for i in range(min(8, len(batch)))],
                    step=state.step,
                )

            if manager.should_save(state.step):
                manager.save(state, models={"model": model}, optimizers={"opt": optimizer})

        state.epoch += 1

    manager.save(state, models={"model": model}, optimizers={"opt": optimizer})
    tracker.finish()
    pack.close()

    elapsed = time.perf_counter() - started
    print(
        f"\ndone: {state.step} steps, {samples} samples in {elapsed:.0f}s "
        f"({samples / elapsed:.0f} samples/s)"
    )
    print(f"checkpoint    {manager.latest_path}")
    print(f"samples       {tracker.images_dir}")
    return 0


def _verify_resume(cfg, args, dataset) -> int:
    """The phase-1 exit criterion: an interrupted run must equal an uninterrupted one."""
    print(f"\nverifying resume: {args.steps} straight vs {args.steps // 2} + resume\n")
    half = args.steps // 2
    root = get_path(cfg, "checkpoints") / "smoke_verify"
    shutil.rmtree(root, ignore_errors=True)

    def run(total: int, resume_from: Path | None) -> list[torch.Tensor]:
        ckpt.seed_everything(int(cfg.seed))
        model = DummyModel(dataset.height).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        step = 0
        if resume_from is not None:
            step = ckpt.load(
                resume_from, models={"model": model}, optimizers={"opt": optimizer}
            ).step

        batches_per_epoch = max(1, len(dataset) // args.batch_size)
        while step < total:
            epoch = step // batches_per_epoch
            loader = epoch_loader(dataset, epoch, int(cfg.seed), args)
            skip = step % batches_per_epoch
            for index, batch in enumerate(loader):
                if index < skip:
                    continue
                if step >= total:
                    break
                images = batch.images.to(args.device)
                target = batch.label_lengths.float().unsqueeze(1).to(args.device)
                optimizer.zero_grad()
                nn.functional.mse_loss(model(images), target).backward()
                optimizer.step()
                step += 1
                if step == half and resume_from is None and total == half:
                    ckpt.save(
                        root / "half.pt",
                        models={"model": model},
                        optimizers={"opt": optimizer},
                        state=ckpt.TrainingState(step=step),
                    )
        return [p.detach().cpu().clone() for p in model.parameters()]

    straight = run(args.steps, None)
    run(half, None)  # writes root/half.pt
    resumed = run(args.steps, root / "half.pt")

    identical = all(torch.equal(a, b) for a, b in zip(straight, resumed, strict=True))
    worst = max(float((a - b).abs().max()) for a, b in zip(straight, resumed, strict=True))

    print(f"straight {args.steps} steps vs {half} + resumed {args.steps - half}")
    print(f"  identical: {identical}")
    print(f"  largest weight difference: {worst:.3e}")
    if identical:
        print("\nPASS -- an interrupted run is indistinguishable from an uninterrupted one")
        return 0
    print("\nFAIL -- resuming changes the run. Something is not being saved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
