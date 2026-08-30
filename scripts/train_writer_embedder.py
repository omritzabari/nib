"""Train the style embedding that the writer-retrieval metric needs.

    python scripts/train_writer_embedder.py --epochs 8

Exists because that metric, wired to ImageNet Inception, scored 3.7% top-1 on
real handwriting against 0.9% chance. Inception describes photographic texture,
not how a person forms letters.

**The evaluation is on writers the network has never seen.** Training writers come
from the train side of the committed split, and retrieval is scored on the 94
test-side writers. That is not a formality: a network that identifies its own
training writers proves nothing about a person who has just uploaded a photo, and
that person is the entire product.

The classification head is thrown away after training. What is kept is the
embedding, and the number that matters is top-1 retrieval on the held-out side.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nib.config import ensure_dirs, get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.engine import checkpoint as ckpt
from nib.engine.metrics.writer import RETRIEVAL_FLOOR, WriterRetrieval
from nib.models.writer_embedder import (
    EmbedderConfig,
    TorchEmbedderAdapter,
    WriterClassifier,
)


class WriterWords(Dataset):
    """Word images labelled by writer, from one side of the split."""

    def __init__(self, pack: PackReader, writers: list[str], min_words: int = 20):
        by_writer = pack.writers()
        kept = {
            w: sorted(k)
            for w in writers
            if len(by_writer.get(w, [])) >= min_words
            for k in [by_writer[w]]
        }
        if not kept:
            raise RuntimeError(f"no writer has {min_words} words")
        self.pack = pack
        self.writer_ids = sorted(kept)
        self.index = {w: i for i, w in enumerate(self.writer_ids)}
        self.items = [(k, self.index[w]) for w in self.writer_ids for k in kept[w]]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        key, label = self.items[i]
        image = self.pack[key].image.astype(np.float32) / 255.0
        return image, label


def collate(batch):
    """Pad to the widest, with white. Zero is ink here."""
    images, labels = zip(*batch, strict=True)
    height = images[0].shape[0]
    width = max(im.shape[1] for im in images)
    out = np.ones((len(images), 1, height, width), dtype=np.float32)
    for i, im in enumerate(images):
        out[i, 0, :, : im.shape[1]] = im
    return torch.from_numpy(out), torch.tensor(labels, dtype=torch.long)


def evaluate_retrieval(embedder, pack, writers, seed, device, per_writer=12):
    """Top-1 retrieval on writers the network never trained on.

    Gallery and query sets are disjoint words from the same writers, which is
    exactly the situation at inference: a few samples of someone's hand on one
    side, something new on the other.
    """
    import random

    rng = random.Random(seed)
    by_writer = pack.writers()
    gallery, gallery_ids, queries, query_ids = [], [], [], []

    for writer in writers:
        keys = sorted(by_writer.get(writer, []))
        if len(keys) < per_writer * 2:
            continue
        chosen = rng.sample(keys, per_writer * 2)
        for key in chosen[:per_writer]:
            gallery.append(pack[key].image)
            gallery_ids.append(writer)
        for key in chosen[per_writer:]:
            queries.append(pack[key].image)
            query_ids.append(writer)

    if len(set(gallery_ids)) < 5:
        raise RuntimeError("too few writers with enough words to evaluate")

    adapter = TorchEmbedderAdapter(embedder, device=device)
    retrieval = WriterRetrieval(adapter).fit(gallery, gallery_ids)
    return retrieval.evaluate(queries, query_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--also-save-to",
        nargs="*",
        default=[],
        metavar="DIR",
        help="extra directories to write the result to, saved in the same breath "
        "as the local copy. On Colab, point this at Drive: a session can recycle "
        "between finishing and a separate save cell, and then the run is gone.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--limit-per-writer",
        type=int,
        default=120,
        help="cap words per writer, to keep a CPU run to minutes",
    )
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    ckpt.seed_everything(int(cfg.seed))

    pack = PackReader(get_path(cfg, "processed") / f"cvl_words_{cfg.data.image_height}.lmdb")
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")

    train_writers = [w for w in split.writers["train"] if w in pack.writers()]
    test_writers = [w for w in split.writers["test"] if w in pack.writers()]

    dataset = WriterWords(pack, train_writers)
    if args.limit_per_writer:
        capped: dict[int, int] = {}
        items = []
        for key, label in dataset.items:
            if capped.get(label, 0) < args.limit_per_writer:
                items.append((key, label))
                capped[label] = capped.get(label, 0) + 1
        dataset.items = items

    print(f"device        {args.device}")
    print(f"train         {len(dataset)} words from {len(dataset.writer_ids)} writers")
    print(f"held out      {len(test_writers)} writers the network never sees")

    model = WriterClassifier(
        num_writers=len(dataset.writer_ids),
        config=EmbedderConfig(embedding_dim=args.embedding_dim),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.workers,
        drop_last=True,
    )

    print("\nbaseline before training:")
    before = evaluate_retrieval(model.embedder, pack, test_writers, int(cfg.seed), args.device)
    print(f"  top-1 {before.top1:.1%}   (chance {before.chance:.1%})")

    print(f"\ntraining {args.epochs} epochs")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, losses = 0, 0, []
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)

        model.eval()
        result = evaluate_retrieval(model.embedder, pack, test_writers, int(cfg.seed), args.device)
        print(
            f"  epoch {epoch:>2}  loss {np.mean(losses):6.3f}  "
            f"train acc {correct / total:5.1%}  "
            f"HELD-OUT top-1 {result.top1:6.1%}  top-5 {result.topk:6.1%}"
        )

    elapsed = time.perf_counter() - started
    print(f"\ntrained in {elapsed:.0f}s")

    # Saved to every destination given, not just the VM's local disk. An earlier
    # run put the save in a separate notebook cell, the Colab session recycled
    # between the two, and an hour of training went with it. A window in which a
    # disconnect destroys finished work is exactly what this project exists to
    # close, and leaving one here was careless.
    destinations = [get_path(cfg, "checkpoints") / "writer_embedder.pt"]
    for extra in args.also_save_to:
        destinations.append(Path(extra) / "writer_embedder.pt")

    ensure_dirs(cfg, "checkpoints")
    out = destinations[0]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        ckpt.save(
            destination,
            models={"embedder": model.embedder},
            state=ckpt.TrainingState(step=args.epochs, extra={"top1": result.top1}),
            config=cfg,
        )
        print(f"saved         {destination}")

    print()
    print(result.summary())
    print(
        f"\nbefore training: {before.top1:.1%}   after: {result.top1:.1%}   "
        f"(ImageNet Inception scored 3.7%)"
    )

    if result.top1 < RETRIEVAL_FLOOR:
        print(f"\nBELOW the {RETRIEVAL_FLOOR:.0%} floor the metric needs to be usable.")
        return 1
    print(f"\nPASS -- above the {RETRIEVAL_FLOOR:.0%} floor; the metric can now detect")
    print("a generator that ignores its style input.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
