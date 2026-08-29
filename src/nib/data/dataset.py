"""Serve style-conditioned training samples from a pack.

A sample is not one image. It is *a target word, plus a handful of other words by
the same person*, because that is the shape of the task: look at a few examples
of someone's hand, then write something they never wrote.

Two properties make or break this, and neither of them raises when broken:

**The style references must come from the same writer as the target.** Otherwise
the model is conditioned on noise and learns to ignore the style input entirely.

**They must be *different words* from the target.** If the target word can appear
among its own references, the model can copy instead of generalise. Loss falls,
samples look wonderful, and the whole few-shot claim is hollow. This is the
single most important thing in the file, and it has a test of its own.

Widths vary because words vary, so batching pads to the widest and carries a mask.
Padding with white rather than zero matters: zero is *black* in this encoding, so
a naive pad puts a solid black bar next to every short word and the model learns
to draw it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nib.data import charset as cs
from nib.data.pack import PackReader
from nib.data.split import WriterSplit

try:
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional extra
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False

PAD_VALUE = 1.0
"""White. Images are stored as ink-on-paper and scaled to [0, 1], so 1.0 is blank
paper and 0.0 is solid ink. Padding with 0.0 would draw a black bar."""


class DatasetError(RuntimeError):
    pass


@dataclass
class Sample:
    """One training example, before batching."""

    key: str
    writer_id: str
    text: str
    image: np.ndarray  # (H, W) float32 in [0, 1]
    label: list[int]
    style: list[np.ndarray]  # num_style_refs images, same writer, other words
    style_keys: list[str]
    """Which records the references came from.

    Carried explicitly because the two properties that matter -- same writer,
    different word -- are otherwise unverifiable: a test could only compare
    pixels, and would silently pass on a reference that merely happened to look
    different. It is also the first thing you want when a sample looks wrong."""


class WordDataset(Dataset):
    """Style-conditioned words from a pack, restricted to one side of the split."""

    def __init__(
        self,
        pack: PackReader | Path | str,
        writer_split: WriterSplit | None = None,
        split: str = "train",
        charset_name: str | None = None,
        num_style_refs: int = 5,
        seed: int = 1337,
        min_words_per_writer: int | None = None,
    ) -> None:
        self.pack = pack if isinstance(pack, PackReader) else PackReader(pack)
        self.num_style_refs = num_style_refs
        self.seed = seed
        self.charset = cs.get(charset_name or self.pack.header.charset)
        self.height = self.pack.header.height

        by_writer = self.pack.writers()
        if writer_split is not None:
            if split not in writer_split.writers:
                raise DatasetError(f"split {split!r} is not in {sorted(writer_split.writers)}")
            allowed = set(writer_split.writers[split])
            by_writer = {w: keys for w, keys in by_writer.items() if w in allowed}
            if not by_writer:
                raise DatasetError(
                    f"no writers from split {split!r} are present in {self.pack.path}"
                )

        # A writer needs the target plus its references, or there is nothing to
        # condition on. Dropping them here, loudly, beats an IndexError at step
        # 40,000 of a training run.
        floor = min_words_per_writer if min_words_per_writer is not None else num_style_refs + 1
        self.dropped_writers = {w: len(k) for w, k in by_writer.items() if len(k) < floor}
        self.by_writer = {w: sorted(k) for w, k in by_writer.items() if len(k) >= floor}
        if not self.by_writer:
            raise DatasetError(
                f"every writer has fewer than {floor} words, so none can be styled from"
            )

        self.keys: list[str] = sorted(k for keys in self.by_writer.values() for k in keys)
        self.writer_of = {k: w for w, keys in self.by_writer.items() for k in keys}

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def writers(self) -> list[str]:
        return sorted(self.by_writer)

    def __getitem__(self, index: int) -> Sample:
        key = self.keys[index]
        writer_id = self.writer_of[key]
        record = self.pack[key]

        # Seeded on (seed, key) rather than on a shared generator: the references
        # for a given sample are then the same whether the DataLoader has one
        # worker or eight, which is what makes an epoch reproducible.
        rng = random.Random(f"{self.seed}:{key}")
        pool = [k for k in self.by_writer[writer_id] if k != key]
        chosen = rng.sample(pool, min(self.num_style_refs, len(pool)))
        while len(chosen) < self.num_style_refs:  # tiny writers: allow repeats
            chosen.append(rng.choice(pool))

        return Sample(
            key=key,
            writer_id=writer_id,
            text=record.text,
            image=_to_float(record.image),
            label=self.charset.encode(record.text),
            style=[_to_float(self.pack[k].image) for k in chosen],
            style_keys=chosen,
        )

    def summary(self) -> str:
        counts = [len(v) for v in self.by_writer.values()]
        lines = [
            f"words        {len(self)}",
            f"writers      {len(self.by_writer)}",
            f"words/writer min {min(counts)}  median {int(np.median(counts))}  max {max(counts)}",
            f"style refs   {self.num_style_refs}",
        ]
        if self.dropped_writers:
            lines.append(
                f"dropped      {len(self.dropped_writers)} writers with too few words: "
                f"{dict(list(self.dropped_writers.items())[:5])}"
            )
        return "\n".join(lines)


def _to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


@dataclass
class Batch:
    """A padded batch. Every tensor carries the mask needed to ignore the padding."""

    images: torch.Tensor  # (B, 1, H, W)
    image_widths: torch.Tensor  # (B,)
    image_mask: torch.Tensor  # (B, W) True where real
    labels: torch.Tensor  # (B, L)
    label_lengths: torch.Tensor  # (B,)
    style: torch.Tensor  # (B, K, 1, H, Ws)
    style_mask: torch.Tensor  # (B, K, Ws)
    writer_ids: list[str]
    texts: list[str]
    keys: list[str]

    def __len__(self) -> int:
        return len(self.keys)


def collate(samples: list[Sample]) -> Batch:
    """Pad a list of samples into rectangular tensors.

    Padding is white, not zero. In this encoding zero is solid ink, so padding
    with zeros would paint a black bar beside every short word -- and the model
    would dutifully learn to reproduce it.
    """
    if not _HAS_TORCH:
        raise DatasetError(
            "PyTorch is not installed. Locally: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    if not samples:
        raise DatasetError("cannot collate an empty batch")

    height = samples[0].image.shape[0]
    if any(s.image.shape[0] != height for s in samples):
        raise DatasetError("every image in a batch must have the same height")

    widths = [s.image.shape[1] for s in samples]
    max_width = max(widths)
    images = np.full((len(samples), 1, height, max_width), PAD_VALUE, dtype=np.float32)
    image_mask = np.zeros((len(samples), max_width), dtype=bool)
    for i, sample in enumerate(samples):
        w = sample.image.shape[1]
        images[i, 0, :, :w] = sample.image
        image_mask[i, :w] = True

    lengths = [len(s.label) for s in samples]
    max_length = max(lengths) if any(lengths) else 1
    labels = np.full((len(samples), max_length), cs.PAD_INDEX, dtype=np.int64)
    for i, sample in enumerate(samples):
        labels[i, : len(sample.label)] = sample.label

    refs = len(samples[0].style)
    if any(len(s.style) != refs for s in samples):
        raise DatasetError("every sample in a batch must carry the same number of style refs")
    style_width = max(ref.shape[1] for s in samples for ref in s.style)
    style = np.full((len(samples), refs, 1, height, style_width), PAD_VALUE, dtype=np.float32)
    style_mask = np.zeros((len(samples), refs, style_width), dtype=bool)
    for i, sample in enumerate(samples):
        for j, ref in enumerate(sample.style):
            w = ref.shape[1]
            style[i, j, 0, :, :w] = ref
            style_mask[i, j, :w] = True

    return Batch(
        images=torch.from_numpy(images),
        image_widths=torch.tensor(widths, dtype=torch.long),
        image_mask=torch.from_numpy(image_mask),
        labels=torch.from_numpy(labels),
        label_lengths=torch.tensor(lengths, dtype=torch.long),
        style=torch.from_numpy(style),
        style_mask=torch.from_numpy(style_mask),
        writer_ids=[s.writer_id for s in samples],
        texts=[s.text for s in samples],
        keys=[s.key for s in samples],
    )
