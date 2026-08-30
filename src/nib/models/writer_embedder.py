"""A small network that turns handwriting into a style vector.

Built because the writer-retrieval metric, wired to ImageNet Inception, scored
**3.7% top-1 on real handwriting** against 0.9% chance. Inception describes
photographic texture; it does not describe how a person forms letters. A metric
that cannot tell real writers apart cannot detect a generator that ignores its
style input, which is the one thing that metric exists to catch.

**Trained on writers, evaluated on writers it has never seen.** That is not a
detail -- it is the whole test. A network that identifies its training writers
proves nothing, because the product's job is to handle a person who has just
uploaded a photo. So the classification head exists only during training and is
thrown away; what survives is the embedding, and it is judged by nearest-neighbour
retrieval among the 94 held-out writers.

Deliberately small. This is a measuring instrument, not a contribution: it has to
be trainable on a laptop CPU in minutes, or it becomes another thing that only
works on Colab and therefore rarely gets rerun.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


class EmbedderError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbedderConfig:
    embedding_dim: int = 128
    width: int = 32
    """Channels in the first block. Doubling per block, so 32 gives 32/64/128/256."""
    dropout: float = 0.2


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise EmbedderError(
            "PyTorch is not installed. pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu"
        )


class WriterEmbedder(nn.Module if _HAS_TORCH else object):  # type: ignore[misc]
    """Grayscale word image in, unit-length style vector out.

    Width is pooled away rather than cropped or resized. Words differ in length
    and that length is a property of the *word*, not of the hand -- a style
    representation that changed with the word would be measuring the wrong thing.
    """

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or EmbedderConfig()
        w = self.config.width

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, w), block(w, w * 2), block(w * 2, w * 4))
        self.dropout = nn.Dropout(self.config.dropout)
        self.project = nn.Linear(w * 8, self.config.embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H, W) in [0, 1]. Returns (B, D), unit length."""
        x = self.features(images)

        # Mean *and* max over width, concatenated. Mean alone washes out a rare
        # distinctive stroke; max alone is decided by a single pixel. Together
        # they survive the variable width that makes this awkward in the first
        # place.
        x = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        x = self.project(self.dropout(x))
        return nn.functional.normalize(x, dim=1)


class WriterClassifier(nn.Module if _HAS_TORCH else object):  # type: ignore[misc]
    """Embedder plus a training-only head over the training writers.

    The head is discarded afterwards. Keeping it would make the model useless for
    the actual task: its outputs are fixed to writers seen during training, and
    every writer that matters is one it has never seen.
    """

    def __init__(self, num_writers: int, config: EmbedderConfig | None = None) -> None:
        _require_torch()
        super().__init__()
        self.embedder = WriterEmbedder(config)
        self.head = nn.Linear(self.embedder.config.embedding_dim, num_writers)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Embeddings are unit length, so raw logits would be bounded by 1 and the
        # loss could never sharpen. A learned scale is the standard fix.
        return self.head(self.embedder(images)) * self.scale


class TorchEmbedderAdapter:
    """Wraps a trained embedder to satisfy the ``Embedder`` protocol in metrics."""

    def __init__(self, embedder: WriterEmbedder, device: str = "cpu", batch_size: int = 128):
        _require_torch()
        self.embedder = embedder.eval().to(torch.device(device))
        self.device = torch.device(device)
        self.batch_size = batch_size

    def __call__(self, images) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                chunk = images[start : start + self.batch_size]
                out.append(self.embedder(_stack(chunk).to(self.device)).cpu().numpy())
        return np.concatenate(out, axis=0)


def _stack(images) -> torch.Tensor:
    """Pad a ragged list of grayscale images into one batch.

    Padded with white, for the same reason as everywhere else in this project:
    zero is solid ink here, so a naive pad would append a black bar and the
    embedding would describe the padding.
    """
    arrays = []
    for image in images:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 2:
            raise EmbedderError(f"expected a grayscale image, got shape {array.shape}")
        if array.max() > 1.5:
            array = array / 255.0
        arrays.append(array)

    height = arrays[0].shape[0]
    if any(a.shape[0] != height for a in arrays):
        raise EmbedderError("every image in a batch must have the same height")

    width = max(a.shape[1] for a in arrays)
    batch = np.ones((len(arrays), 1, height, width), dtype=np.float32)
    for i, array in enumerate(arrays):
        batch[i, 0, :, : array.shape[1]] = array
    return torch.from_numpy(batch)
