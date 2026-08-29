"""TrOCR: the pre-trained model that reads handwriting so CER can be computed.

Not part of the system being built. It is a judge, borrowed for measurement, and
the distinction matters: its mistakes are noise in our numbers, not faults in our
model, which is why every CER figure is reported next to this same model's score
on *real* handwriting.

The weights are downloaded once from HuggingFace (about 1.4 GB) and cached. On
Colab that download happens per session unless the cache directory points at
Drive, so ``cache_dir`` is exposed for exactly that.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"


class RecogniserError(RuntimeError):
    pass


class TrOcrRecogniser:
    """Reads word images. Satisfies the ``Recogniser`` protocol in ``cer``."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        cache_dir: Path | str | None = None,
        max_new_tokens: int = 24,
    ) -> None:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RecogniserError(
                "TrOCR needs torch and transformers. pip install transformers"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        cache = str(cache_dir) if cache_dir else None

        self.processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache)
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache)
            .eval()
            .to(self.device)
        )

    def read(self, images: Sequence[np.ndarray]) -> list[str]:
        """Transcribe a batch of word images.

        TrOCR expects RGB. Our images are grayscale ink-on-paper, in either
        [0, 1] floats or uint8, so both are handled here rather than leaving each
        caller to remember which it has.
        """
        if not images:
            return []

        rgb = [_to_rgb_uint8(image) for image in images]
        inputs = self.processor(images=rgb, return_tensors="pt").pixel_values.to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(inputs, max_new_tokens=self.max_new_tokens)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 3:
        return array.astype(np.uint8) if array.dtype != np.uint8 else array
    if array.ndim != 2:
        raise RecogniserError(f"expected a grayscale image, got shape {array.shape}")

    if array.dtype != np.uint8:
        # Floats may be [0, 1] from the dataset or [0, 255] from a raw decode.
        array = array.astype(np.float32)
        if array.max() <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.repeat(array[:, :, None], 3, axis=2)
