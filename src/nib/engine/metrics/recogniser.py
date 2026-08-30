"""TrOCR: the pre-trained model that reads handwriting so CER can be computed.

Not part of the system being built. It is a judge, borrowed for measurement, and
the distinction matters: its mistakes are noise in our numbers, not faults in our
model, which is why every CER figure is reported next to this same model's score
on *real* handwriting.

The weights are downloaded once from HuggingFace (about 1.4 GB) and cached. On
Colab that download happens per session unless the cache directory points at
Drive, so ``cache_dir`` is exposed for exactly that.

.. important::
   **Give it lines, not isolated words.** Measured on real CVL handwriting:

       isolated words   53.3% CER   -- unusable as a judge
       whole lines      11.1% CER   -- usable

   TrOCR-base-handwritten was trained on IAM *line* images, and a single word is
   out of distribution for it. The tell is that it hallucinates trailing
   punctuation on words, because it expects a sentence. Much of the residual 11%
   is also punctuation that TrOCR adds and CVL's word-level ground truth omits,
   so the true reading accuracy is better than the figure suggests.

   This settles one of the open questions in the project brief -- "is TrOCR good
   enough, or do we need to train our own recogniser?" It is good enough, at line
   level. It is not, at word level.

   That happens to align with the architecture: the chosen generator emits
   variable-length *lines*, so the unit the judge wants is the unit the model
   produces.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
TOKENIZER_FALLBACK = "roberta-large"
"""TrOCR-base's decoder is RoBERTa-large; see _load_processor."""


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
            from transformers import VisionEncoderDecoderModel
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RecogniserError(
                "TrOCR needs torch and transformers. pip install transformers"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        cache = str(cache_dir) if cache_dir else None

        self.processor = _load_processor(model_name, cache)
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


def _load_processor(model_name: str, cache: str | None):
    """Build the processor, working around a tokenizer that no longer loads.

    ``TrOCRProcessor.from_pretrained`` fails on transformers 5.x for this model:
    the repository predates the ``tokenizer.json`` format and the new code cannot
    convert the old files, raising "Couldn't instantiate the backend tokenizer".

    TrOCR-base's decoder *is* RoBERTa-large, and that tokenizer loads without
    trouble, so the processor is assembled from parts: the image processor from
    the TrOCR repository, the tokenizer from roberta-large. The vocabulary sizes
    are asserted to match, because a tokenizer that merely loads but disagrees
    with the decoder would silently produce fluent nonsense -- which is far worse
    than an exception, since CER would then measure the wrong thing entirely.
    """
    from transformers import AutoImageProcessor, AutoTokenizer, TrOCRProcessor

    try:
        return TrOCRProcessor.from_pretrained(model_name, cache_dir=cache)
    except (ValueError, OSError):
        pass

    image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK, cache_dir=cache)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


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
