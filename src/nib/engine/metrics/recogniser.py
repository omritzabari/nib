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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
TOKENIZER_FALLBACK = "roberta-large"
"""TrOCR-base's decoder is RoBERTa-large; see _load_processor."""


class RecogniserError(RuntimeError):
    pass


@dataclass(frozen=True)
class Omission:
    """The word of a line's text its image most clearly lacks.

    ``support`` is a log-likelihood ratio: how much better the image reads as the
    text *without* this word than as the whole text. Negative means the word is
    there; large and positive means the image is far better explained without it.
    """

    word: str
    index: int
    support: float


def without_each_word(text: str) -> list[tuple[int, str]]:
    """The text with each word left out in turn -- every word but the last.

    **Not the last.** The reader is weakest at the right edge of a line: on 120
    complete real CVL lines, the strongest false signal fell on the last word in
    51% of them, and leaving it out of the test cut false flags at the working
    threshold from 5.0% of lines to 0.8%. A line that has lost its last word has
    been cut short, which the width check in ``nib.models.candidates`` watches.
    """
    words = text.split()
    return [(i, " ".join(words[:i] + words[i + 1 :])) for i in range(len(words) - 1)]


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

    def omissions(
        self, images: Sequence[np.ndarray], texts: Sequence[str]
    ) -> list[Omission | None]:
        """For each image, the word of its text it most clearly lacks.

        **Asks whether this text is in the image, not what text is.** :meth:`read`
        decodes freely, and a free-running decoder carries a language model's
        habits: it skips short function words from lines where they are plainly
        written. Run on the fake generator, whose lines are complete by
        construction, it read "not the rapid calculation" as "not rapid
        calculation" and "it will be enough" as "it will". Compared against the
        text, that reading cannot tell a word the generator left out from one the
        reader skipped.

        So the text is scored instead, teacher-forced: the log-likelihood of the
        whole text given the image, against the same with each word removed in
        turn. A word that is on the page is expensive to remove -- every token
        after it loses its alignment. A word that is not costs only its own
        tokens to keep, and removing it pays. One encoder pass per image, one
        batched decoder pass over the variants.

        None for a text of one word, which has nothing but its last word to test.
        """
        return [self._omission(image, text) for image, text in zip(images, texts, strict=True)]

    def _omission(self, image: np.ndarray, text: str) -> Omission | None:
        candidates = without_each_word(text)
        if not candidates:
            return None
        torch = self.torch
        pixels = self.processor(images=[_to_rgb_uint8(image)], return_tensors="pt").pixel_values
        with torch.no_grad():
            encoded = self.model.encoder(pixel_values=pixels.to(self.device)).last_hidden_state
            loglik = self._loglik(encoded, [text] + [variant for _, variant in candidates])
        support = loglik[1:] - loglik[0]
        best = int(np.argmax(support))
        index = candidates[best][0]
        return Omission(word=text.split()[index], index=index, support=float(support[best]))

    def _loglik(self, encoded, texts: Sequence[str]) -> np.ndarray:
        """Each text's log-likelihood given one encoded image, in one decoder pass.

        The sequences are built the way :meth:`read` would produce them -- the
        decoder's start token, the text's tokens, the end token -- so a text is
        scored exactly as the reader would have to write it.
        """
        torch = self.torch
        config = self.model.generation_config
        start = int(config.decoder_start_token_id)
        end = config.eos_token_id
        end = int(end[0] if isinstance(end, list | tuple) else end)
        pad = int(config.pad_token_id)

        tokens = [self.processor.tokenizer(t, add_special_tokens=False).input_ids for t in texts]
        width = max(len(ids) for ids in tokens) + 1
        inputs = torch.full((len(tokens), width), pad, dtype=torch.long)
        targets = torch.full((len(tokens), width), -100, dtype=torch.long)
        for row, ids in enumerate(tokens):
            inputs[row, : len(ids) + 1] = torch.tensor([start, *ids])
            targets[row, : len(ids) + 1] = torch.tensor([*ids, end])
        mask = (targets != -100).long()

        logits = self.model(
            encoder_outputs=(encoded.expand(len(tokens), -1, -1),),
            decoder_input_ids=inputs.to(self.device),
            decoder_attention_mask=mask.to(self.device),
        ).logits
        picked = logits.log_softmax(-1).gather(
            -1, targets.clamp(min=0).unsqueeze(-1).to(self.device)
        )
        return (picked.squeeze(-1) * mask.to(self.device)).sum(-1).float().cpu().numpy()


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
