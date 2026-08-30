"""Emuru behind the project's Generator interface.

Emuru (Pippi et al., *Zero-Shot Styled Text Image Generation, but Make It
Autoregressive*, CVPR 2025) is a T5 encoder-decoder over a small image VAE. It is
handed a style image and continues writing after it, one vertical slice at a
time, then returns only the part it added.

Three things about it are worth knowing before reading any number it produces.

**It needs the style sample's transcription.** ``generate(style_text, gen_text,
style_img)`` puts ``style_text + " " + gen_text`` through T5, so it has to be told
what the reference *says*. That is a real deployment cost: a user who photographs
a page has transcribed nothing, so the product would have to read the sample
first -- which is what TrOCR is already here for, at the accuracy TrOCR happens to
have. Recorded rather than hidden, because it constrains the architecture.

**It takes one style image, not several.** The interface accepts a list because
other models use more; here only the first is used, and the rest are ignored.

**It was trained on synthetic fonts and never on CVL.** Every writer we evaluate
on is unseen by construction, which is exactly the claim being tested and is the
reason this model was chosen.

.. warning::
   **Do not feed it single words normalised to a fixed height.** The first real
   generation exposed a flaw in this project's own pipeline, not in the model:
   ``normalise_word`` stretches every crop to exactly 64px, so a one-letter word
   becomes as tall as a whole line and a long word shrinks. Relative scale --
   which is part of how a hand looks -- is destroyed. Emuru was trained on lines,
   where scale is consistent, and it generates lines natively. Style references
   and targets should be lines.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from nib.models.generator import GenerationRequest, GeneratorError

MODEL_ID = "blowing-up-groundhogs/emuru"
NATIVE_HEIGHT = 64
"""The height Emuru was trained at. Feeding it anything else is off-distribution."""


class EmuruGenerator:
    """The released Emuru checkpoint, adapted to this project's interface."""

    def __init__(
        self,
        device: str = "cpu",
        output_height: int = NATIVE_HEIGHT,
        model_id: str = MODEL_ID,
        max_new_tokens: int = 96,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise GeneratorError("Emuru needs torch and transformers") from exc

        self.torch = torch
        self.device = torch.device(device)
        self._output_height = output_height
        self.max_new_tokens = max_new_tokens

        if output_height != NATIVE_HEIGHT:
            print(
                f"[emuru] warning: asked for {output_height}px output but the model "
                f"was trained at {NATIVE_HEIGHT}px. Generating native and resizing."
            )

        # trust_remote_code executes the model repository's own Python. That is
        # how this model is distributed and it is MIT-licensed from a named
        # academic group, but it is third-party code and the fact is recorded
        # here rather than buried.
        self.model = (
            AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().to(self.device)
        )

    @property
    def name(self) -> str:
        return "emuru"

    @property
    def output_height(self) -> int:
        return self._output_height

    def generate(self, requests: Sequence[GenerationRequest]) -> list[np.ndarray]:
        """One image per request, in the same order. Failures raise rather than
        being skipped -- a dropped request would misalign every later pairing."""
        out: list[np.ndarray] = []
        for request in requests:
            if not request.style_texts:
                raise GeneratorError(
                    "Emuru needs the style sample's transcription. Pass style_texts, "
                    "or read the sample with a recogniser first -- the model puts "
                    "style_text and gen_text through T5 together."
                )
            image = self.model.generate(
                style_text=request.style_texts[0],
                gen_text=request.text,
                style_img=self._as_tensor(request.style_images[0]),
                max_new_tokens=self.max_new_tokens,
            )
            out.append(self._from_pil(image))
        return out

    def _as_tensor(self, image: np.ndarray):
        """Our uint8 ink-on-white to the model's expected tensor.

        Three channels at the native height, scaled to **[-1, 1]** with ink dark.

        Measured, not assumed. The same style image was fed four ways and the
        output inspected: [0,1] ink-dark produced a barely visible mark (darkest
        pixel 183), inverted [0,1] produced nothing at all (233), and inverted
        [-1,1] produced a blob. [-1,1] with ink dark produced the only legible
        letter. That matches the model's own output convention, which is
        ``(x + 1) / 2`` -- its VAE works in [-1, 1] and expects the same going in.
        """
        array = np.asarray(image)
        if array.ndim != 2:
            raise GeneratorError(f"style image must be grayscale, got {array.shape}")

        if array.shape[0] != NATIVE_HEIGHT:
            scale = NATIVE_HEIGHT / array.shape[0]
            array = cv2.resize(
                array,
                (max(1, round(array.shape[1] * scale)), NATIVE_HEIGHT),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )

        tensor = self.torch.from_numpy(array.astype(np.float32) / 127.5 - 1.0)
        return tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(self.device)

    def _from_pil(self, image) -> np.ndarray:
        """Back to uint8 grayscale at the requested height."""
        array = np.asarray(image.convert("L"), dtype=np.uint8)
        if array.size == 0 or array.shape[1] == 0:
            raise GeneratorError("the model returned an empty image")
        if array.shape[0] != self._output_height:
            scale = self._output_height / array.shape[0]
            array = cv2.resize(
                array,
                (max(1, round(array.shape[1] * scale)), self._output_height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
        return array
