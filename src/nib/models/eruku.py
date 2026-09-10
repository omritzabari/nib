"""Eruku behind the project's Generator interface.

Eruku (Pippi et al., *Autoregressive Styled Text Image Generation, but Make it
Reliable*, WACV 2026) is Emuru's successor from the same group, and it exists
because of the failure this project measured independently: Emuru decides it has
finished by watching for ten consecutive latent slices that resemble a padding
token, and that heuristic misfires in both directions. Our first full evaluation
lost 8.7% of its output to a stop that never came and two requests to a stop
that came inside the style image.

Eruku replaces the heuristic with a **learned end-of-generation token**. Its loop
breaks on ``predicted_special == 1`` rather than on a similarity threshold.

Three other differences matter here.

**The style transcription is optional.** ``style_text`` defaults to ``""``.
Emuru required it, which forced a recogniser into the product: a user
photographing a page has transcribed nothing, so TrOCR had to read it first and
the whole system inherited TrOCR's 11.45% error rate. Eruku removes that link
from the chain. Passing the transcription still helps where one is available,
so this adapter passes it when the request carries one.

**It takes a PIL image and does its own preprocessing** -- ``convert('RGB')``
then a LANCZOS resize to height 64. Emuru wanted a tensor in [-1, 1] that we
scaled by hand. Less for us to get wrong, and one fewer place where a convention
mismatch produces a faint image rather than an error.

**cfg_scale.** Classifier-free guidance, default 1.25: a dial between following
the style closely and following the text closely. Emuru had no such knob. It is
left at the model's default here, and it is the obvious first thing to sweep
once the two models have been compared on equal terms.

.. warning::
   The comparison with Emuru is only meaningful if this returns the *generated*
   part alone. Emuru slices the style prefix off itself; whether Eruku does is
   not something to assume, because an output that silently included the prefix
   would contain a real crop of the writer's hand and send writer retrieval up
   for a reason that has nothing to do with the model. :meth:`_check_no_prefix`
   tests it on the first request and refuses to continue if it holds.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from nib.models.emuru import (
    EMPTY_RETRIES,
    NATIVE_HEIGHT,
    PIXELS_PER_TOKEN,
    TOKENS_PER_CHAR,
    EmptyOutputLog,
    Truncation,
    TruncationLog,
    token_budget,
)
from nib.models.generator import EmptyGeneration, GenerationRequest, GeneratorError
from nib.models.style import join_style

MODEL_ID = "blowing-up-groundhogs/eruku"

DEFAULT_CFG_SCALE = 1.25
"""The model's own default for classifier-free guidance. Kept rather than tuned,
so the first comparison against Emuru changes one thing and not two."""

PREFIX_MATCH_TOLERANCE = 8.0
"""Mean absolute pixel difference below which two crops are 'the same image'.

Generously loose: the style image goes through a LANCZOS resize and a VAE round
trip, so an included prefix would come back close but not identical. A false
alarm here costs a confusing error message; a miss costs every number in the
run."""


class ErukuGenerator:
    """The released Eruku checkpoint, adapted to this project's interface."""

    def __init__(
        self,
        device: str = "cpu",
        output_height: int = NATIVE_HEIGHT,
        model_id: str = MODEL_ID,
        max_new_tokens: int | None = None,
        tokens_per_char: float = TOKENS_PER_CHAR,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        empty_retries: int = EMPTY_RETRIES,
        use_style_text: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise GeneratorError("Eruku needs torch and transformers") from exc

        self.torch = torch
        self.device = torch.device(device)
        self._output_height = output_height
        self.max_new_tokens = max_new_tokens
        self.tokens_per_char = tokens_per_char
        self.cfg_scale = cfg_scale
        self.empty_retries = empty_retries

        self.use_style_text = use_style_text
        """Whether to pass the style sample's transcription when the request has
        one. Turning it off measures the deployable case directly: what the
        system scores when nobody has transcribed anything, which is the
        situation an actual user is in."""

        self.truncations = TruncationLog()
        self.empties = EmptyOutputLog()
        self._prefix_checked = False

        # trust_remote_code executes the model repository's own Python, as with
        # Emuru. Apache-2.0, same named academic group; recorded here rather than
        # buried, because it is the same decision each time it is made.
        self.model = (
            AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().to(self.device)
        )

    @property
    def name(self) -> str:
        return "eruku" if self.use_style_text else "eruku-no-style-text"

    @property
    def output_height(self) -> int:
        return self._output_height

    def budget_for(self, text: str) -> int:
        if self.max_new_tokens is not None:
            return int(self.max_new_tokens)
        return token_budget(text, tokens_per_char=self.tokens_per_char)

    def generate(self, requests: Sequence[GenerationRequest]) -> list[np.ndarray]:
        return [self._generate_one(request) for request in requests]

    def _generate_one(self, request: GenerationRequest) -> np.ndarray:
        from PIL import Image

        style, joined_text = join_style(request.style_images, request.style_texts)
        style_pil = Image.fromarray(np.asarray(style, dtype=np.uint8))
        style_text = joined_text if (self.use_style_text and joined_text) else ""

        budget = self.budget_for(request.text)

        for attempt in range(1 + self.empty_retries):
            image = self.model.generate_handwriting(
                style_image=style_pil,
                gen_text=request.text,
                style_text=style_text,
                cfg_scale=self.cfg_scale,
                max_new_tokens=budget,
            )
            array = np.asarray(image.convert("L"), dtype=np.uint8)
            if array.size and array.shape[1]:
                self._check_no_prefix(array, style)
                if attempt:
                    self.empties.retried += 1
                    self.empties.extra_draws += attempt
                scaled = self._to_output_height(array)
                self._record(request.text, scaled, budget)
                return scaled

        self.empties.failed.append(request.text)
        raise EmptyGeneration(
            f"Eruku wrote nothing for {request.text!r} on {1 + self.empty_retries} "
            "attempts. Its end-of-generation token fired immediately."
        )

    def _check_no_prefix(self, generated: np.ndarray, style: np.ndarray) -> None:
        """Refuse to run if the output begins with the style image.

        Checked once, on the first output, and then trusted -- the answer is a
        property of the model, not of the request.

        This is the failure that would not look like one. An output carrying its
        style prefix contains a real crop of the writer's hand, so writer
        retrieval would rise, FID would improve, and every number would move in
        the direction that suggests success.
        """
        if self._prefix_checked:
            return
        self._prefix_checked = True

        width = min(style.shape[1], generated.shape[1])
        if width < 16:
            return

        head = generated[:, :width].astype(np.float32)
        reference = style[:, :width].astype(np.float32)
        if head.shape != reference.shape:
            reference = cv2.resize(reference, (width, head.shape[0]))

        difference = float(np.abs(head - reference).mean())
        if difference < PREFIX_MATCH_TOLERANCE:
            raise GeneratorError(
                f"Eruku's output starts with its own style image (mean pixel "
                f"difference {difference:.1f} over the first {width}px). Every "
                "metric would then be scoring a real crop of the writer's hand "
                "as though the model had produced it, and all three would move "
                "the way success moves. Slice the prefix off before returning."
            )

    def _record(self, text: str, image: np.ndarray, budget: int) -> None:
        """Count anything that ran to its budget instead of emitting EOG.

        Same width test as Emuru's, and kept deliberately identical so the two
        models' truncation rates mean the same thing. For Eruku a truncation is
        a stronger signal: it means a *learned* stop token never fired, rather
        than a similarity threshold going unmet.
        """
        self.truncations.generated += 1
        ceiling = (budget - 2) * PIXELS_PER_TOKEN
        native_width = round(image.shape[1] * NATIVE_HEIGHT / self._output_height)
        if native_width >= ceiling:
            self.truncations.events.append(
                Truncation(text=text, width=int(image.shape[1]), budget=budget)
            )

    def _to_output_height(self, array: np.ndarray) -> np.ndarray:
        if array.shape[0] == self._output_height:
            return array
        scale = self._output_height / array.shape[0]
        return cv2.resize(
            array,
            (max(1, round(array.shape[1] * scale)), self._output_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
