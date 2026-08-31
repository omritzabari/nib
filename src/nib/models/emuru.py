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

**Its output length is bounded by a token budget, and one token is 8 pixels.**
That conversion comes from its own ``_generate``, which does ``lengths =
(lengths / 8).ceil()``. This wrapper originally passed a flat 96 tokens, chosen
when the unit was a word: 768 pixels, against an average real line of 886 and a
longest of 1885. It could not finish an average line. The budget is now derived
from the target's length -- see :func:`token_budget` -- and anything that still
runs to its cap is counted in :class:`TruncationLog` rather than passing quietly
into a metric.

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

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from nib.models.generator import GenerationRequest, GeneratorError

MODEL_ID = "blowing-up-groundhogs/emuru"
NATIVE_HEIGHT = 64
"""The height Emuru was trained at. Feeding it anything else is off-distribution."""

PIXELS_PER_TOKEN = 8
"""The VAE's width compression. From the model's own ``_generate``, which does
``lengths = (lengths / 8).ceil()``: one generated token is eight pixels of width.
This is the conversion the whole token budget rests on."""

TOKENS_PER_CHAR = 4.0
"""How many tokens a character of target text is allowed.

Measured on 300 real CVL lines at 64px: 23.2 pixels per character on average,
37.8 at the 95th percentile -- that is 2.9 and 4.7 tokens. Four is chosen to sit
above the ordinary case with room to spare while still costing less than a flat
maximum, because every unused token is an autoregressive step nobody needed."""

MIN_TOKENS = 32
"""256px, so a very short target still has room to finish."""

MAX_TOKENS = 256
"""The model's own default, and 2048px -- wider than the widest real CVL line at
this height, which is 1885px. A cap this high is a guard against a runaway, not a
length limit."""


def token_budget(
    text: str,
    tokens_per_char: float = TOKENS_PER_CHAR,
    minimum: int = MIN_TOKENS,
    maximum: int = MAX_TOKENS,
) -> int:
    """How many tokens this text is allowed to take.

    A pure function, and module-level rather than a method, so it can be tested
    without downloading a 3 GB checkpoint.

    The fixed budget it replaces was 96 tokens, chosen when the unit was a word
    and never revisited. 96 tokens is 768 pixels; a real CVL line at 64px
    averages 886 and reaches 1885. The model could not finish an average line,
    and the two 756px outputs recorded as "runaway generation" were 94-95 tokens
    hitting that cap. Scaling with the text length fixes both directions: long
    lines get room, short ones stop costing steps they never use.
    """
    wanted = math.ceil(len(text) * tokens_per_char)
    return max(minimum, min(maximum, wanted))


@dataclass(frozen=True)
class Truncation:
    """One request whose output ran to its budget instead of stopping.

    Kept rather than raised. A truncated image is a real, if incomplete, result,
    and dropping it would misalign every later pairing of image to target text --
    which the Generator interface forbids for exactly that reason. It is counted
    and reported instead, because an evaluation over silently truncated output is
    measuring something other than what it claims.
    """

    text: str
    width: int
    budget: int


@dataclass
class TruncationLog:
    """What ran to the cap, over the life of one generator."""

    generated: int = 0
    events: list[Truncation] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return len(self.events) / self.generated if self.generated else 0.0

    def summary(self) -> str:
        if not self.generated:
            return "nothing generated yet"
        if not self.events:
            return f"truncated       0 of {self.generated} -- every request stopped on its own"
        widest = max(self.events, key=lambda event: len(event.text))
        return "\n".join(
            [
                f"truncated       {len(self.events)} of {self.generated}  ({self.rate:.1%})",
                "  these hit their token budget instead of stopping, so their text is",
                "  cut short and their CER is charged for it. Raise TOKENS_PER_CHAR if",
                "  the rate is high; a few are the known Emuru non-stopping failure.",
                f"  longest affected: {len(widest.text)} chars, "
                f"{widest.width}px at a budget of {widest.budget} tokens",
            ]
        )


class EmuruGenerator:
    """The released Emuru checkpoint, adapted to this project's interface."""

    def __init__(
        self,
        device: str = "cpu",
        output_height: int = NATIVE_HEIGHT,
        model_id: str = MODEL_ID,
        max_new_tokens: int | None = None,
        tokens_per_char: float = TOKENS_PER_CHAR,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise GeneratorError("Emuru needs torch and transformers") from exc

        self.torch = torch
        self.device = torch.device(device)
        self._output_height = output_height

        # None means a budget per request, from the target's length. An explicit
        # integer pins every request to the same cap, which is what the earlier
        # runs did and is kept only so one can be reproduced.
        self.max_new_tokens = max_new_tokens
        self.tokens_per_char = tokens_per_char
        self.truncations = TruncationLog()

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

    def budget_for(self, text: str) -> int:
        """The token budget this request gets. A pinned cap overrides the rule."""
        if self.max_new_tokens is not None:
            return int(self.max_new_tokens)
        return token_budget(text, tokens_per_char=self.tokens_per_char)

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
            budget = self.budget_for(request.text)
            image = self.model.generate(
                style_text=request.style_texts[0],
                gen_text=request.text,
                style_img=self._as_tensor(request.style_images[0]),
                max_new_tokens=budget,
            )
            array = self._from_pil(image)
            self._record(request.text, array, budget)
            out.append(array)
        return out

    def _record(self, text: str, image: np.ndarray, budget: int) -> None:
        """Note whether this output stopped on its own or ran out of room.

        The model returns no flag for it, so the width is the tell. When its
        stopping criterion never fires, ``seq_stops`` stays -1 and its own
        ``generate`` slices to ``-8``, which hands back the whole canvas bar the
        last eight pixels -- that is ``budget * 8 - 8`` of new content. Anything
        within a token of that ceiling was cut off rather than finished.
        """
        self.truncations.generated += 1
        ceiling = (budget - 2) * PIXELS_PER_TOKEN
        native_width = round(image.shape[1] * NATIVE_HEIGHT / self._output_height)
        if native_width >= ceiling:
            self.truncations.events.append(
                Truncation(text=text, width=int(image.shape[1]), budget=budget)
            )

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
