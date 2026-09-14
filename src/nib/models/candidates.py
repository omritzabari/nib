"""Several draws per line, and the broken ones rejected before anyone sees them.

Emuru's output is bimodal. Over 296 held-out lines with one style line, the
median line read at 9.3% CER -- close to real handwriting -- while 10% came out
above 90%: two or three words and then a smear of repeated strokes to the end of
the budget, or nothing. Those are not bad handwriting, they are no handwriting,
and the model's sampling makes a second draw a genuine second chance. Removing
the unreadable lines, from generated and real alike, lifted HWD identity from
56.5% to 67.3% as well as CER from 30.4% to 9.5% -- a bound on what catching them
could buy, not a measurement of doing it.

This module does it. For each line to write:

1. order the style lines on offer, best-suited first -- lines between 500 and
   1100px gave 26-28% mean CER and 9-10% failures, lines under 500px 69% and 22%;
2. draw with **one** style line at a time -- two joined side by side broke the
   text itself (CER 30.4% -> 59.6%), so a page is used by selection, never by
   concatenation;
3. read the draw with a recogniser and score it against the intended text;
4. stop at the first draw readable enough, otherwise keep the best of the lot.

Most lines pass on the first draw, so the extra cost falls on the lines that
needed it.

**The selector must not be the judge.** Choosing by one recogniser's reading and
then reporting CER from the same recogniser rewards its particular mistakes, and
the figure flatters the method. The evaluation selects with TrOCR-small and
measures with TrOCR-base; HWD and FID never see a recogniser and are the clean
measurements of what selection did.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nib.engine.metrics.cer import Recogniser, cer
from nib.models.emuru import EmptyOutputLog, TruncationLog
from nib.models.generator import (
    EmptyGeneration,
    GenerationRequest,
    Generator,
    GeneratorError,
    to_uint8,
)

DEFAULT_CANDIDATES = 4

ACCEPT_CER = 0.5
"""A draw reading at or below this is kept without drawing again.

Set to reject the broken tail, not to polish good lines: on cell 6 the median
line read at 9.3% and the failures above 90%. The selector is TrOCR-small, whose
own rate on real lines has not been measured here, so the threshold is loose on
purpose -- a tight one would spend draws re-rolling lines that were fine."""

STYLE_WIDTH_RANGE = (500, 1100)
"""Style line widths, at the reference height, that generated well. See above."""

REFERENCE_HEIGHT = 64
"""The height the widths above were measured at."""


def style_order(
    images: Sequence[np.ndarray],
    width_range: tuple[int, int] = STYLE_WIDTH_RANGE,
    reference_height: int = REFERENCE_HEIGHT,
) -> list[int]:
    """Indices of the style lines, best-suited first.

    Lines inside the range come first, nearest its middle first; lines outside it
    follow, nearest the range first. None is discarded: a writer whose every line
    is short still gets drawn from.
    """
    low, high = width_range
    middle = (low + high) / 2

    def key(index: int) -> tuple[int, float]:
        image = np.asarray(images[index])
        width = image.shape[1] * reference_height / max(1, image.shape[0])
        if low <= width <= high:
            return (0, abs(width - middle))
        return (1, min(abs(width - low), abs(width - high)))

    return sorted(range(len(images)), key=key)


@dataclass(frozen=True)
class Draw:
    image: np.ndarray
    score: float
    style_index: int
    truncated: bool


@dataclass
class SelectionLog:
    """What selection cost and what it rejected."""

    requests: int = 0
    draws: int = 0
    first_draw_accepted: int = 0
    none_accepted: int = 0
    """Requests where no draw met the threshold and the best was kept anyway."""

    chosen_scores: list[float] = field(default_factory=list)
    """The selector's CER for each kept image, in request order."""

    @property
    def draws_per_request(self) -> float:
        return self.draws / self.requests if self.requests else 0.0

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "draws": self.draws,
            "draws_per_request": self.draws_per_request,
            "first_draw_accepted": self.first_draw_accepted,
            "none_accepted": self.none_accepted,
        }

    def summary(self) -> str:
        if not self.requests:
            return "selection       nothing generated yet"
        return "\n".join(
            [
                f"selection       {self.draws} draws for {self.requests} requests "
                f"({self.draws_per_request:.2f} per request)",
                f"  {self.first_draw_accepted} accepted on the first draw, "
                f"{self.none_accepted} kept as the best of an unreadable set",
                "  scored by the selector, not by the recogniser that measures CER below",
            ]
        )


class CandidateGenerator:
    """Wraps any generator: one style line per draw, several draws, best kept.

    Satisfies the ``Generator`` interface, so the evaluation and the product use
    it exactly as they would the model underneath.
    """

    def __init__(
        self,
        base: Generator,
        selector: Recogniser,
        candidates: int = DEFAULT_CANDIDATES,
        accept_cer: float = ACCEPT_CER,
        width_range: tuple[int, int] = STYLE_WIDTH_RANGE,
    ) -> None:
        if candidates < 1:
            raise GeneratorError(f"need at least one candidate, got {candidates}")
        self.base = base
        self.selector = selector
        self.candidates = candidates
        self.accept_cer = accept_cer
        self.width_range = width_range
        self.selection = SelectionLog()
        # Over the images this wrapper returns, not over every draw: a rejected
        # draw that ran to its budget is not in the output and must not be
        # counted as if it were.
        self.truncations = TruncationLog()
        self.empties = EmptyOutputLog()

    @property
    def name(self) -> str:
        return f"{self.base.name}+best-of-{self.candidates}"

    @property
    def output_height(self) -> int:
        return self.base.output_height

    def generate(self, requests: Sequence[GenerationRequest]) -> list[np.ndarray]:
        return [self._one(request) for request in requests]

    def _one(self, request: GenerationRequest) -> np.ndarray:
        order = style_order(request.style_images, self.width_range)
        best: Draw | None = None
        empty_draws = 0
        used = 0

        for attempt in range(self.candidates):
            used += 1
            draw = self._draw(request, order[attempt % len(order)])
            if draw is None:
                empty_draws += 1
                continue
            if best is None or draw.score < best.score:
                best = draw
            if draw.score <= self.accept_cer:
                break

        self.selection.requests += 1
        self.selection.draws += used
        self.empties.extra_draws += empty_draws

        if best is None:
            self.empties.failed.append(request.text)
            raise EmptyGeneration(
                f"no draw of {self.candidates} produced an image for {request.text!r}"
            )

        if empty_draws:
            self.empties.retried += 1
        if used == 1:
            self.selection.first_draw_accepted += 1
        if best.score > self.accept_cer:
            self.selection.none_accepted += 1
        self.selection.chosen_scores.append(best.score)

        self.truncations.generated += 1
        if best.truncated:
            self.truncations.events.append(self._last_truncation(request, best))
        return best.image

    def _draw(self, request: GenerationRequest, index: int) -> Draw | None:
        single = GenerationRequest(
            text=request.text,
            style_images=[request.style_images[index]],
            style_texts=[request.style_texts[index]] if request.style_texts else None,
        )
        base_log = getattr(self.base, "truncations", None)
        before = len(base_log.events) if base_log is not None else 0
        try:
            image = to_uint8(self.base.generate([single])[0])
        except EmptyGeneration:
            return None
        truncated = base_log is not None and len(base_log.events) > before
        reading = self.selector.read([image])[0]
        return Draw(image, cer(reading, request.text), index, truncated)

    def _last_truncation(self, request: GenerationRequest, draw: Draw):
        from nib.models.emuru import Truncation

        base_log = getattr(self.base, "truncations", None)
        matching = [e for e in (base_log.events if base_log else []) if e.text == request.text]
        if matching:
            return matching[-1]
        return Truncation(text=request.text, width=int(draw.image.shape[1]), budget=0)
