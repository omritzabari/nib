"""Several draws per line, the broken ones rejected, and optionally the best hand kept.

Emuru's output is bimodal. Over 296 held-out lines with one style line, the
median line read at 9.3% CER -- close to real handwriting -- while 10% came out
above 90%: two or three words and then a smear of repeated strokes to the end of
the budget, or nothing. Those are not bad handwriting, they are no handwriting,
and the model's sampling makes a second draw a genuine second chance.

This module draws again. For each line to write:

1. order the style lines on offer, best-suited first -- lines between 500 and
   1100px gave 26-28% mean CER and 9-10% failures, lines under 500px 69% and 22%;
2. draw with **one** style line at a time -- two joined side by side broke the
   text itself (CER 30.4% -> 59.6%), so a page is used by selection, never by
   concatenation;
3. read the draw with a recogniser and score it against the intended text --
   its CER, and how much it wrote beyond that text;
4. keep a draw, by one of two rules.

**Readable (T28).** Stop at the first draw readable enough, otherwise keep the
best-read of the lot. Most lines pass first time, so it cost 1.35 draws a line.
Measured: CER gap to real 19.1 -> 1.8 points, FID 67.70 -> 55.87, and identity
+8.3 points [2.4, 14.8] paired by writer.

**Closest to the hand.** Draw every candidate, set aside the unreadable ones, and
keep the one whose writer embedding is nearest the mean embedding of the style
lines -- the writer's page. The re-draws that fixed legibility also moved
identity, which says draws differ in how much of the hand they carry; this picks
for that directly, at the cost of every draw being made.

**The selector must not be the judge**, in either rule. Choosing by one model's
opinion and reporting a metric computed by the same model rewards that model's
particular mistakes. Readability is chosen by TrOCR-small and measured by
TrOCR-base. Closeness to the hand is chosen by this project's writer embedding
and measured by HWD, a different network trained on different data -- which
also means the writer-retrieval figure, computed with that embedding, is no
longer independent in this mode and should not be read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nib.engine.metrics.cer import Recogniser, cer
from nib.engine.metrics.writer import Embedder
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
"""A draw reading at or below this counts as readable.

Set to reject the broken tail, not to polish good lines: on cell 6 the median
line read at 9.3% and the failures above 90%. The selector is TrOCR-small, whose
own rate on real lines has not been measured here, so the threshold is loose on
purpose -- a tight one would spend draws re-rolling lines that were fine."""

ACCEPT_OVERRUN = 2
"""A readable draw may carry at most this many characters beyond either end of
its text -- see :func:`overrun`.

A draw can write the whole line and then keep going ("about the t.t. 1/",
"cafe: Te-Te" on Amri's page) and still read under ``ACCEPT_CER``, because a few
extra characters are small against a line. Calibrated with TrOCR-small, the
selector, on 2026-09-16. Readable real lines it would reject: 2 of 290 CVL lines
at 3 or more (0.7%, both misread ends, "Framework" for "Zemanek"), against 8 at 2
or more (2.8%); 0 of Amri's 22. Readable generated lines it rejects at 3 or more:
2 of his 27 -- exactly the two with junk after the text -- and 13 of 293 kept by
cell 7c, most of them a repeated last word ("they they", "in ins", "on on on")."""

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


def overrun(reading: str, target: str) -> int:
    """The most characters a reading carries beyond either end of the target.

    The whole target is aligned to the reading by edit distance, and the alignment
    may begin and end anywhere in it; what lies outside it, before or after, is
    writing the line was not asked for. Spaces are removed from both first, so the
    space a reader puts before punctuation ("dream .") costs nothing. Among
    alignments of least cost the widest wins, so a misread end letter is charged
    to CER rather than here.
    """
    read = "".join(reading.split())
    text = "".join(target.split())
    # cost[j], start[j]: the cheapest alignment of the target so far ending at
    # reading position j, and where in the reading it began.
    cost = [0] * (len(read) + 1)
    start = list(range(len(read) + 1))
    for i, char in enumerate(text, start=1):
        previous_cost, previous_start = cost, start
        cost, start = [i] + [0] * len(read), [0] * (len(read) + 1)
        for j, seen in enumerate(read, start=1):
            cost[j], start[j] = min(
                (previous_cost[j - 1] + (char != seen), previous_start[j - 1]),
                (previous_cost[j] + 1, previous_start[j]),
                (cost[j - 1] + 1, start[j - 1]),
            )
    _, _, before, end = min(
        (cost[j], max(start[j], len(read) - j), start[j], j) for j in range(len(read) + 1)
    )
    return max(before, len(read) - end)


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@dataclass(frozen=True)
class Draw:
    image: np.ndarray
    score: float
    """The selector's CER for this draw."""

    style_index: int
    truncated: bool
    overrun: int = 0
    """Characters read beyond either end of the text."""

    similarity: float | None = None
    """Cosine similarity to the writer's page, when selecting by hand."""


@dataclass
class SelectionLog:
    """What selection cost and what it rejected."""

    mode: str = "readable"
    requests: int = 0
    draws: int = 0
    first_draw_accepted: int = 0
    none_accepted: int = 0
    """Requests where no draw was readable and the best-read was kept anyway."""

    moved_by_hand: int = 0
    """Requests where closeness to the hand chose a different draw than the first
    readable one would have been -- how often the second rule changed anything."""

    rejected_for_overrun: int = 0
    """Draws that read well enough but wrote beyond the text, set aside for it."""

    chosen_scores: list[float] = field(default_factory=list)
    """The selector's CER for each kept image, in request order."""

    @property
    def draws_per_request(self) -> float:
        return self.draws / self.requests if self.requests else 0.0

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "requests": self.requests,
            "draws": self.draws,
            "draws_per_request": self.draws_per_request,
            "first_draw_accepted": self.first_draw_accepted,
            "none_accepted": self.none_accepted,
            "moved_by_hand": self.moved_by_hand,
            "rejected_for_overrun": self.rejected_for_overrun,
        }

    def summary(self) -> str:
        if not self.requests:
            return "selection       nothing generated yet"
        lines = [
            f"selection       {self.draws} draws for {self.requests} requests "
            f"({self.draws_per_request:.2f} per request), kept by {self.mode}",
        ]
        if self.mode == "hand":
            lines.append(
                f"  closeness to the hand picked a different draw than the first readable "
                f"one for {self.moved_by_hand} requests"
            )
        else:
            lines.append(f"  {self.first_draw_accepted} accepted on the first draw")
        lines += [
            f"  {self.rejected_for_overrun} read well but wrote beyond the text, set aside",
            f"  {self.none_accepted} kept as the best of an unreadable set",
            "  scored by selectors, not by the models that measure below",
        ]
        return "\n".join(lines)


class CandidateGenerator:
    """Wraps any generator: one style line per draw, several draws, one kept.

    Satisfies the ``Generator`` interface, so the evaluation and the product use
    it exactly as they would the model underneath. Pass ``hand`` to keep the
    readable draw closest to the style lines instead of the first readable one.
    """

    def __init__(
        self,
        base: Generator,
        selector: Recogniser,
        candidates: int = DEFAULT_CANDIDATES,
        accept_cer: float = ACCEPT_CER,
        width_range: tuple[int, int] = STYLE_WIDTH_RANGE,
        hand: Embedder | None = None,
        accept_overrun: int = ACCEPT_OVERRUN,
    ) -> None:
        if candidates < 1:
            raise GeneratorError(f"need at least one candidate, got {candidates}")
        self.base = base
        self.selector = selector
        self.candidates = candidates
        self.accept_cer = accept_cer
        self.accept_overrun = accept_overrun
        self.width_range = width_range
        self.hand = hand
        self.selection = SelectionLog(mode="readable" if hand is None else "hand")
        # Over the images this wrapper returns, not over every draw: a rejected
        # draw that ran to its budget is not in the output and must not be
        # counted as if it were.
        self.truncations = TruncationLog()
        self.empties = EmptyOutputLog()
        self._pages: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        rule = "" if self.hand is None else "-by-hand"
        return f"{self.base.name}+best-of-{self.candidates}{rule}"

    @property
    def output_height(self) -> int:
        return self.base.output_height

    def generate(self, requests: Sequence[GenerationRequest]) -> list[np.ndarray]:
        return [self._one(request) for request in requests]

    def _one(self, request: GenerationRequest) -> np.ndarray:
        order = style_order(request.style_images, self.width_range)
        draws: list[Draw] = []
        empty_draws = 0
        used = 0

        for attempt in range(self.candidates):
            used += 1
            draw = self._draw(request, order[attempt % len(order)])
            if draw is None:
                empty_draws += 1
                continue
            draws.append(draw)
            if draw.score <= self.accept_cer and draw.overrun > self.accept_overrun:
                self.selection.rejected_for_overrun += 1
            if self.hand is None and self._readable(draw):
                break

        self.selection.requests += 1
        self.selection.draws += used
        self.empties.extra_draws += empty_draws

        if not draws:
            self.empties.failed.append(request.text)
            raise EmptyGeneration(
                f"no draw of {self.candidates} produced an image for {request.text!r}"
            )

        best = self._choose(request, draws)
        if empty_draws:
            self.empties.retried += 1
        if used == 1:
            self.selection.first_draw_accepted += 1
        if not self._readable(best):
            self.selection.none_accepted += 1
        self.selection.chosen_scores.append(best.score)

        self.truncations.generated += 1
        if best.truncated:
            self.truncations.events.append(self._last_truncation(request, best))
        return best.image

    def _readable(self, draw: Draw) -> bool:
        return draw.score <= self.accept_cer and draw.overrun <= self.accept_overrun

    def _choose(self, request: GenerationRequest, draws: list[Draw]) -> Draw:
        readable = [draw for draw in draws if self._readable(draw)]
        if not readable:
            return min(draws, key=lambda draw: draw.score)
        if self.hand is None:
            return readable[0]

        page = self._page(request)
        vectors = _unit(self.hand([draw.image for draw in readable]))
        similarities = vectors @ page
        scored = [
            Draw(d.image, d.score, d.style_index, d.truncated, d.overrun, float(s))
            for d, s in zip(readable, similarities, strict=True)
        ]
        best = max(scored, key=lambda draw: draw.similarity)
        if best.image is not readable[0].image:
            self.selection.moved_by_hand += 1
        return best

    def _page(self, request: GenerationRequest) -> np.ndarray:
        """The writer's page as one unit vector: the mean of its lines' embeddings.

        Cached by content, because every request for a writer carries the same
        lines and embedding them again for each would be wasted work."""
        digest = hashlib.sha1()
        for image in request.style_images:
            array = np.ascontiguousarray(image)
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        key = digest.hexdigest()
        if key not in self._pages:
            embeddings = _unit(self.hand(list(request.style_images)))
            self._pages[key] = _unit(embeddings.mean(axis=0))
        return self._pages[key]

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
        return Draw(
            image,
            cer(reading, request.text),
            index,
            truncated,
            overrun(reading, request.text),
        )

    def _last_truncation(self, request: GenerationRequest, draw: Draw):
        from nib.models.emuru import Truncation

        base_log = getattr(self.base, "truncations", None)
        matching = [e for e in (base_log.events if base_log else []) if e.text == request.text]
        if matching:
            return matching[-1]
        return Truncation(text=request.text, width=int(draw.image.shape[1]), budget=0)
