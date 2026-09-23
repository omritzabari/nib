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
   its CER, and how much it wrote beyond that text; check its width against
   what this hand needs for the text; and, given a verifier, ask whether any
   word of the text is missing from it;
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
from itertools import pairwise
from typing import Protocol

import numpy as np

from nib.engine.metrics.cer import Recogniser, cer
from nib.engine.metrics.recogniser import Omission
from nib.engine.metrics.writer import Embedder
from nib.models.emuru import EmptyOutputLog, TruncationLog
from nib.models.generator import (
    EmptyGeneration,
    GenerationRequest,
    Generator,
    GeneratorError,
    to_uint8,
)


class Verifier(Protocol):
    """Anything that can say which word of a text an image most clearly lacks.

    Separate from :class:`Recogniser` because it answers a different question --
    whether this text is in the image, rather than what text is -- and a reader
    that answers the second well can answer the first badly.
    """

    def omissions(
        self, images: Sequence[np.ndarray], texts: Sequence[str]
    ) -> list[Omission | None]: ...


DEFAULT_CANDIDATES = 4

EXTRA_DRAWS = 4
"""Further draws when none of the first ones is acceptable, stopping at the first
that is. In 7s, 19 of 150 requests had no acceptable draw of four and kept the
least bad -- and one kept line in ten still left a word out. Costs time only on
the lines that failed."""

ACCEPT_CER = 0.5
"""A draw reading at or below this counts as readable.

Set to reject the broken tail, not to polish good lines: on cell 6 the median
line read at 9.3% and the failures above 90%. The selector is TrOCR-small, whose
own rate on real lines has not been measured here, so the threshold is loose on
purpose -- a tight one would spend draws re-rolling lines that were fine."""

OMISSION_SUPPORT = 10.0
"""How much better a draw must read *without* one of its words before that word
counts as left out -- see ``TrOcrRecogniser.omissions``. A log-likelihood ratio.

**Emuru leaves a word out of one line in five that this module used to keep.** On
cell 7e's 150 kept lines, 31 (20.7%) score above this, and every one of the ten
looked at by eye is missing text: "differ much more from each other" written as
"much more from each other", "about on or in the surface" as "about or in the
surface", "a few years ago I should" as "few years ago . should". In 13 of the 31
the missing word is the *first* -- the seam where the style line ends and the new
text begins. CER did not show it: one short word lost costs about what the
reader's own noise does, so 7e read at 12.0% against 10.8% for real lines.

**The first attempt read the draw and compared.** A free-running reader has a
language model's habits and skips short words from complete lines by itself --
on the fake generator, whose lines are whole by construction, it would have set
aside 16 of 24 correct draws. This asks the reader whether the text is in the
image instead, which it can answer.

Measured at this threshold: 1 of 120 complete real CVL lines flagged (0.8%);
of 69 typeface lines drawn with one short inner word left out, 62 caught (90%),
with the removed word the one named in 86%. On Amri's page, "warm at noon"
written without "at" scores +26.2 on "at", "Order #378 at" written without
"#378" scores +18.2 on "#378", and every complete line, his or generated, scores
-6.1 or below. Anywhere from 6 to 12 flags the same 7e lines within 2 points.
"""

WIDTH_BAND = (0.70, 1.40)
"""What share of its predicted width a readable draw must occupy.

A line far too narrow for its text has lost some of it; one far too wide has
repeated itself or run into a smear. Both are visible without a recogniser at
all, from what the style lines say this writer's hand costs per character --
which is the one thing enrolment always knows, since the user supplies the
transcription of their own page.

**The relation is U-shaped, which is why nothing had found it.** Across a run the
linear correlation between this ratio and CER is +0.03. Split into bands it is
stark. Outside [0.70, 1.40) the mean CER of a draw is 29.4%, 31.3% and 75.1% on
cells 7c, 7e and the zero-shot run, against 13.9%, 12.4% and 30.8% over all draws
there, while only 21.7%, 13.3% and 27.7% of draws fall outside. Tightening the
band rejects far more and rejects better draws: at [0.90, 1.25) it sets aside
47-61% of everything and what it sets aside averages 17-41%.

The centre is below 1 on purpose -- the generator writes about 15% narrower than
the hand it is copying -- so a band centred on 1.0 would reject sound draws for
being compact. These bounds are measured against that centre, not around it.
"""

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


ORDERINGS = ("width", "letters")
"""How the style lines on offer are ranked. See :func:`style_order`."""


def style_order(
    images: Sequence[np.ndarray],
    width_range: tuple[int, int] = STYLE_WIDTH_RANGE,
    reference_height: int = REFERENCE_HEIGHT,
    style_texts: Sequence[str] | None = None,
    target: str | None = None,
    by: str = "width",
) -> list[int]:
    """Indices of the style lines, best-suited first.

    **By width**, the default and what every run up to 2026-09-18 used: lines inside
    the range come first, nearest its middle first; lines outside it follow, nearest
    the range first. Width was measured to matter -- style lines under 500px produced
    69% mean CER against 26-28% inside the range.

    **By letters**: among lines of a workable width, the one showing most of the
    characters the target needs comes first. The model is given one line per draw,
    so a page of twenty lines reaches it one line at a time; a line that shows how
    this writer forms the letters about to be written is better evidence than a line
    that happens to be 800px wide. Falls back to width when the style lines carry no
    text, since the characters are then unknown.
    """
    if by not in ORDERINGS:
        raise GeneratorError(f"style ordering must be one of {ORDERINGS}, got {by!r}")
    low, high = width_range
    middle = (low + high) / 2

    def width_of(index: int) -> float:
        image = np.asarray(images[index])
        return image.shape[1] * reference_height / max(1, image.shape[0])

    def coverage(index: int) -> float:
        wanted = set(target or "") - {" "}
        if not wanted:
            return 0.0
        return -len(wanted & set(style_texts[index])) / len(wanted)

    def key(index: int) -> tuple[int, float, float]:
        width = width_of(index)
        outside = not (low <= width <= high)
        # Outside the range, all that matters is how far outside.
        if outside:
            return (1, min(abs(width - low), abs(width - high)), 0.0)
        if by == "letters" and style_texts is not None and target:
            return (0, coverage(index), abs(width - middle))
        return (0, abs(width - middle), 0.0)

    return sorted(range(len(images)), key=key)


def doubled(reading: str, target: str) -> str | None:
    """A word the reading says twice in a row where the text says it once.

    Emuru sometimes writes a word and then writes it again -- "of our" came back
    as "own own" on a line the reader still read at 18% CER, well inside
    :data:`ACCEPT_CER`, so nothing set it aside. A reader rarely invents a
    repetition of its own: over 7s's 150 kept lines this names 5, and one of them
    is that line. Repetitions the text itself asks for ("had had") are allowed.
    """
    said = [word.lower() for word in reading.split()]
    asked = [word.lower() for word in target.split()]
    allowed = {second for first, second in pairwise(asked) if first == second}
    for first, second in pairwise(said):
        if first == second and second not in allowed:
            return second
    return None


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


def predicted_width(
    style_images: Sequence[np.ndarray], style_texts: Sequence[str] | None, target: str
) -> float | None:
    """How wide this hand should need for ``target``, from its own style lines.

    Every style line carries a known transcription -- the user supplies one for
    their page at enrolment, always -- so each gives this writer's pixels per
    character, and their mean predicts any other text in the same hand. Nothing
    is needed from the target's real line, which at generation time does not
    exist.

    Returns None when there is nothing to predict from: a generator that needs no
    transcription of its style line, such as DiffBrush, leaves the width
    unjudged rather than judged on a guess.
    """
    if not style_texts or not target.strip():
        return None
    rates = [
        float(np.asarray(image).shape[1]) / len(text)
        for image, text in zip(style_images, style_texts, strict=True)
        if len(text) > 0
    ]
    if not rates:
        return None
    return float(np.mean(rates)) * len(target)


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

    omission: Omission | None = None
    """The word the draw most clearly lacks, when a verifier was given."""

    width_ratio: float | None = None
    """This draw's width as a share of what the style lines predict for the text,
    or None where there was no transcription to predict from."""

    reading: str = ""
    """What the selector read, kept so the thresholds above can be calibrated."""

    similarity: float | None = None
    """Cosine similarity to the writer's page, when selecting by hand."""

    doubled: str | None = None
    """A word this draw writes twice in a row that the text asks for once."""


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

    rejected_for_omission: int = 0
    """Draws that read well enough but left a word of the text out. One draw can be
    counted here and under the two neighbouring fields at once: these name the
    faults found, not a partition of the draws."""

    rejected_for_width: int = 0
    """Draws that read well enough but were the wrong width for their text."""

    rejected_for_doubling: int = 0
    """Draws that read well enough but wrote a word twice."""

    chosen_scores: list[float] = field(default_factory=list)
    """The selector's CER for each kept image, in request order."""

    chosen_readings: list[str] = field(default_factory=list)
    """What the selector read of each kept image, in request order. No run before
    2026-09-21 stored it, so no threshold on it could be checked afterwards."""

    chosen_omissions: list[Omission | None] = field(default_factory=list)
    """The word each kept image most clearly lacks, and by how much, where a
    verifier was given -- so a kept line that still lost a word can be found."""

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
            "rejected_for_omission": self.rejected_for_omission,
            "rejected_for_width": self.rejected_for_width,
            "rejected_for_doubling": self.rejected_for_doubling,
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
            f"  {self.rejected_for_omission} read well but left a word out, set aside",
            f"  {self.rejected_for_width} read well but was the wrong width for the text",
            f"  {self.rejected_for_doubling} read well but wrote a word twice",
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
        verifier: Verifier | None = None,
        omission_support: float = OMISSION_SUPPORT,
        width_band: tuple[float, float] = WIDTH_BAND,
        style_by: str = "width",
        extra_draws: int = EXTRA_DRAWS,
    ) -> None:
        if candidates < 1:
            raise GeneratorError(f"need at least one candidate, got {candidates}")
        self.base = base
        self.selector = selector
        self.candidates = candidates
        self.accept_cer = accept_cer
        self.accept_overrun = accept_overrun
        self.verifier = verifier
        self.omission_support = omission_support
        self.width_band = width_band
        self.width_range = width_range
        self.style_by = style_by
        self.extra_draws = extra_draws
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
        chosen = "" if self.style_by == "width" else f"-style-by-{self.style_by}"
        return f"{self.base.name}+best-of-{self.candidates}{rule}{chosen}"

    @property
    def output_height(self) -> int:
        return self.base.output_height

    def generate(self, requests: Sequence[GenerationRequest]) -> list[np.ndarray]:
        return [self._one(request) for request in requests]

    def _one(self, request: GenerationRequest) -> np.ndarray:
        order = style_order(
            request.style_images,
            self.width_range,
            style_texts=request.style_texts,
            target=request.text,
            by=self.style_by,
        )
        draws: list[Draw] = []
        empty_draws = 0
        used = 0

        for attempt in range(self.candidates + self.extra_draws):
            if attempt >= self.candidates and any(self._readable(d) for d in draws):
                break
            used += 1
            draw = self._draw(request, order[attempt % len(order)])
            if draw is None:
                empty_draws += 1
                continue
            draws.append(draw)
            # Named faults, not a partition: a draw can carry more than one.
            if draw.score <= self.accept_cer:
                if draw.overrun > self.accept_overrun:
                    self.selection.rejected_for_overrun += 1
                if self._omitted(draw):
                    self.selection.rejected_for_omission += 1
                if not self._in_band(draw):
                    self.selection.rejected_for_width += 1
                if draw.doubled is not None:
                    self.selection.rejected_for_doubling += 1
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
        self.selection.chosen_readings.append(best.reading)
        self.selection.chosen_omissions.append(best.omission)

        self.truncations.generated += 1
        if best.truncated:
            self.truncations.events.append(self._last_truncation(request, best))
        return best.image

    def _readable(self, draw: Draw) -> bool:
        return (
            draw.score <= self.accept_cer
            and draw.overrun <= self.accept_overrun
            and not self._omitted(draw)
            and self._in_band(draw)
            and draw.doubled is None
        )

    def _omitted(self, draw: Draw) -> bool:
        return draw.omission is not None and draw.omission.support > self.omission_support

    def _in_band(self, draw: Draw) -> bool:
        """True when the width was not judged at all, so a generator that carries
        no style transcription is never rejected for a ratio nobody could compute."""
        if draw.width_ratio is None:
            return True
        low, high = self.width_band
        return low <= draw.width_ratio < high

    def _choose(self, request: GenerationRequest, draws: list[Draw]) -> Draw:
        readable = [draw for draw in draws if self._readable(draw)]
        if not readable:
            # Nothing is acceptable, so keep the least bad. Over the CER threshold
            # a draw is a smear whatever else it has; below it, a draw that says
            # the whole text beats one that left a word out -- which ordering on
            # CER alone got backwards, since a missing short word costs fewer
            # characters than misreading the line that carried it.
            return min(
                draws,
                key=lambda draw: (
                    draw.score > self.accept_cer,
                    self._omitted(draw),
                    draw.doubled is not None,
                    draw.score,
                ),
            )
        if self.hand is None:
            return readable[0]

        page = self._page(request)
        vectors = _unit(self.hand([draw.image for draw in readable]))
        similarities = vectors @ page
        scored = [
            Draw(
                d.image,
                d.score,
                d.style_index,
                d.truncated,
                d.overrun,
                d.omission,
                d.width_ratio,
                d.reading,
                float(s),
            )
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
        expected = predicted_width(request.style_images, request.style_texts, request.text)
        return Draw(
            image,
            cer(reading, request.text),
            index,
            truncated,
            overrun(reading, request.text),
            self.verifier.omissions([image], [request.text])[0] if self.verifier else None,
            None if not expected else float(image.shape[1]) / expected,
            reading,
            doubled=doubled(reading, request.text),
        )

    def _last_truncation(self, request: GenerationRequest, draw: Draw):
        from nib.models.emuru import Truncation

        base_log = getattr(self.base, "truncations", None)
        matching = [e for e in (base_log.events if base_log else []) if e.text == request.text]
        if matching:
            return matching[-1]
        return Truncation(text=request.text, width=int(draw.image.shape[1]), budget=0)
