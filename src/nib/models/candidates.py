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

ACCEPT_UNDERRUN: int | None = None
"""The longest run of its text a readable draw may leave out -- see
:func:`underrun` -- or None for not judged. **Off by default, because the reader
cannot be trusted with this question.**

The check was built for a real failure: on Amri's page the generator wrote "warm
noon" for "warm at noon", visibly, in the pixels. It was set to 1 on readings of
that page -- but those readings were taken by eye, a reader that misses nothing.
The pipeline's reader is TrOCR-small, and run end to end on the fake generator,
which draws every character of its text in a clean typeface, it left words out on
its own: "not the rapid calculation" read "not rapid calculation", "Whirlwind or
Typhoon" read "whirlwind typhoon", "it will be enough" read "it will". Over twelve
complete lines its runs reached 3 routinely and 8 once, and at 1 it set aside 16
of 24 draws that were entirely correct.

A reader that drops short words cannot tell a word the generator left out from
one it skipped itself, and the size of the failure being hunted -- one short word
-- is exactly the size of its noise. Turned on at a threshold its noise does not
reach, the check would catch nothing that matters; turned on at 1, it rejects
sound draws, and in ``hand`` mode a request with nothing readable skips the choice
by hand altogether.

Kept, with ``SelectionLog.chosen_readings``, for a reader that can answer it: one
asked *whether this text is in the image* rather than *what text is in the image*,
which a free-running decoder with a language model's habits is not.
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


def underrun(reading: str, target: str) -> int:
    """The longest run of consecutive target characters the reading does not carry.

    The mirror of :func:`overrun`, and the check this pipeline never had. A draw
    can leave a word out, read well under ``ACCEPT_CER``, and carry nothing
    beyond its text -- so every rule in this module passed it. On Amri's page
    "warm at noon" came back "walm noon", "Order #378 at Lior's Cafe" came back
    "Order 3 t Lior's Cafe", and "If You find it, Please" came back "I You find
    .. please". A reader rejects a page with words missing before forming any
    opinion of the handwriting.

    **A run, not a total.** Single characters missed here and there are the
    reader's own noise, and CER counts them already; three in a row are a word
    that was not written. This is the same distinction :func:`overrun` draws on
    the other side, where the two ends are taken by ``max`` rather than summed.

    Spaces are removed from both sides first, as there, so a reader's spacing
    costs nothing -- which also means a dropped two-letter word scores 2 and not
    3, the space going with it.

    **Ties, and why they break the other way here.** The alignment is the
    cheapest one; among equally cheap ones the walk prefers the deletion. That
    is the opposite of :func:`overrun`, which charges an ambiguous end to CER and
    errs toward keeping the draw. Writing a little beyond the text is cosmetic,
    so erring toward keeping is right there. A word left out is the failure that
    makes a page unusable, so erring toward *catching* it is right here: a miss
    ships a page with a word gone, while a false alarm costs one redraw -- and
    none at all in ``hand`` mode, where every draw is made regardless.

    A misread letter is still not counted, because substituting it is strictly
    cheaper than deleting and inserting, and this rule only decides ties.
    """
    read = "".join(reading.split())
    text = "".join(target.split())
    if not text:
        return 0

    # cost[i][j]: the cheapest alignment of the first i characters of the text
    # with the first j of the reading.
    cost = [[0] * (len(read) + 1) for _ in range(len(text) + 1)]
    for i in range(1, len(text) + 1):
        cost[i][0] = i
    for j in range(1, len(read) + 1):
        cost[0][j] = j
    for i in range(1, len(text) + 1):
        previous, row = cost[i - 1], cost[i]
        for j in range(1, len(read) + 1):
            row[j] = min(
                previous[j - 1] + (text[i - 1] != read[j - 1]),
                row[j - 1] + 1,  # the reading carries a character the text does not
                previous[j] + 1,  # the text carries one the reading does not
            )

    longest = run = 0
    i, j = len(text), len(read)
    while i > 0:
        here = cost[i][j]
        if cost[i - 1][j] + 1 == here:  # the text's character is not in the reading
            i -= 1
            run += 1
            longest = max(longest, run)
        elif j > 0 and cost[i - 1][j - 1] + (text[i - 1] != read[j - 1]) == here:
            i, j, run = i - 1, j - 1, 0
        else:
            # A character of the reading that answers to nothing in the text does
            # not interrupt a run: the text's characters either side of it are
            # both still absent from the reading.
            j -= 1
    return longest


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

    underrun: int = 0
    """The longest run of the text's characters the reading does not carry."""

    width_ratio: float | None = None
    """This draw's width as a share of what the style lines predict for the text,
    or None where there was no transcription to predict from."""

    reading: str = ""
    """What the selector read, kept so the thresholds above can be calibrated."""

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

    rejected_for_underrun: int = 0
    """Draws that read well enough but left part of the text out. One draw can be
    counted here and under the two neighbouring fields at once: these name the
    faults found, not a partition of the draws."""

    rejected_for_width: int = 0
    """Draws that read well enough but were the wrong width for their text."""

    chosen_scores: list[float] = field(default_factory=list)
    """The selector's CER for each kept image, in request order."""

    chosen_readings: list[str] = field(default_factory=list)
    """What the selector read of each kept image, in request order. Stored because
    ``ACCEPT_UNDERRUN`` cannot be calibrated without it, and no run before this
    one kept it."""

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
            "rejected_for_underrun": self.rejected_for_underrun,
            "rejected_for_width": self.rejected_for_width,
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
            f"  {self.rejected_for_underrun} read well but left part of the text out, set aside",
            f"  {self.rejected_for_width} read well but was the wrong width for the text",
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
        accept_underrun: int | None = ACCEPT_UNDERRUN,
        width_band: tuple[float, float] = WIDTH_BAND,
        style_by: str = "width",
    ) -> None:
        if candidates < 1:
            raise GeneratorError(f"need at least one candidate, got {candidates}")
        self.base = base
        self.selector = selector
        self.candidates = candidates
        self.accept_cer = accept_cer
        self.accept_overrun = accept_overrun
        self.accept_underrun = accept_underrun
        self.width_band = width_band
        self.width_range = width_range
        self.style_by = style_by
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

        for attempt in range(self.candidates):
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
                if self._left_out(draw):
                    self.selection.rejected_for_underrun += 1
                if not self._in_band(draw):
                    self.selection.rejected_for_width += 1
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

        self.truncations.generated += 1
        if best.truncated:
            self.truncations.events.append(self._last_truncation(request, best))
        return best.image

    def _readable(self, draw: Draw) -> bool:
        return (
            draw.score <= self.accept_cer
            and draw.overrun <= self.accept_overrun
            and not self._left_out(draw)
            and self._in_band(draw)
        )

    def _left_out(self, draw: Draw) -> bool:
        return self.accept_underrun is not None and draw.underrun > self.accept_underrun

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
            return min(draws, key=lambda draw: draw.score)
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
                d.underrun,
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
            underrun(reading, request.text),
            None if not expected else float(image.shape[1]) / expected,
            reading,
        )

    def _last_truncation(self, request: GenerationRequest, draw: Draw):
        from nib.models.emuru import Truncation

        base_log = getattr(self.base, "truncations", None)
        matching = [e for e in (base_log.events if base_log else []) if e.text == request.text]
        if matching:
            return matching[-1]
        return Truncation(text=request.text, width=int(draw.image.shape[1]), budget=0)
