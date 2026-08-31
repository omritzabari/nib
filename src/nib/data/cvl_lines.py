"""Read CVL's line-level release.

Emuru generates *lines*, so lines are what it must be evaluated on. This module
is the line-level twin of :mod:`nib.data.cvl_words`, and it exists because word
crops turned out to be the wrong unit: ``normalise_word`` stretches every crop to
a fixed height, so a one-letter word ends up as tall as a whole line and relative
scale -- part of how a hand looks -- is destroyed. Lines are all roughly the same
height, so normalising them to a fixed height preserves scale instead.

CVL ships 13,473 line crops but **no line-level transcription**. The text has to
be reassembled from the word filenames::

    testset/lines/0052/0052-1-0.tif              the image
    testset/words/0052/0052-1-0-0-Imagine.tif    word 0 of that line
    testset/words/0052/0052-1-0-1-all.tif        word 1

so the line reads "Imagine all ...", in word-index order.

**The trap this module exists to close.** CVL dropped the word crops whose
segmentation failed, which leaves gaps in the word indices -- ``0, 2, 3, 4`` for a
line that has five words. The *line image still contains the missing word's ink*.
Pairing that image with a transcription that omits the word would tell the CER
metric that the recogniser hallucinated something it in fact read correctly.

Measured on the real database: lines with index gaps carry 0.58 more ink blobs
per line than complete lines do, relative to their own word count. There is real
untranscribed ink there. 1,157 lines are affected; they are dropped and counted
under :data:`DROP_INCOMPLETE_TRANSCRIPTION`. 10,862 clean lines remain, and every
one of the 94 held-out writers keeps at least 16 -- so dropping them costs
nothing and keeping them would cost a corrupted metric.

**Counting order.** A line is counted under the *first* reason that applies, and
the order is deliberate: out-of-charset comes before incomplete, so the German
passages are attributed to the English-only scope decision and the "incomplete"
figure reports only the English lines genuinely lost to data quality.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from nib.data import charset as cs
from nib.data.cvl_words import EXCLUDED_WRITERS, parse_word_filename

# 0052-1-0  ->  writer 0052, text 1, line 0. Exactly three fields: a line image
# has no transcription in its name, which is the whole reason this module exists.
_LINE_NAME = re.compile(r"^(?P<writer>\d{4})-(?P<text>\d+)-(?P<line>\d+)$")

DROP_UNPARSED_NAME = "filename did not parse"
DROP_NO_WORDS = "no word files for the line"
DROP_EMPTY_TEXT = "empty transcription"
DROP_OUT_OF_CHARSET = "characters outside charset"
DROP_INCOMPLETE_TRANSCRIPTION = "word indices have gaps -- untranscribed ink"


@dataclass(frozen=True)
class CvlLine:
    """One cropped line image, and the text reassembled from its words."""

    writer_id: str
    text_id: str
    line_index: int
    text: str
    image_path: Path
    split: str  # "trainset" or "testset", CVL's own division
    word_count: int
    complete: bool = True
    """False when the word indices had gaps and ``keep_incomplete`` let the line
    through anyway. Such a record's image contains ink its text does not account
    for, so it is usable for FID and writer retrieval but never for CER."""

    @property
    def line_id(self) -> str:
        return f"{self.writer_id}-{self.text_id}-{self.line_index}"

    @property
    def page_id(self) -> str:
        return f"{self.writer_id}-{self.text_id}"


@dataclass
class LineReport:
    """What was kept, what was dropped, and why."""

    kept: int = 0
    incomplete_kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    writers: set[str] = field(default_factory=set)
    excluded_writers: set[str] = field(default_factory=set)
    excluded_lines: int = 0

    @property
    def total_seen(self) -> int:
        """Every line image looked at, deliberate exclusions included, so the
        total matches the file count on disk."""
        return self.kept + sum(self.dropped.values()) + self.excluded_lines

    def summary(self) -> str:
        lines = [
            f"writers        {len(self.writers)}",
            f"lines kept     {self.kept} of {self.total_seen}",
        ]
        if self.incomplete_kept:
            lines.append(
                f"  of those, {self.incomplete_kept} have incomplete transcriptions "
                "(kept on request; not usable for CER)"
            )
        for reason, count in self.dropped.most_common():
            share = 100.0 * count / self.total_seen if self.total_seen else 0.0
            lines.append(f"  dropped: {reason:<44} {count:>6}  ({share:.1f}%)")
        if self.excluded_writers:
            lines.append(
                f"  excluded: {self.excluded_lines} lines from writers "
                f"{sorted(self.excluded_writers)} (see the CVL readme)"
            )
        return "\n".join(lines)


def index_words_by_line(root: Path | str) -> dict[tuple[str, str, int], list[tuple[int, str]]]:
    """Every word filename under ``root``, bucketed by the line it belongs to.

    One pass over the names, no image reads. Globbing a writer's word directory
    once per line instead would be 13,473 directory scans over ~100,000 files.

    Keyed by writer, text and line without the split, because CVL writer ids are
    unique across ``trainset`` and ``testset`` -- so the key is unambiguous, and
    pairing a line image to its words needs no guessing at directory layout.

    Deliberately **unfiltered**: charset and writer exclusions are applied by the
    caller. Filtering here would remove the very filenames that reveal a gap, and
    a line whose German word was filtered out would then look complete.
    """
    root = Path(root)
    index: dict[tuple[str, str, int], list[tuple[int, str]]] = {}

    for path in root.rglob("*.tif"):
        if "words" not in path.parts:
            continue
        parsed = parse_word_filename(path.stem)
        if parsed is None:
            continue
        writer_id, text_id, line_index, word_index, text = parsed
        index.setdefault((writer_id, text_id, line_index), []).append((word_index, text))

    for words in index.values():
        words.sort()
    return index


def scan_lines(
    root: Path | str,
    charset_name: str = "english",
    exclude_writers: frozenset[str] = EXCLUDED_WRITERS,
    keep_out_of_charset: bool = False,
    keep_incomplete: bool = False,
) -> tuple[list[CvlLine], LineReport]:
    """Find every line image under ``root`` whose transcription can be trusted.

    Args:
        root: the CVL directory. Both ``trainset`` and ``testset`` are searched.
        charset_name: lines containing characters outside this alphabet are
            dropped. The German passages are what this mostly removes.
        exclude_writers: writer ids to skip entirely. Defaults to the one the CVL
            readme says was excluded from the published evaluation.
        keep_out_of_charset: keep German and other unsupported lines anyway.
        keep_incomplete: keep lines whose word indices have gaps, flagged with
            ``complete=False``. Their images hold ink their text does not
            describe, so they must never reach CER.

    Returns:
        The lines, sorted by id for reproducibility, and a report of exclusions.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no CVL directory at {root}")

    alphabet = cs.get(charset_name)
    words_by_line = index_words_by_line(root)
    report = LineReport()
    lines: list[CvlLine] = []

    for path in root.rglob("*.tif"):
        if "lines" not in path.parts:
            continue
        split = "trainset" if "trainset" in path.parts else "testset"

        match = _LINE_NAME.match(path.stem)
        if match is None:
            report.dropped[DROP_UNPARSED_NAME] += 1
            continue

        writer_id = match.group("writer")
        text_id = match.group("text")
        line_index = int(match.group("line"))

        if writer_id in exclude_writers:
            report.excluded_writers.add(writer_id)
            report.excluded_lines += 1
            continue

        words = words_by_line.get((writer_id, text_id, line_index), [])
        if not words:
            report.dropped[DROP_NO_WORDS] += 1
            continue

        text = " ".join(word for _, word in words)
        if not text.strip():
            report.dropped[DROP_EMPTY_TEXT] += 1
            continue
        if not keep_out_of_charset and not alphabet.supports(text):
            report.dropped[DROP_OUT_OF_CHARSET] += 1
            continue

        # Complete means the surviving word crops are 0, 1, .. n-1 with nothing
        # missing. Anything else and the image holds ink the text does not cover.
        indices = [index for index, _ in words]
        complete = indices == list(range(len(indices)))
        if not complete:
            if not keep_incomplete:
                report.dropped[DROP_INCOMPLETE_TRANSCRIPTION] += 1
                continue
            report.incomplete_kept += 1

        lines.append(
            CvlLine(
                writer_id=writer_id,
                text_id=text_id,
                line_index=line_index,
                text=text,
                image_path=path,
                split=split,
                word_count=len(words),
                complete=complete,
            )
        )
        report.writers.add(writer_id)

    lines.sort(key=lambda line: (line.writer_id, line.text_id, line.line_index))
    report.kept = len(lines)
    return lines, report


def group_by_writer(lines: list[CvlLine]) -> dict[str, list[CvlLine]]:
    """Writer id -> their lines, the shape the split and the retrieval metric need."""
    grouped: dict[str, list[CvlLine]] = {}
    for line in lines:
        grouped.setdefault(line.writer_id, []).append(line)
    return grouped
