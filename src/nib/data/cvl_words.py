"""Read CVL's word-level release.

The full CVL release turns out to carry everything this project needs, in an
easier form than expected: **99,904 word images, already cropped**, with the
transcription encoded in the filename. No bounding-box extraction, no XML
parsing -- the ``_attributes.xml`` files hold only line geometry and contain no
text at all.

Filename layout::

    testset/words/0052/0052-1-0-0-Imagine.tif
                   |    |    | | |  |
                   |    |    | | |  transcription
                   |    |    | | word index within the line
                   |    |    | line index within the page
                   |    |    text id (which of the seven passages)
                   |    writer id
                   writer folder

Three traps live in that layout, and all three are handled below.

**Hyphens.** The transcription may itself contain hyphens, so the text is
everything *after the fourth* hyphen -- not "the fifth field". Roughly 875 of the
99,904 filenames have more than five fields and would be truncated by a naive
split.

**Gaps.** Word and line indices are not contiguous. Words whose segmentation
failed were dropped, so ``...-1-0`` can be followed by ``...-1-2``. Nothing may
assume a dense range.

**German.** One of the seven passages is Goethe's *Faust*, and another is a
German Wikipedia article. Those pages produce words outside an English charset.
They are excluded by default and counted, rather than dropped in silence.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from nib.data import charset as cs

# 0052-1-0-0-Imagine  ->  writer, text, line, word, transcription.
# The transcription is greedy to the end, so hyphens inside a word survive.
_WORD_NAME = re.compile(
    r"^(?P<writer>\d{4})-(?P<text>\d+)-(?P<line>\d+)-(?P<word>\d+)-(?P<text_content>.+)$"
)

DROP_OUT_OF_CHARSET = "characters outside charset"
DROP_EMPTY_TEXT = "empty transcription"
DROP_UNPARSED_NAME = "filename did not parse"

# Writer 0431 is present in the files, but the CVL readme states it was excluded
# from the writer-identification evaluation in the paper. Excluding it keeps our
# retrieval numbers comparable with published ones.
EXCLUDED_WRITERS = frozenset({"0431"})


@dataclass(frozen=True)
class CvlWord:
    """One cropped word image, and everything known about it."""

    writer_id: str
    text_id: str
    line_index: int
    word_index: int
    text: str
    image_path: Path
    split: str  # "trainset" or "testset", CVL's own division

    @property
    def word_id(self) -> str:
        return f"{self.writer_id}-{self.text_id}-{self.line_index}-{self.word_index}"

    @property
    def page_id(self) -> str:
        return f"{self.writer_id}-{self.text_id}"


@dataclass
class WordReport:
    """What was kept, what was dropped, and why."""

    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    writers: set[str] = field(default_factory=set)
    excluded_writers: set[str] = field(default_factory=set)
    excluded_words: int = 0

    @property
    def total_seen(self) -> int:
        """Every file looked at, deliberate exclusions included.

        Excluded writers are counted here rather than left out, so this number
        matches the file count on disk. A total that quietly disagrees with
        ``ls | wc -l`` is how small losses go unnoticed.
        """
        return self.kept + sum(self.dropped.values()) + self.excluded_words

    def summary(self) -> str:
        lines = [
            f"writers        {len(self.writers)}",
            f"words kept     {self.kept} of {self.total_seen}",
        ]
        for reason, count in self.dropped.most_common():
            share = 100.0 * count / self.total_seen if self.total_seen else 0.0
            lines.append(f"  dropped: {reason:<32} {count:>7}  ({share:.1f}%)")
        if self.excluded_writers:
            lines.append(
                f"  excluded: {self.excluded_words} words from writers "
                f"{sorted(self.excluded_writers)} (see the CVL readme)"
            )
        return "\n".join(lines)


def parse_word_filename(stem: str) -> tuple[str, str, int, int, str] | None:
    """Split a word filename into its five parts, hyphens in the text included."""
    match = _WORD_NAME.match(stem)
    if match is None:
        return None
    return (
        match.group("writer"),
        match.group("text"),
        int(match.group("line")),
        int(match.group("word")),
        match.group("text_content"),
    )


def scan_words(
    root: Path | str,
    charset_name: str = "english",
    exclude_writers: frozenset[str] = EXCLUDED_WRITERS,
    keep_out_of_charset: bool = False,
) -> tuple[list[CvlWord], WordReport]:
    """Find every cropped word image under ``root``.

    Args:
        root: the CVL directory. Both ``trainset`` and ``testset`` are searched.
        charset_name: transcriptions outside this alphabet are excluded. The
            German passages are what this mostly removes.
        exclude_writers: writer ids to skip entirely. Defaults to the one the CVL
            readme says was excluded from the published evaluation.
        keep_out_of_charset: keep German and other unsupported words anyway.

    Returns:
        The words, sorted by id for reproducibility, and a report of exclusions.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no CVL directory at {root}")

    alphabet = cs.get(charset_name)
    report = WordReport()
    words: list[CvlWord] = []

    for path in root.rglob("*.tif"):
        parts = path.parts
        if "words" not in parts:
            continue
        split = "trainset" if "trainset" in parts else "testset"

        parsed = parse_word_filename(path.stem)
        if parsed is None:
            report.dropped[DROP_UNPARSED_NAME] += 1
            continue

        writer_id, text_id, line_index, word_index, text = parsed

        if writer_id in exclude_writers:
            report.excluded_writers.add(writer_id)
            report.excluded_words += 1
            continue
        if not text.strip():
            report.dropped[DROP_EMPTY_TEXT] += 1
            continue
        if not keep_out_of_charset and not alphabet.supports(text):
            report.dropped[DROP_OUT_OF_CHARSET] += 1
            continue

        words.append(
            CvlWord(
                writer_id=writer_id,
                text_id=text_id,
                line_index=line_index,
                word_index=word_index,
                text=text,
                image_path=path,
                split=split,
            )
        )
        report.writers.add(writer_id)

    words.sort(key=lambda w: (w.writer_id, w.text_id, w.line_index, w.word_index))
    report.kept = len(words)
    return words, report


def group_by_writer(words: list[CvlWord]) -> dict[str, list[CvlWord]]:
    grouped: dict[str, list[CvlWord]] = {}
    for word in words:
        grouped.setdefault(word.writer_id, []).append(word)
    return grouped


def group_by_page(words: list[CvlWord]) -> dict[str, list[CvlWord]]:
    grouped: dict[str, list[CvlWord]] = {}
    for word in words:
        grouped.setdefault(word.page_id, []).append(word)
    return grouped


def official_split(words: list[CvlWord]) -> dict[str, list[CvlWord]]:
    """CVL's own division, which is writer-disjoint: 27 writers train, 283 test.

    Lopsided because it was built for writer-identification benchmarks, not for
    us. Useful only when reporting numbers meant to be comparable with published
    CVL results; otherwise prefer the project's own split in
    ``configs/splits/``.
    """
    out: dict[str, list[CvlWord]] = {"trainset": [], "testset": []}
    for word in words:
        out[word.split].append(word)
    return out
