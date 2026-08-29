"""Read an IAM-layout dataset into word records.

Input is a directory laid out the way IAM lays itself out::

    <root>/xml/<form-id>.xml
    <root>/words/<prefix>/<form-id>/<word-id>.png

Output is a flat list of :class:`WordRecord` -- one per usable word image, each
carrying the writer id, which is the field everything downstream depends on.

The parser never drops a sample silently. Every exclusion is counted by reason and
reported, because "why do I have 94k words and not 115k" is a question that will
be asked, and guessing at the answer later is much more expensive than counting
now.

.. warning::
   This is written against the schema produced by :mod:`nib.data.fixture`, which
   is a *reconstruction* of IAM's -- the FKI site was unreachable when it was
   written. :func:`validate_schema` exists to catch the difference loudly rather
   than mislabelling every sample. ``tests/test_iam_real.py`` runs automatically
   against genuine IAM XML as soon as any appears under ``data/raw/iam``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from nib.data import charset as cs

# Reasons a word may be excluded. Keys are stable: they appear in reports and in
# the experiment log, so renaming one breaks comparisons across runs.
DROP_BAD_SEGMENTATION = "line segmentation marked err"
DROP_EMPTY_TEXT = "empty transcription"
DROP_OUT_OF_CHARSET = "characters outside charset"
DROP_MISSING_IMAGE = "image file missing"


@dataclass(frozen=True)
class WordRecord:
    """One word image and everything known about it."""

    word_id: str
    form_id: str
    line_id: str
    writer_id: str
    text: str
    image_path: Path

    @property
    def num_chars(self) -> int:
        return len(self.text)


@dataclass
class ParseReport:
    """What was kept, what was dropped, and why."""

    forms: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    writers: set[str] = field(default_factory=set)

    @property
    def total_seen(self) -> int:
        return self.kept + sum(self.dropped.values())

    def summary(self) -> str:
        lines = [
            f"forms parsed   {self.forms}",
            f"writers        {len(self.writers)}",
            f"words kept     {self.kept} of {self.total_seen}",
        ]
        for reason, count in self.dropped.most_common():
            share = 100.0 * count / self.total_seen if self.total_seen else 0.0
            lines.append(f"  dropped: {reason:<32} {count:>7}  ({share:.1f}%)")
        return "\n".join(lines)


class SchemaMismatch(RuntimeError):
    """Raised when the XML does not look like the structure this parser expects."""


def validate_schema(xml_path: Path) -> None:
    """Fail loudly, and usefully, if an XML file is not shaped as expected.

    Without this, a schema difference does not raise -- it produces zero records,
    or records with an empty writer id, and the failure surfaces much later as a
    mysterious quality problem.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise SchemaMismatch(f"{xml_path} is not well-formed XML: {exc}") from exc

    problems: list[str] = []
    if root.tag != "form":
        problems.append(f"root element is <{root.tag}>, expected <form>")
    if root.get("writer-id") is None:
        problems.append("<form> has no writer-id attribute")
    if root.get("id") is None:
        problems.append("<form> has no id attribute")

    lines = root.findall(".//line")
    if not lines:
        problems.append("no <line> elements found")

    words = root.findall(".//word")
    if not words:
        problems.append("no <word> elements found")
    else:
        first = words[0]
        if first.get("id") is None:
            problems.append("<word> has no id attribute")
        if first.get("text") is None:
            problems.append("<word> has no text attribute")

    if problems:
        raise SchemaMismatch(
            f"{xml_path} does not match the expected IAM schema:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nThe parser was written against a reconstruction of IAM's schema "
            "(see the warning in nib/data/iam_parser.py). If this is a genuine IAM "
            "file, the reconstruction is wrong and the parser needs adjusting to "
            "match -- not the other way round."
        )


def word_image_path(root: Path, form_id: str, word_id: str) -> Path:
    """Where IAM keeps the image for a word: words/<prefix>/<form-id>/<word-id>.png"""
    prefix = form_id.split("-")[0]
    return root / "words" / prefix / form_id / f"{word_id}.png"


def parse_form(
    xml_path: Path,
    root: Path,
    alphabet: cs.Charset,
    report: ParseReport,
    skip_bad_segmentation: bool = True,
    require_image: bool = True,
) -> list[WordRecord]:
    """Parse one form's XML into records, accumulating exclusions into ``report``."""
    form = ET.parse(xml_path).getroot()
    form_id = form.get("id", xml_path.stem)
    writer_id = form.get("writer-id")
    if writer_id is None:
        raise SchemaMismatch(f"{xml_path}: <form> has no writer-id")

    records: list[WordRecord] = []

    for line in form.findall(".//line"):
        line_id = line.get("id", "")
        # IAM flags lines whose automatic word segmentation failed. Their word
        # boxes are unreliable, so the images do not match their labels.
        bad_line = skip_bad_segmentation and line.get("segmentation", "ok") != "ok"

        for word in line.findall("word"):
            word_id = word.get("id", "")
            text = word.get("text", "")

            if bad_line:
                report.dropped[DROP_BAD_SEGMENTATION] += 1
                continue
            if not text.strip():
                report.dropped[DROP_EMPTY_TEXT] += 1
                continue
            if not alphabet.supports(text):
                report.dropped[DROP_OUT_OF_CHARSET] += 1
                continue

            image_path = word_image_path(root, form_id, word_id)
            if require_image and not image_path.is_file():
                report.dropped[DROP_MISSING_IMAGE] += 1
                continue

            records.append(
                WordRecord(
                    word_id=word_id,
                    form_id=form_id,
                    line_id=line_id,
                    writer_id=writer_id,
                    text=text,
                    image_path=image_path,
                )
            )

    report.writers.add(writer_id)
    report.kept += len(records)
    return records


def parse_dataset(
    root: Path | str,
    charset_name: str = "english",
    skip_bad_segmentation: bool = True,
    require_image: bool = True,
) -> tuple[list[WordRecord], ParseReport]:
    """Parse every form under ``root``.

    Returns the records in a stable order -- sorted by word id -- so that a run is
    reproducible regardless of how the filesystem happens to order directories.
    """
    root = Path(root)
    xml_dir = root / "xml"
    if not xml_dir.is_dir():
        raise FileNotFoundError(
            f"no xml/ directory under {root}. Expected an IAM-style layout: "
            "xml/<form-id>.xml and words/<prefix>/<form-id>/<word-id>.png"
        )

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"no .xml files in {xml_dir}")

    # Check the first file before parsing thousands, so a schema difference costs
    # a second rather than a full pass.
    validate_schema(xml_files[0])

    alphabet = cs.get(charset_name)
    report = ParseReport()
    records: list[WordRecord] = []

    for xml_path in xml_files:
        records.extend(
            parse_form(
                xml_path,
                root=root,
                alphabet=alphabet,
                report=report,
                skip_bad_segmentation=skip_bad_segmentation,
                require_image=require_image,
            )
        )
        report.forms += 1

    records.sort(key=lambda r: r.word_id)
    return records, report


def group_by_writer(records: list[WordRecord]) -> dict[str, list[WordRecord]]:
    """Records bucketed by writer.

    This is the shape the writer-disjoint split (T5) and writer-aware sampling
    (T8) both need, and getting it from one place keeps them consistent.
    """
    grouped: dict[str, list[WordRecord]] = {}
    for record in records:
        grouped.setdefault(record.writer_id, []).append(record)
    return grouped
