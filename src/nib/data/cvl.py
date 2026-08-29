"""Read the CVL database.

CVL is our primary evaluation set: 310 writers, 1604 pages, and -- unusually --
writer identity is a first-class field, because writer retrieval is the task the
database was built for. That is exactly one of our three metrics.

Two releases exist and they carry different things:

``cvl-database-cropped-1-1.zip`` (1.2 GB)
    Page images only, pre-trimmed to the handwritten region. Writer id and text id
    are encoded in the filename: ``0001-2-cropped.tif`` is writer 0001, text 2.
    No transcriptions, no word boxes.

``cvl-database-1-1.zip`` (4.2 GB)
    The full release: the same pages, plus ~99,900 pre-cropped **word** images and
    ~13,800 line images, split into CVL's own writer-disjoint ``trainset`` and
    ``testset``.

    Its ``_attributes.xml`` files, despite the name, contain **no text** -- only
    line geometry, in PRImA PAGE format. The transcriptions live in the word image
    *filenames*. That is why :mod:`nib.data.cvl_words` exists and why this module
    does not try to parse XML.

This module handles pages only. Word-level reading lives in
:mod:`nib.data.cvl_words`.

Observed in the cropped release and not explained in the documentation: text ids
run 1, 2, 3, 4, 6, 7, 8. **There is no text 5.** 283 writers produced texts
1-4 and 6; 27 produced all seven. Do not assume contiguous numbering.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# 0001-2-cropped.tif  ->  writer 0001, text 2
_CROPPED = re.compile(r"^(?P<writer>\d{4})-(?P<text>\d+)-cropped\.(?:tif|tiff|png|jpg)$", re.I)
# 0001-2.tif in the full release
_PLAIN = re.compile(r"^(?P<writer>\d{4})-(?P<text>\d+)\.(?:tif|tiff|png|jpg)$", re.I)


@dataclass(frozen=True)
class CvlPage:
    """One scanned page, and the writer who wrote it."""

    page_id: str  # "0001-2"
    writer_id: str  # "0001"
    text_id: str  # "2"
    image_path: Path
    text: str | None = None  # filled in only when the full release is present


@dataclass
class CvlInventory:
    """What was found, and what it is therefore possible to compute."""

    root: Path
    pages: list[CvlPage]
    unparsed: list[Path]
    has_annotations: bool  # full release present; NOT "transcriptions present"

    @property
    def writers(self) -> set[str]:
        return {p.writer_id for p in self.pages}

    @property
    def texts(self) -> set[str]:
        return {p.text_id for p in self.pages}

    def pages_per_writer(self) -> Counter:
        return Counter(p.writer_id for p in self.pages)

    def summary(self) -> str:
        counts = self.pages_per_writer()
        spread = Counter(counts.values())
        lines = [
            f"root            {self.root}",
            f"pages           {len(self.pages)}",
            f"writers         {len(self.writers)}",
            f"text ids        {sorted(self.texts, key=_as_int)}",
            f"pages/writer    {dict(sorted(spread.items()))}",
            # XML presence signals the full release, not transcriptions -- the XML
            # holds line geometry only. See the module docstring.
            f"full release    {'yes' if self.has_annotations else 'no -- pages only'}",
        ]
        if self.unparsed:
            lines.append(f"unparsed files  {len(self.unparsed)} (first: {self.unparsed[0].name})")
        return "\n".join(lines)


def _as_int(value: str) -> int:
    return int(value)


def scan(root: Path | str) -> CvlInventory:
    """Find every CVL page under ``root``, at any depth.

    Searching recursively rather than at a fixed depth means it does not matter
    whether the archive was extracted into a subdirectory or flattened.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no CVL directory at {root}")

    pages: list[CvlPage] = []
    unparsed: list[Path] = []
    seen: set[str] = set()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg"}:
            continue
        # The full release also ships cropped word and line images. They belong to
        # nib.data.cvl_words, and reporting them here as "unparsed" would fire a
        # warning on 113,000 perfectly healthy files -- which trains you to ignore
        # the warning, and then it is worthless when something is genuinely wrong.
        if {"words", "lines"} & set(path.parts):
            continue
        match = _CROPPED.match(path.name) or _PLAIN.match(path.name)
        if match is None:
            unparsed.append(path)
            continue

        writer_id = match.group("writer")
        text_id = match.group("text")
        page_id = f"{writer_id}-{text_id}"
        if page_id in seen:
            # Both releases extracted side by side. Keep the first, which is the
            # cropped one by sort order, and do not double-count the writer.
            continue
        seen.add(page_id)

        pages.append(
            CvlPage(
                page_id=page_id,
                writer_id=writer_id,
                text_id=text_id,
                image_path=path,
            )
        )

    pages.sort(key=lambda p: p.page_id)
    has_annotations = any(root.rglob("*.xml"))
    return CvlInventory(root=root, pages=pages, unparsed=unparsed, has_annotations=has_annotations)


def group_by_writer(pages: list[CvlPage]) -> dict[str, list[CvlPage]]:
    """Pages bucketed by writer -- the shape the split and the retrieval metric need."""
    grouped: dict[str, list[CvlPage]] = {}
    for page in pages:
        grouped.setdefault(page.writer_id, []).append(page)
    return grouped
