"""Pack CVL's cropped images into a single LMDB file.

    python scripts/build_index.py                       # words (the default)
    python scripts/build_index.py --unit lines          # lines
    python scripts/build_index.py --unit lines data.image_height=96

Reads every usable crop, normalises it to the configured height, and writes one
file. That file is what goes to Drive and gets copied to the Colab VM at the
start of a session -- one large sequential copy instead of a hundred thousand
small reads over a network filesystem.

**Two units, one script.** Words and lines differ in four things and nothing
else: which reader finds them, which normaliser fits them, what a record's key
is, and what the header calls the source. Those four live in :data:`UNITS` below,
and everything after it is shared. A second near-identical script would be the
easier thing to write and the worse thing to read.

Which unit you want depends on what the pack is for. The style embedding was
trained on words. The generator was trained on lines and must be evaluated on
lines -- fixing a *word* to a common height destroys its relative scale, which is
part of how a hand looks, and that is what made the first word-level generation
attempt produce faint marks and truncated output.

Safe to interrupt: the header is written last, so a partial file is detectably
incomplete rather than quietly short.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from nib.config import ensure_dirs, get_path, load_config
from nib.data.cvl_lines import scan_lines
from nib.data.cvl_words import scan_words
from nib.data.pack import PackedSample, PackHeader, PackReader, PackWriter, compact
from nib.data.preprocessing import normalise_line, normalise_word


@dataclass(frozen=True)
class _Unit:
    """Everything that differs between packing words and packing lines."""

    scan: Callable
    """(root, charset_name=...) -> (records, report). The report is printed in
    full: every excluded sample is counted with a reason, and a build that
    silently loses a tenth of the data is exactly what that rule exists to stop."""

    normalise: Callable
    """(image, height) -> image."""

    key: Callable
    """record -> the string a record is stored and looked up under."""

    source: str
    """Written into the pack header, so a file can say what it holds."""


UNITS: dict[str, _Unit] = {
    "words": _Unit(
        scan=scan_words,
        normalise=normalise_word,
        key=lambda record: record.word_id,
        # "cvl" rather than "cvl-words": the pack already built and uploaded to
        # Drive carries this value, and every phase-1 number was measured on that
        # artefact. Renaming it now would make a rebuild disagree with the file
        # the results came from, for no gain.
        source="cvl",
    ),
    "lines": _Unit(
        scan=scan_lines,
        normalise=normalise_line,
        key=lambda record: record.line_id,
        source="cvl-lines",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=sorted(UNITS), default="words")
    parser.add_argument("--out", default=None, help="output pack path")
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N records (for testing)"
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="skip the compacted copy under processed/upload/",
    )
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)
    unit = UNITS[args.unit]

    ensure_dirs(cfg, "processed")
    out = (
        Path(args.out)
        if args.out
        else get_path(cfg, "processed") / f"cvl_{args.unit}_{height}.lmdb"
    )

    print(f"scanning {get_path(cfg, 'raw') / 'cvl'} for {args.unit} ...")
    records, report = unit.scan(get_path(cfg, "raw") / "cvl", charset_name=str(cfg.data.charset))
    print(report.summary())
    if not records:
        print("\nnothing to pack. Is the full CVL release extracted?")
        return 1

    if args.limit:
        records = records[: args.limit]

    header = PackHeader(
        height=height,
        charset=str(cfg.data.charset),
        source=unit.source,
        config={"image_height": height, "charset": str(cfg.data.charset), "unit": args.unit},
    )

    print(f"\npacking {len(records)} {args.unit} at {height}px into {out}")
    started = time.perf_counter()
    unreadable = 0
    every = max(1000, len(records) // 10)

    with PackWriter(out, header) as writer:
        for index, record in enumerate(records, start=1):
            image = cv2.imread(str(record.image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                unreadable += 1
                continue
            writer.add(
                PackedSample(
                    key=unit.key(record),
                    writer_id=record.writer_id,
                    text=record.text,
                    split=record.split,
                    image=unit.normalise(image, height),
                )
            )
            if index % every == 0:
                rate = index / (time.perf_counter() - started)
                print(f"  {index:>6} / {len(records)}   {rate:.0f}/s")

    elapsed = time.perf_counter() - started
    print(f"\ndone in {elapsed:.0f}s")
    if unreadable:
        # Counted, not shrugged at. cv2.imread returns None for a genuinely
        # corrupt file and also for a perfectly good one whose path it could not
        # encode, and the two want different responses.
        print(f"  {unreadable} images could not be read and were skipped")

    with PackReader(out) as reader:
        print()
        print(reader.summary())
        sample = reader[0]
        print(
            f"\nfirst record: {sample.key}  writer={sample.writer_id}  "
            f"text={sample.text!r}  image={sample.image.shape[1]}x{sample.image.shape[0]}"
        )

    if not args.no_compact:
        _write_upload_copy(cfg, out)
    return 0


def _write_upload_copy(cfg, out: Path) -> None:
    """Write the compacted copy that actually gets uploaded.

    LMDB reserves its whole map size up front, and on Windows that means the file
    is *created* at 8 GB. It is sparse, so it costs nothing on disk -- and copying
    it anywhere transfers the full 8 GB, which would undo the entire point of
    packing. This has already cost one upload here.

    So the compacted copy is produced by the build rather than left as a step
    someone has to remember. ``pack.compact`` leaves the original alone when it
    is given a destination.
    """
    upload = get_path(cfg, "processed") / "upload"
    upload.mkdir(parents=True, exist_ok=True)

    print(f"\ncompacting to {upload / out.name} ...")
    started = time.perf_counter()
    target = compact(out, upload / out.name)
    size_mb = target.stat().st_size / 1024**2
    print(f"  {size_mb:.0f} MB in {time.perf_counter() - started:.0f}s   <- upload this one")


if __name__ == "__main__":
    sys.exit(main())
