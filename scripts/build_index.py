"""Pack the CVL word images into a single LMDB file.

    python scripts/build_index.py
    python scripts/build_index.py data.image_height=96

Reads every usable word crop, normalises it to the configured height, and writes
one file. That file is what goes to Drive and gets copied to the Colab VM at the
start of a session -- one large sequential copy instead of a hundred thousand
small reads over a network filesystem.

Safe to interrupt: the header is written last, so a partial file is detectably
incomplete rather than quietly short.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from nib.config import ensure_dirs, get_path, load_config
from nib.data.cvl_words import scan_words
from nib.data.pack import PackedWord, PackHeader, PackReader, PackWriter
from nib.data.preprocessing import normalise_word


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output pack path")
    parser.add_argument("--limit", type=int, default=None, help="stop after N words (for testing)")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)

    ensure_dirs(cfg, "processed")
    out = Path(args.out) if args.out else get_path(cfg, "processed") / f"cvl_words_{height}.lmdb"

    print(f"scanning {get_path(cfg, 'raw') / 'cvl'} ...")
    words, report = scan_words(get_path(cfg, "raw") / "cvl", charset_name=str(cfg.data.charset))
    print(report.summary())
    if not words:
        print("\nnothing to pack. Is the full CVL release extracted?")
        return 1

    if args.limit:
        words = words[: args.limit]

    header = PackHeader(
        height=height,
        charset=str(cfg.data.charset),
        source="cvl",
        config={"image_height": height, "charset": str(cfg.data.charset)},
    )

    print(f"\npacking {len(words)} words at {height}px into {out}")
    started = time.perf_counter()
    unreadable = 0

    with PackWriter(out, header) as writer:
        for index, word in enumerate(words, start=1):
            image = cv2.imread(str(word.image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                unreadable += 1
                continue
            writer.add(
                PackedWord(
                    key=word.word_id,
                    writer_id=word.writer_id,
                    text=word.text,
                    split=word.split,
                    image=normalise_word(image, height),
                )
            )
            if index % 10000 == 0:
                rate = index / (time.perf_counter() - started)
                print(f"  {index:>6} / {len(words)}   {rate:.0f} words/s")

    elapsed = time.perf_counter() - started
    print(f"\ndone in {elapsed:.0f}s")
    if unreadable:
        print(f"  {unreadable} images could not be read and were skipped")

    with PackReader(out) as reader:
        print()
        print(reader.summary())
        sample = reader[0]
        print(
            f"\nfirst record: {sample.key}  writer={sample.writer_id}  "
            f"text={sample.text!r}  image={sample.image.shape[1]}x{sample.image.shape[0]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
