"""Validate the three metrics against real data, before any model exists.

    python scripts/check_metrics.py --samples 300

This is the point of building the evaluation harness first. Each metric is run on
*real* handwriting in a configuration whose answer is known in advance, so that a
broken metric is caught now rather than being mistaken for a bad model later.

    FID(real, real)         must be near zero. Anything else means the feature
                            extraction or the Fréchet computation is wrong.
    FID(real, other real)   must be small but non-zero -- two disjoint halves of
                            the same dataset. This is the floor any generated set
                            will be measured against.
    CER on real images      the recogniser's own error rate. Every CER figure the
                            project ever reports is only meaningful beside this.
    Writer retrieval        real images against real writers must score far above
                            chance, or the embedding carries no style and the
                            metric cannot detect its absence in generated images.

Downloads about 1.5 GB of model weights on first run.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from nib.config import get_path, load_config
from nib.data.pack import PackReader
from nib.engine.metrics import cer as cer_mod
from nib.engine.metrics import references as ref_mod
from nib.engine.metrics.fid import InceptionFeatures, compute_fid
from nib.engine.metrics.writer import RETRIEVAL_FLOOR, WriterRetrieval


def _writer_embedder(cfg, device: str, fallback):
    """The trained style embedding if it exists, otherwise Inception.

    Measured on 94 writers the network never trained on:

        ImageNet Inception     3.7% top-1
        untrained, same shape  8.0%
        trained                66.9%   (90.1% top-5)

    The middle number is the interesting one: random weights on the right
    architecture already beat trained weights on the wrong one, which is what
    says the problem was photographic features applied to handwriting rather than
    a task that is inherently hard.

    Falling back rather than failing, because FID and CER are still worth running
    on a machine where the embedding has not been trained yet -- but the source is
    printed, so a 3.7% result is never mistaken for a 67% one.
    """
    from nib.engine import checkpoint as ckpt
    from nib.models.writer_embedder import TorchEmbedderAdapter, WriterEmbedder

    path = get_path(cfg, "checkpoints") / "writer_embedder.pt"
    if not path.is_file():
        return (lambda ims: fallback(list(ims))), "ImageNet Inception (untrained for this task)"

    embedder = WriterEmbedder()
    ckpt.load(path, models={"embedder": embedder})
    return TorchEmbedderAdapter(embedder, device=device), f"trained embedding from {path.name}"


def _real_lines(cfg, limit: int) -> list[tuple[np.ndarray, str]]:
    """Real CVL lines and their transcriptions, for the recogniser's own error rate.

    The reassembly itself lives in :mod:`nib.data.cvl_lines`; this only loads and
    normalises. The version replaced here did the reassembly inline and had two
    faults that both flattered nothing and hurt the recogniser:

    * it filtered nothing, so roughly one line in twelve carried a transcription
      missing a word whose ink was still in the image -- and TrOCR was charged
      with a deletion error for reading that word correctly;
    * it took the first N lines in filename order, which groups by writer, so the
      whole measurement came from one or two hands.

    Both are fixed below: the reader drops incomplete lines, and the sample is
    drawn at random across writers with the configured seed.
    """
    import cv2

    from nib.data.cvl_lines import scan_lines
    from nib.data.preprocessing import normalise_line

    root = get_path(cfg, "raw") / "cvl"
    if not root.is_dir():
        return []

    lines, report = scan_lines(root, charset_name=str(cfg.data.charset))
    print(report.summary())
    if not lines:
        return []

    height = int(cfg.data.image_height)
    chosen = random.Random(int(cfg.seed)).sample(lines, min(limit, len(lines)))

    out: list[tuple[np.ndarray, str]] = []
    unreadable = 0
    for line in chosen:
        image = cv2.imread(str(line.image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            unreadable += 1
            continue
        out.append((normalise_line(image, height), line.text))
    if unreadable:
        print(f"  {unreadable} line images could not be read and were skipped")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300, help="images per side")
    parser.add_argument("--pack", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-cer", action="store_true", help="skip the 1.4 GB download")
    parser.add_argument("--cer-lines", type=int, default=40)
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    pack_path = (
        Path(args.pack)
        if args.pack
        else (get_path(cfg, "processed") / f"cvl_words_{cfg.data.image_height}.lmdb")
    )

    with PackReader(pack_path) as pack:
        print(pack.summary())
        rng = random.Random(cfg.seed)
        indices = list(range(len(pack)))
        rng.shuffle(indices)

        need = args.samples * 2
        if len(indices) < need:
            print(f"\nneed {need} records, the pack has {len(indices)}")
            return 1

        first = [pack[i] for i in indices[: args.samples]]
        second = [pack[i] for i in indices[args.samples : need]]

    failures: list[str] = []

    # What this run measures, written out at the end. The three numbers are the
    # baseline every generated result is read against, and they are unit-specific
    # -- a generated *line* compared to a floor measured on *words* is comparing
    # different things. Writing them to a file rather than leaving them in the
    # console means the next run reads them instead of someone retyping them.
    references: dict = {"pack": pack_path.name, "samples": args.samples}

    # ---- FID ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FID -- loading Inception")
    inception = InceptionFeatures(device=args.device)
    features_a = inception([r.image for r in first])
    features_b = inception([r.image for r in second])

    identical = compute_fid(features_a, features_a)
    print(f"\nFID(real, same real)   {identical.value:8.4f}   expected ~0")
    if identical.value > 0.5:
        failures.append(f"FID against itself is {identical.value:.3f}, not ~0")

    disjoint = compute_fid(features_a, features_b)
    references["fid_floor"] = disjoint.value
    print(f"FID(real, other real)  {disjoint.value:8.4f}   the floor for any model")
    if disjoint.value <= identical.value:
        failures.append("two disjoint real halves scored no worse than a set against itself")

    # ---- writer retrieval --------------------------------------------------
    print("\n" + "=" * 60)
    print("Writer retrieval -- real images against real writers")

    gallery = [r for r in first if r.writer_id in {x.writer_id for x in second}]
    queries = [r for r in second if r.writer_id in {x.writer_id for x in gallery}]
    if len(gallery) < 20 or len(queries) < 20:
        print("  too few overlapping writers in the sample; raise --samples")
    else:
        embedder, source = _writer_embedder(cfg, args.device, inception)
        print(f"  embedder: {source}")
        retrieval = WriterRetrieval(embedder)
        retrieval.fit([r.image for r in gallery], [r.writer_id for r in gallery])
        result = retrieval.evaluate([r.image for r in queries], [r.writer_id for r in queries])
        references["retrieval_real"] = result.top1
        references["retrieval_top5_real"] = result.topk
        references["retrieval_writers"] = len({r.writer_id for r in queries})
        print("\n" + result.summary())
        if result.top1 < RETRIEVAL_FLOOR:
            failures.append(
                f"retrieval scored {result.top1:.1%} top-1 on REAL handwriting "
                f"(chance {result.chance:.1%}). Beating chance is not the bar: a metric "
                f"that cannot tell real writers apart cannot detect a generator that "
                f"ignores its style input. ImageNet Inception features describe photo "
                f"texture, not handwriting -- this needs an embedding trained for the job."
            )

    # ---- CER ---------------------------------------------------------------
    if not args.skip_cer:
        print("\n" + "=" * 60)
        print("CER -- loading TrOCR (about 1.4 GB on first run)")
        from nib.engine.metrics.recogniser import TrOcrRecogniser

        recogniser = TrOcrRecogniser(device=args.device)

        # Lines, not words. Measured on real CVL handwriting: 53.3% CER on
        # isolated words against 11.1% on lines. TrOCR was trained on IAM lines,
        # and a lone word is out of distribution for it -- the tell is that it
        # hallucinates trailing punctuation, because it expects a sentence.
        pairs = _real_lines(cfg, limit=args.cer_lines)
        if not pairs:
            print("  no CVL line images found; skipping")
        else:
            result = cer_mod.evaluate(
                recogniser,
                generated_images=[image for image, _ in pairs],
                targets=[text for _, text in pairs],
            )
            references["cer_real"] = result.generated
            references["cer_lines"] = result.num_samples
            print(f"\nCER on REAL lines   {result.generated:.2%}   over {result.num_samples}")
            print("  the recogniser's own error rate, and the baseline every generated")
            print("  CER must be reported against. Part of it is punctuation TrOCR adds")
            print("  that CVL's word-level ground truth does not contain.")
            print("  Supersedes the 12.33% measured in phase 1, which was taken over")
            print("  40 lines of one writer with no filtering, and so charged the")
            print("  recogniser for words the ground truth had lost.")
            if result.generated > 0.25:
                failures.append(
                    f"the recogniser scores {result.generated:.1%} on real lines, "
                    "which is too poor for it to judge anything"
                )

    reference_file = ref_mod.save(get_path(cfg, "references"), pack_path.name, references)

    print("\n" + "=" * 60)
    print(f"references written to {reference_file}")
    print("  scripts/evaluate_generator.py reads this file, so the baseline it")
    print("  reports against is the one actually measured here, on this pack.")
    absent = ref_mod.missing(references)
    if absent:
        print(f"  incomplete: {', '.join(absent)} not measured in this run")

    if failures:
        print("\nPROBLEMS:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall metric sanity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
