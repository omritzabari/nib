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

from nib.config import get_path, load_config
from nib.data.pack import PackReader
from nib.engine.metrics import cer as cer_mod
from nib.engine.metrics.fid import InceptionFeatures, compute_fid
from nib.engine.metrics.writer import WriterRetrieval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300, help="images per side")
    parser.add_argument("--pack", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-cer", action="store_true", help="skip the 1.4 GB download")
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
        retrieval = WriterRetrieval(lambda ims: inception(list(ims)))
        retrieval.fit([r.image for r in gallery], [r.writer_id for r in gallery])
        result = retrieval.evaluate([r.image for r in queries], [r.writer_id for r in queries])
        print("\n" + result.summary())
        if result.top1 <= result.chance * 2:
            failures.append(
                f"retrieval scored {result.top1:.1%} against chance {result.chance:.1%} -- "
                "the embedding carries no writer identity, so this metric cannot "
                "detect a model that ignores its style input"
            )

    # ---- CER ---------------------------------------------------------------
    if not args.skip_cer:
        print("\n" + "=" * 60)
        print("CER -- loading TrOCR (about 1.4 GB on first run)")
        from nib.engine.metrics.recogniser import TrOcrRecogniser

        recogniser = TrOcrRecogniser(device=args.device)
        subset = first[: min(64, len(first))]
        result = cer_mod.evaluate(
            recogniser,
            generated_images=[r.image for r in subset],
            targets=[r.text for r in subset],
        )
        print(f"\nCER on REAL handwriting  {result.generated:.2%}   over {result.num_samples}")
        print("  this is the recogniser's own error rate -- the baseline every")
        print("  generated-image CER must be reported against")
        if result.generated > 0.5:
            failures.append(
                f"the recogniser scores {result.generated:.1%} on real handwriting, "
                "which is too poor for it to judge anything"
            )

    print("\n" + "=" * 60)
    if failures:
        print("PROBLEMS:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all metric sanity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
