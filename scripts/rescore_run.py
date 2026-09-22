"""Score a finished run's lines again -- changed after the fact -- without generating.

    python scripts/rescore_run.py outputs/eval_emuru_lines_refs4_cand4_byhand_omission \\
        --calibrate --tag toned --device cuda

Generating 150 lines in ``hand`` mode costs about two hours of GPU. Anything done to
the lines *after* the model wrote them -- fitting them to each writer's page is the
first such thing -- can be judged on lines already written, by every metric the
evaluation has, in minutes. That is what this does:

1. rebuild the run's requests exactly -- same held-out writers, same seed, same
   count and style lines -- and check them against the run's ``per_sample.json``;
2. read the run's saved lines and, with ``--calibrate``, fit each one's ink to its
   request's own style lines (:mod:`nib.inference.calibrate`);
3. save them as a run of their own and score them with ``evaluate_generator``'s
   metrics, unchanged.

The calibration sees only the style lines; every metric compares against the
target lines, which neither the model nor the calibration ever saw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from nib.config import ensure_dirs, get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.inference.calibrate import calibrate, measure_hand

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_generator as evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a finished evaluation's output directory")
    parser.add_argument("--samples", type=int, default=150, help="as the run was made")
    parser.add_argument("--style-refs", type=int, default=4, help="as the run was made")
    parser.add_argument("--calibrate", action="store_true", help="fit each line's ink to its hand")
    parser.add_argument("--tag", required=True, help="appended to the run's name for the new one")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cer-samples", type=int, default=0, help="as in evaluate_generator")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    ensure_dirs(cfg, "outputs")
    height = int(cfg.data.image_height)
    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{height}.lmdb")
    reference, provenance = evaluation.load_references(cfg, pack.path.name, len(pack))

    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    held_out = [w for w in split.writers["test"] if w in pack.writers()]
    requests, truths, consumed = evaluation.build_requests(
        pack, held_out, args.style_refs, args.samples, int(cfg.seed)
    )

    records = json.loads((args.run / "per_sample.json").read_text(encoding="utf-8"))
    by_key = {truth.key: index for index, truth in enumerate(truths)}
    missing = [record["key"] for record in records if record["key"] not in by_key]
    if missing:
        print(
            f"the run's samples do not come from these requests ({len(missing)} unknown keys, "
            f"e.g. {missing[:2]}). Pass the run's own --samples and --style-refs."
        )
        return 1
    order = [by_key[record["key"]] for record in records]
    requests = [requests[i] for i in order]
    truths = [truths[i] for i in order]

    generated = []
    for index, request in enumerate(requests):
        image = cv2.imread(str(args.run / "generated" / f"{index:03d}.png"), cv2.IMREAD_GRAYSCALE)
        if args.calibrate:
            image = calibrate(image, measure_hand(request.style_images, height=height))
        generated.append(image)

    out_dir = get_path(cfg, "outputs") / f"{args.run.name}_{args.tag}"
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    print(f"rescoring          {len(generated)} lines of {args.run.name}")
    print(f"calibrated         {'yes' if args.calibrate else 'no'}")
    print(f"into               {out_dir}")
    evaluation._save_generated(out_dir, truths=truths, generated=generated)

    results = evaluation._measure(
        cfg,
        generated,
        truths,
        held_out,
        pack,
        args.device,
        out_dir,
        reference,
        provenance,
        args.cer_samples,
        consumed,
    )
    results |= {"rescored_from": args.run.name, "calibrated": args.calibrate}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults            {out_dir / 'results.json'}")
    pack.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
