"""Does a short per-writer fine-tune make DiffBrush write like that writer -- and
more like them than Emuru does?

    python scripts/evaluate_finetune_diffbrush.py --writers 24 --device cuda \\
        --compare-with /content/drive/MyDrive/nib/results/finetune_w24_t16_s150_r8

Stage 3 of the DiffBrush plan, and the one that decides it. The same held-out
writers, train lines and targets as cell 7d -- chosen by the same function from the
same seed -- generated twice with DiffBrush: as released, and after a LoRA
fine-tune on the writer's train lines. Both conditions go through the quality
control of ``nib.models.candidates``. The adapter is reset before the next writer.

``--compare-with`` points at 7d's saved run. Its ``baseline`` images are Emuru as
released on exactly these targets, and they are scored here as a third condition
against the same reference -- which also checks the harness, since Emuru's figure
must come out as 7d printed it. The criterion, set before the run: the fine-tuned
DiffBrush beats released Emuru on identity, paired by writer, with the interval
above zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.engine.metrics import bootstrap
from nib.engine.metrics import cer as cer_mod
from nib.engine.metrics import hwd as hwd_mod
from nib.models import diffbrush_finetune as tune
from nib.models.generator import GenerationRequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_finetune import MIN_WRITERS_FOR_VERDICT, generate_all, split_writers


def load_comparison(directory: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """An earlier run's released-model images, by target key."""
    saved = json.loads((directory / "per_sample.json").read_text(encoding="utf-8"))
    wanted = set(keys)
    images = {}
    for index, record in enumerate(saved):
        if record["key"] in wanted:
            image = cv2.imread(
                str(directory / "baseline" / f"{index:03d}.png"), cv2.IMREAD_GRAYSCALE
            )
            if image is None:
                raise SystemExit(f"missing {directory / 'baseline' / f'{index:03d}.png'}")
            images[record["key"]] = image
    return images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writers", type=int, default=24)
    parser.add_argument("--train-lines", type=int, default=16)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--steps", type=int, default=tune.DEFAULT_CONFIG.steps)
    parser.add_argument("--rank", type=int, default=tune.DEFAULT_CONFIG.rank)
    parser.add_argument("--learning-rate", type=float, default=tune.DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--batch-size", type=int, default=tune.DEFAULT_CONFIG.batch_size)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--selector", default="microsoft/trocr-small-handwritten")
    parser.add_argument(
        "--scope",
        choices=tuple(tune.LORA_SCOPES),
        default=tune.DEFAULT_CONFIG.scope,
        help="which attention the adapter sits on: 'style' leaves the layers that "
        "tie the drawing to the glyphs alone, 'all' adapts them too, as T37 did.",
    )
    parser.add_argument(
        "--keep-wide",
        action="store_true",
        help="train on lines wider than the canvas, squeezed to fit, instead of leaving them out.",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="an earlier fine-tune run on the same writers -- 7d's -- whose released "
        "images are scored here as a third condition",
    )
    parser.add_argument("--device", default="cuda")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)
    seed = int(cfg.seed)
    config = tune.DiffBrushFinetuneConfig(
        rank=args.rank,
        alpha=args.rank,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        scope=args.scope,
        skip_wide=not args.keep_wide,
        seed=seed,
    )

    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{height}.lmdb")
    by_writer = pack.writers()
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    held_out = [w for w in split.writers["test"] if w in by_writer]
    eligible, plan = split_writers(
        by_writer, held_out, args.writers, args.train_lines, args.targets, seed
    )
    print(f"writers   {len(plan)} of {len(eligible)} held-out writers with enough lines")
    print(f"per writer {args.train_lines} train lines, {args.targets} targets, the rest reference")
    print(f"fine-tune {config}")

    ensure_dirs(cfg, "outputs")
    name = f"finetune_diffbrush_w{len(plan)}_t{args.train_lines}_s{args.steps}_r{args.rank}"
    if config.scope != tune.DEFAULT_CONFIG.scope or not config.skip_wide:
        name += f"_{config.scope}" + ("" if config.skip_wide else "_wide")
    out_dir = get_path(cfg, "outputs") / name
    for sub in ("baseline", "adapted"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    from nib.engine.metrics.recogniser import TrOcrRecogniser
    from nib.models.candidates import CandidateGenerator
    from nib.models.diffbrush import DiffBrushGenerator, locate
    from nib.models.finetune import reset_lora

    code_dir, checkpoint = locate(cfg)
    diffbrush = DiffBrushGenerator(
        code_dir, checkpoint, device=args.device, output_height=height, seed=seed
    )
    selector = TrOcrRecogniser(model_name=args.selector, device=args.device, max_new_tokens=64)
    generator = CandidateGenerator(diffbrush, selector, candidates=args.candidates)
    trainable = tune.attach_lora(diffbrush.unet, config, device=args.device)
    total = sum(p.numel() for p in diffbrush.unet.parameters())
    print(
        f"LoRA      {trainable / 1e6:.2f}M trainable of {total / 1e6:.0f}M ({trainable / total:.2%})"
    )

    records, training = [], {}
    started = time.perf_counter()
    for number, (writer, parts) in enumerate(plan.items(), start=1):
        train = [pack[k] for k in parts["train"]]
        targets = [pack[k] for k in parts["targets"]]
        requests = [
            GenerationRequest(
                text=target.text,
                style_images=[line.image for line in train],
                style_texts=[line.text for line in train],
            )
            for target in targets
        ]

        baseline = generate_all(generator, requests)
        report = tune.train_writer(
            diffbrush.unet,
            diffbrush.vae,
            [line.image for line in train],
            [line.text for line in train],
            diffbrush.glyphs,
            config,
            device=args.device,
        )
        adapted = generate_all(generator, requests)
        reset_lora(diffbrush.unet)

        training[writer] = {
            "seconds": report.seconds,
            "losses": report.losses,
            "squeezed": report.squeezed,
            "skipped": report.skipped,
        }
        for target, before, after in zip(targets, baseline, adapted, strict=True):
            records.append(
                {
                    "key": target.key,
                    "writer_id": writer,
                    "text": target.text,
                    "baseline": before,
                    "adapted": after,
                }
            )
        elapsed = (time.perf_counter() - started) / 60
        print(
            f"  {number:>3}/{len(plan)} writer {writer}: {report.summary()}, "
            f"{report.squeezed} squeezed, {report.skipped} left out   "
            f"[{elapsed:.0f} min, about {elapsed / number * (len(plan) - number):.0f} to go]",
            flush=True,
        )

    kept = [r for r in records if r["baseline"] is not None and r["adapted"] is not None]
    comparison = None
    if args.compare_with is not None:
        comparison = load_comparison(args.compare_with, [r["key"] for r in kept])
        kept = [r for r in kept if r["key"] in comparison]
    lost = len(records) - len(kept)
    if lost:
        print(f"\nexcluded  {lost} targets missing from a condition, from every condition")

    per_sample = []
    for index, record in enumerate(kept):
        for condition in ("baseline", "adapted"):
            cv2.imwrite(str(out_dir / condition / f"{index:03d}.png"), record[condition])
        per_sample.append({k: record[k] for k in ("key", "writer_id", "text")})
    (out_dir / "per_sample.json").write_text(
        json.dumps(per_sample, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "training.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    print(f"images    {out_dir}  (saved before any metric)")

    conditions = {
        "DiffBrush released": [r["baseline"] for r in kept],
        "DiffBrush fine-tuned": [r["adapted"] for r in kept],
    }
    if comparison is not None:
        conditions["Emuru released"] = [comparison[r["key"]] for r in kept]
    results = measure(args, pack, by_writer, plan, kept, conditions, training, generator, out_dir)
    results |= {
        "excluded": lost,
        "config": config.__dict__,
        "trainable_parameters": trainable,
        "compare_with": str(args.compare_with) if args.compare_with else None,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pack.close()
    print(f"\nresults   {out_dir / 'results.json'}")
    return 0


def measure(args, pack, by_writer, plan, kept, conditions, training, generator, out_dir):
    """HWD identity and CER for every condition against one shared reference, and
    the differences that decide the stage, paired by writer."""
    results: dict = {"selection": generator.selection.as_dict()}
    seconds = np.array([t["seconds"] for t in training.values()])
    results["train_seconds_per_writer"] = {
        "mean": float(seconds.mean()),
        "max": float(seconds.max()),
    }
    results["squeezed_train_lines"] = int(sum(t["squeezed"] for t in training.values()))
    results["skipped_train_lines"] = int(sum(t["skipped"] for t in training.values()))

    real = [pack[r["key"]].image for r in kept]
    ids = [r["writer_id"] for r in kept]
    texts = [r["text"] for r in kept]

    print("\n" + "=" * 62)
    print("HWD identity -- same writers, same targets, one reference")
    consumed = {k for parts in plan.values() for k in parts["train"] + parts["targets"]}
    reference = hwd_mod.select_reference(by_writer, consumed, plan.keys(), seed=0)
    extractor = hwd_mod.VggExtractor()
    means = extractor([pack[k].image for k in reference.keys], reference.writer_ids)
    real_d = hwd_mod.distance(extractor(real, ids), means)
    distances = {
        name: hwd_mod.distance(extractor(images, ids), means) for name, images in conditions.items()
    }
    real_gap = real_d.gap
    count = len(real_gap)

    def share(gap, indices):
        return float(gap[indices].mean() / real_gap[indices].mean())

    print(
        f"  reference  {len(reference.keys)} lines over {len(means)} writers (chance {1 / len(means):.1%})"
    )
    results["hwd"] = {"real": real_d.value, "writers": count}
    for name, d in distances.items():
        interval = bootstrap.bootstrap_statistic(lambda i, g=d.gap: share(g, i), count)
        print(
            f"  {name:<21} identity {interval.format(as_percent=True)}   HWD {d.value:.2f}   "
            f"own writer nearest {d.nearest_is_right.mean():.1%}"
        )
        results["hwd"][name] = {
            "identity": interval.value,
            "identity_ci": [interval.low, interval.high],
            "distance": d.value,
            "nearest": float(d.nearest_is_right.mean()),
        }
    print(
        f"  {'real':<21} HWD {real_d.value:.2f}   own writer nearest {real_d.nearest_is_right.mean():.1%}"
    )

    tuned = distances["DiffBrush fine-tuned"].gap
    pairs = [("fine-tuned minus released", distances["DiffBrush released"].gap)]
    if "Emuru released" in distances:
        pairs.append(("fine-tuned minus Emuru released", distances["Emuru released"].gap))
    results["differences"] = {}
    for label, other in pairs:
        diff = bootstrap.bootstrap_statistic(
            lambda i, o=other: share(tuned, i) - share(o, i), count
        )
        if count < MIN_WRITERS_FOR_VERDICT:
            verdict = f"{count} writers is too few to judge; this run only checks the plumbing"
        elif diff.low > 0:
            verdict = "higher, and the difference is real"
        elif diff.high < 0:
            verdict = "lower, and the difference is real"
        else:
            verdict = "no difference that survives resampling"
        print(f"  {label:<33} {diff.format(as_percent=True)}  -> {verdict}")
        results["differences"][label] = {
            "value": diff.value,
            "ci": [diff.low, diff.high],
            "verdict": verdict,
        }

    print("\n" + "=" * 62)
    print("CER -- measured by TrOCR-base; TrOCR-small chose between draws")
    from nib.engine.metrics.recogniser import TrOcrRecogniser

    judge = TrOcrRecogniser(device=args.device)
    results["cer"] = {}
    for name, images in [*conditions.items(), ("real", real)]:
        outcome = cer_mod.evaluate(judge, generated_images=images, targets=texts)
        interval = bootstrap.cer_interval(outcome.errors, outcome.lengths)
        print(f"  {name:<21} {interval.format(as_percent=True)}")
        results["cer"][name] = [interval.value, interval.low, interval.high]

    np.savez_compressed(
        out_dir / "analysis.npz",
        writers=np.array(real_d.writers),
        real_gap=real_gap,
        **{
            name.replace(" ", "_").replace("-", "_") + "_gap": d.gap
            for name, d in distances.items()
        },
    )

    print("\n" + "=" * 62)
    print("SUMMARY")
    for name in conditions:
        low, high = results["hwd"][name]["identity_ci"]
        print(
            f"  identity {name:<21} {results['hwd'][name]['identity']:.1%}  [{low:.1%}, {high:.1%}]"
        )
    for label, entry in results["differences"].items():
        low, high = entry["ci"]
        print(f"  {label:<33} {entry['value']:+.1%}  [{low:+.1%}, {high:+.1%}]  {entry['verdict']}")
    cer_line = ", ".join(f"{name} {values[0]:.1%}" for name, values in results["cer"].items())
    print(f"  CER       {cer_line}")
    print(
        f"  training  {seconds.mean():.0f}s a writer on average, {seconds.max():.0f}s at most; "
        f"{results['squeezed_train_lines']} train lines squeezed, "
        f"{results['skipped_train_lines']} left out for being wider than the canvas"
    )
    return results


if __name__ == "__main__":
    sys.exit(main())
