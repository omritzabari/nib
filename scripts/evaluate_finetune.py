"""Does a short per-writer fine-tune make Emuru write more like that writer?

    python scripts/evaluate_finetune.py --writers 24 --device cuda

Measured the only way that settles it: the same writers, the same target lines,
generated twice. For each held-out writer the lines are split three ways --

    train      the writer's "page": the fine-tune learns from these, and both
               conditions draw their style lines from them
    targets    lines to generate, never shown to the model in either condition
    reference  the rest, which HWD measures both conditions against

-- and the targets are generated once with Emuru as released and once after a
LoRA fine-tune on the train lines, both through the quality control of
``nib.models.candidates``. The adapter is reset before the next writer.

A difference between two conditions over the same writers and the same texts is
tested by resampling writers with both conditions attached, which is far sharper
than comparing two intervals. Training time per writer is reported, because
Amri's condition for this approach is that it fits a reasonable time per user.

Nothing here generates on training writers: the split is the committed
writer-disjoint one, and every writer below is on its held-out side.
"""

from __future__ import annotations

import argparse
import json
import random
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
from nib.models import finetune
from nib.models.generator import EmptyGeneration, GenerationRequest

MIN_WRITERS_FOR_VERDICT = 10
"""Below this many writers the paired interval is printed but not judged."""

OTHER_LINES = 400
"""Lines by other writers to draw from in ``--context other``."""


def split_writers(by_writer, held_out, writers, train_lines, targets, seed):
    """Choose writers with enough lines, and split each one's lines three ways."""
    need = train_lines + targets + hwd_mod.REFERENCE_MINIMUM
    eligible = sorted(w for w in held_out if len(by_writer.get(w, ())) >= need)
    if len(eligible) < 2:
        raise SystemExit(f"only {len(eligible)} held-out writers have {need} lines")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(eligible, min(writers, len(eligible))))
    plan = {}
    for writer in chosen:
        keys = sorted(by_writer[writer])
        rng.shuffle(keys)
        plan[writer] = {
            "train": keys[:train_lines],
            "targets": keys[train_lines : train_lines + targets],
        }
    return eligible, plan


def _other_writers(pack, by_writer, split, seed, limit=OTHER_LINES):
    """Lines by writers on the training side of the split, never evaluated here,
    of widths that generated well (500-1100px), as a fixed sample.
    """
    from nib.models.candidates import STYLE_WIDTH_RANGE

    low, high = STYLE_WIDTH_RANGE
    keys = sorted(k for w in split.writers["train"] for k in by_writer.get(w, ()))
    random.Random(seed).shuffle(keys)
    chosen = []
    for key in keys:
        sample = pack[key]
        if low <= sample.image.shape[1] <= high:
            chosen.append((sample.image, sample.text))
        if len(chosen) == limit:
            break
    return chosen


def generate_all(generator, requests):
    """One image per request, None where no draw produced anything."""
    images = []
    for request in requests:
        try:
            images.append(generator.generate([request])[0])
        except EmptyGeneration:
            images.append(None)
    return images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writers", type=int, default=24)
    parser.add_argument("--train-lines", type=int, default=16)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--steps", type=int, default=finetune.DEFAULT_CONFIG.steps)
    parser.add_argument("--rank", type=int, default=finetune.DEFAULT_CONFIG.rank)
    parser.add_argument(
        "--learning-rate", type=float, default=finetune.DEFAULT_CONFIG.learning_rate
    )
    parser.add_argument("--batch-size", type=int, default=finetune.DEFAULT_CONFIG.batch_size)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument(
        "--context",
        choices=finetune.CONTEXTS,
        default=finetune.DEFAULT_CONFIG.context,
        help="own: train on the writer's lines alone, as T29 did. other: put a "
        "line by a writer from the training split in front of each, and count the "
        "loss on the writer's line only, so the hand cannot be copied from context. "
        "same: put another of the writer's own lines in front, loss on the second -- "
        "the task generation performs, from what a user uploads (cell 7t).",
    )
    parser.add_argument("--noise", type=float, default=finetune.DEFAULT_CONFIG.noise)
    parser.add_argument("--selector", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--device", default="cuda")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)
    seed = int(cfg.seed)
    config = finetune.FinetuneConfig(
        rank=args.rank,
        alpha=2 * args.rank,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        noise=args.noise,
        context=args.context,
        seed=seed,
    )

    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{height}.lmdb")
    by_writer = pack.writers()
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    held_out = [w for w in split.writers["test"] if w in by_writer]
    others = _other_writers(pack, by_writer, split, seed) if args.context == "other" else None
    if others is not None:
        print(f"others    {len(others)} lines by training-split writers, one before each line")
    eligible, plan = split_writers(
        by_writer, held_out, args.writers, args.train_lines, args.targets, seed
    )
    print(f"writers   {len(plan)} of {len(eligible)} held-out writers with enough lines")
    print(f"per writer {args.train_lines} train lines, {args.targets} targets, the rest reference")
    print(f"fine-tune {config}")

    ensure_dirs(cfg, "outputs")
    name = f"finetune_w{len(plan)}_t{args.train_lines}_s{args.steps}_r{args.rank}"
    if (
        config.context != finetune.DEFAULT_CONFIG.context
        or config.noise != finetune.DEFAULT_CONFIG.noise
    ):
        name += f"_{config.context}_n{config.noise:g}"
    out_dir = get_path(cfg, "outputs") / name
    for sub in ("baseline", "adapted"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    from nib.engine.metrics.recogniser import TrOcrRecogniser
    from nib.models.candidates import CandidateGenerator
    from nib.models.emuru import EmuruGenerator

    emuru = EmuruGenerator(device=args.device, output_height=height)
    selector = TrOcrRecogniser(model_name=args.selector, device=args.device, max_new_tokens=64)
    generator = CandidateGenerator(emuru, selector, candidates=args.candidates)
    trainable = finetune.attach_lora(emuru.model, config, device=args.device)
    total = sum(p.numel() for p in emuru.model.parameters())
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
        report = finetune.train_writer(
            emuru.model,
            [line.image for line in train],
            [line.text for line in train],
            config,
            device=args.device,
            others=others,
        )
        adapted = generate_all(generator, requests)
        finetune.reset_lora(emuru.model)

        training[writer] = {"seconds": report.seconds, "losses": report.losses}
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
            f"  {number:>3}/{len(plan)} writer {writer}: {report.summary()}   "
            f"[{elapsed:.0f} min, about {elapsed / number * (len(plan) - number):.0f} to go]",
            flush=True,
        )

    # A target is compared only where both conditions produced an image, so the
    # pairing between them never breaks.
    kept = [r for r in records if r["baseline"] is not None and r["adapted"] is not None]
    lost = len(records) - len(kept)
    if lost:
        print(f"\nexcluded  {lost} targets that one condition wrote nothing for, from both")

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

    results = _measure(args, pack, by_writer, plan, kept, training, generator, selector, out_dir)
    results |= {"excluded": lost, "config": config.__dict__, "trainable_parameters": trainable}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pack.close()
    print(f"\nresults   {out_dir / 'results.json'}")
    return 0


def _measure(args, pack, by_writer, plan, kept, training, generator, selector, out_dir):
    results: dict = {}
    real = [pack[r["key"]].image for r in kept]
    ids = [r["writer_id"] for r in kept]
    texts = [r["text"] for r in kept]
    baseline = [r["baseline"] for r in kept]
    adapted = [r["adapted"] for r in kept]

    seconds = np.array([t["seconds"] for t in training.values()])
    results["train_seconds_per_writer"] = {
        "mean": float(seconds.mean()),
        "max": float(seconds.max()),
    }
    results["selection"] = generator.selection.as_dict()

    print("\n" + "=" * 62)
    print("HWD identity -- same writers, same targets, with and without the fine-tune")
    consumed = {k for parts in plan.values() for k in parts["train"] + parts["targets"]}
    reference = hwd_mod.select_reference(by_writer, consumed, plan.keys(), seed=0)
    extractor = hwd_mod.VggExtractor()
    means = extractor([pack[k].image for k in reference.keys], reference.writer_ids)
    real_d = hwd_mod.distance(extractor(real, ids), means)
    base_d = hwd_mod.distance(extractor(baseline, ids), means)
    adapt_d = hwd_mod.distance(extractor(adapted, ids), means)
    assert real_d.writers == base_d.writers == adapt_d.writers

    real_gap, base_gap, adapt_gap = real_d.gap, base_d.gap, adapt_d.gap

    def share(gap, indices):
        return float(gap[indices].mean() / real_gap[indices].mean())

    count = len(real_gap)
    base_ci = bootstrap.bootstrap_statistic(lambda i: share(base_gap, i), count)
    adapt_ci = bootstrap.bootstrap_statistic(lambda i: share(adapt_gap, i), count)
    diff_ci = bootstrap.bootstrap_statistic(
        lambda i: share(adapt_gap, i) - share(base_gap, i), count
    )
    chance = 1 / len(means)
    print(
        f"  reference  {len(reference.keys)} lines over {len(means)} writers (chance {chance:.1%})"
    )
    print(
        f"  HWD        released {base_d.value:.2f}   fine-tuned {adapt_d.value:.2f}   real {real_d.value:.2f}"
    )
    print(f"  identity   released   {base_ci.format(as_percent=True)}")
    print(f"             fine-tuned {adapt_ci.format(as_percent=True)}")
    print(
        f"  difference {diff_ci.format(as_percent=True)}   fine-tuned minus released, paired by writer"
    )
    # Resampling a handful of writers gives an interval that looks tight and means
    # nothing: two writers produced [2.3%, 3.5%] on a one-step smoke run.
    if count < MIN_WRITERS_FOR_VERDICT:
        print(f"  -> {count} writers is too few to judge; this run only checks the plumbing")
    elif diff_ci.low > 0 or diff_ci.high < 0:
        print("  -> the difference is real")
    else:
        print("  -> no difference that survives resampling")
    print(
        f"  own writer nearest: released {base_d.nearest_is_right.mean():.1%}, "
        f"fine-tuned {adapt_d.nearest_is_right.mean():.1%}, real {real_d.nearest_is_right.mean():.1%}"
    )
    results["hwd"] = {
        "released": base_d.value,
        "fine_tuned": adapt_d.value,
        "real": real_d.value,
        "identity_released": base_ci.value,
        "identity_released_ci": [base_ci.low, base_ci.high],
        "identity_fine_tuned": adapt_ci.value,
        "identity_fine_tuned_ci": [adapt_ci.low, adapt_ci.high],
        "identity_difference": diff_ci.value,
        "identity_difference_ci": [diff_ci.low, diff_ci.high],
        "writers": count,
    }

    print("\n" + "=" * 62)
    print("CER -- measured by TrOCR-base; TrOCR-small chose between draws")
    from nib.engine.metrics.recogniser import TrOcrRecogniser

    judge = TrOcrRecogniser(device=args.device)
    scored = {}
    for label, images in (("released", baseline), ("fine-tuned", adapted), ("real", real)):
        outcome = cer_mod.evaluate(judge, generated_images=images, targets=texts)
        interval = bootstrap.cer_interval(outcome.errors, outcome.lengths)
        scored[label] = interval
        print(f"  {label:<11} {interval.format(as_percent=True)}")
    results["cer"] = {label: [i.value, i.low, i.high] for label, i in scored.items()}

    np.savez_compressed(
        out_dir / "analysis.npz",
        writers=np.array(real_d.writers),
        real_gap=real_gap,
        released_gap=base_gap,
        fine_tuned_gap=adapt_gap,
    )

    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  identity released   {base_ci.format(as_percent=True)}")
    print(f"  identity fine-tuned {adapt_ci.format(as_percent=True)}")
    print(f"  difference          {diff_ci.format(as_percent=True)}  paired by writer")
    print(
        f"  CER                 released {scored['released'].value:.1%}, "
        f"fine-tuned {scored['fine-tuned'].value:.1%}, real {scored['real'].value:.1%}"
    )
    print(
        f"  training            {seconds.mean():.0f}s a writer on average, {seconds.max():.0f}s at most"
    )
    return results


if __name__ == "__main__":
    sys.exit(main())
