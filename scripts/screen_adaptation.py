"""Screen a per-writer fine-tune before spending a GPU hour generating with it.

    python scripts/screen_adaptation.py --writers 12 --device cuda

Every Emuru fine-tune so far was judged only by generating with it: 7d moved
nothing, 7f and 7t each cost 18 identity points, and nothing cheaper was checked
first. This checks one thing a useful adapter cannot fail: after training on a
writer's page, does the model predict that writer's *other* lines better --
teacher-forced, with a line of their page in front and nothing noised, the
conditions of generation? Nothing is generated.

Two recipes, on 7t's writers, lines and steps, differing only in the noise:

    7t           0.5 on every teacher-forced slice, the line in front included
    clean front  the line in front clean, as at generation; 0.1 on the line being
                 learned, Emuru's own training noise

Decided before the run (PROGRESS.md, 2026-09-22), on the held-out loss after the
last step against before the first, relative, per writer, with a bootstrap interval:

1. The 7t recipe clearly lowers it (interval wholly below zero): the yardstick
   cannot see what cost 7t 18 points, so no cheap screen exists -> per-writer
   fine-tuning of Emuru is closed.
2. Otherwise the clean-front recipe lowers it by 3% or more, interval wholly below
   zero -> one generating run of that recipe (about 80 minutes) is justified.
3. Otherwise -> per-writer fine-tuning of Emuru is closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.engine.metrics import bootstrap
from nib.models import finetune
from nib.models.candidates import style_order

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_finetune import split_writers

SEVEN_T_WRITERS = 24
"""7t's plan is drawn for 24 writers and the first ones kept, so each writer's
train and target lines are exactly 7t's."""

RECIPES = {
    "7t": {"noise": 0.5},
    "clean front": {"noise": 0.1, "prefix_noise": 0.0},
}

GAIN_NEEDED = -0.03


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writers", type=int, default=12)
    parser.add_argument("--steps", type=int, default=finetune.DEFAULT_CONFIG.steps)
    parser.add_argument("--probe-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    seed = int(cfg.seed)
    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{int(cfg.data.image_height)}.lmdb")
    by_writer = pack.writers()
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    held_out = [w for w in split.writers["test"] if w in by_writer]
    _, plan = split_writers(by_writer, held_out, SEVEN_T_WRITERS, 16, 4, seed)
    writers = list(plan)[: args.writers]
    print(f"writers   {len(writers)} of 7t's {len(plan)}, its train and target lines")

    from nib.models.emuru import EmuruGenerator

    model = EmuruGenerator(device=args.device).model
    finetune.attach_lora(model, finetune.FinetuneConfig(), device=args.device)

    curves: dict[str, dict[str, dict[int, float]]] = {name: {} for name in RECIPES}
    seconds: dict[str, list[float]] = {name: [] for name in RECIPES}
    for name, recipe in RECIPES.items():
        config = finetune.FinetuneConfig(steps=args.steps, context="same", seed=seed, **recipe)
        for number, writer in enumerate(writers, start=1):
            train = [pack[k] for k in plan[writer]["train"]]
            front = train[style_order([line.image for line in train])[0]]
            targets = [pack[k] for k in plan[writer]["targets"]]
            pairs = [(front.image, front.text, t.image, t.text) for t in targets]
            curve = curves[name][writer] = {}

            def probe(step: int, pairs=pairs, curve=curve) -> None:
                if step % args.probe_every == 0 or step == args.steps:
                    curve[step] = float(np.mean(finetune.heldout_loss(model, pairs, args.device)))

            probe(0)
            report = finetune.train_writer(
                model,
                [line.image for line in train],
                [line.text for line in train],
                config,
                device=args.device,
                after_step=probe,
            )
            finetune.reset_lora(model)
            seconds[name].append(report.seconds)
            print(
                f"  {name:<12} {number:>2}/{len(writers)} writer {writer}: held-out loss "
                + "  ".join(f"{s}:{v:.4f}" for s, v in curve.items()),
                flush=True,
            )

    print("\n" + "=" * 62)
    print("HELD-OUT LOSS -- the writer's unseen lines, a page line in front, nothing noised")
    changes = {}
    for name in RECIPES:
        per_writer = list(curves[name].values())
        steps = sorted(per_writer[0])
        means = "  ".join(f"{s}:{np.mean([c[s] for c in per_writer]):.4f}" for s in steps)
        relative = np.array([(c[steps[-1]] - c[0]) / c[0] for c in per_writer])
        interval = bootstrap.bootstrap_statistic(
            lambda i, r=relative: float(r[i].mean()), len(relative)
        )
        changes[name] = interval
        print(f"  {name:<12} {means}")
        print(
            f"  {'':<12} change {interval.value:+.1%} [{interval.low:+.1%}, {interval.high:+.1%}]"
            f"   {np.mean(seconds[name]):.0f}s a writer"
        )

    seven_t, clean = changes["7t"], changes["clean front"]
    if seven_t.high < 0:
        verdict = (
            "CLOSE: the 7t recipe lowers the held-out loss, yet cost 18 identity points -- "
            "this yardstick cannot see identity. Per-writer fine-tuning of Emuru is closed."
        )
    elif clean.value <= GAIN_NEEDED and clean.high < 0:
        verdict = (
            "RUN: the clean-front recipe helps unseen lines -- one generating run is justified."
        )
    else:
        verdict = (
            "CLOSE: the clean-front recipe does not help unseen lines. "
            "Per-writer fine-tuning of Emuru is closed."
        )
    print(f"\n  -> {verdict}")

    ensure_dirs(cfg, "outputs")
    out_dir = get_path(cfg, "outputs") / f"screen_adaptation_w{len(writers)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "curves": curves,
        "seconds": seconds,
        "change": {n: [i.value, i.low, i.high] for n, i in changes.items()},
        "verdict": verdict,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pack.close()
    print(f"\nresults   {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
