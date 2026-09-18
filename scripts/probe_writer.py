"""Write in one person's hand, from one photographed page of it.

    python scripts/probe_writer.py --photo data/raw/personal/passage_page1.jpg --device cuda

The first time the whole system runs on a real user rather than on CVL. The page
must be a copy of a dictated passage, so the text of every written line is
known without reading it.

1. The photo is normalised and split into lines, and the count must match the
   passage -- a page that splits into a different number of lines is refused
   rather than paired wrongly.
2. Lines listed in ``--skip`` are set aside: corrections, and crops the
   segmentation got wrong. What is left is split in two, spread across the page:
   the **page** the system learns the hand from, and **targets** -- real lines the
   system writes again without seeing them, for comparison.
3. The targets are generated, and so is every line of ``--text``, a passage the
   writer never wrote at all, laid out as a page.

Written to ``outputs/probe_<photo>/``:

    comparison.png   each target: the real line above, the system's below
    written.png      the new passage, in the writer's hand
    blind/           each target as a pair in random order, and key.json --
                     show the pairs to people who know the hand and ask which is real
    results.json     CER of generated and real targets, and the selection log
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from nib.config import ensure_dirs, get_path, load_config
from nib.data.preprocessing import normalise_line, normalise_page
from nib.data.segmentation import split_lines
from nib.engine.metrics import bootstrap
from nib.engine.metrics import cer as cer_mod
from nib.models.generator import EmptyGeneration, GenerationRequest


def read_page(photo: Path, texts: list[str], height: int) -> list[np.ndarray]:
    image = cv2.imread(str(photo))
    if image is None:
        raise SystemExit(f"cannot read {photo}")
    result = split_lines(normalise_page(image))
    if len(result.lines) != len(texts):
        raise SystemExit(
            f"{photo.name} split into {len(result.lines)} lines, the passage has {len(texts)}. "
            "Refusing to pair them: every line would carry the wrong text."
        )
    return [normalise_line(line.image, height) for line in result.lines]


def _interval(outcome) -> bootstrap.Interval:
    """CER with its spread, or the bare figure when one target leaves nothing to
    resample -- as a one-target check run does."""
    if len(outcome.errors) < 2:
        return bootstrap.Interval(outcome.generated, outcome.generated, outcome.generated, 0)
    return bootstrap.cer_interval(outcome.errors, outcome.lengths)


def choose(clean: list[int], targets: int) -> tuple[list[int], list[int]]:
    """Targets spread evenly over the page, the rest as the page to learn from."""
    if targets >= len(clean):
        raise SystemExit(f"{targets} targets leaves nothing to learn from {len(clean)} lines")
    step = len(clean) / targets
    picked = sorted({clean[min(len(clean) - 1, int(step * k + step / 2))] for k in range(targets)})
    return [n for n in clean if n not in picked], picked


def stack(rows: list[np.ndarray], gap: int, labels: list[str] | None = None) -> np.ndarray:
    width = max(row.shape[1] for row in rows) + 16
    parts = []
    for index, row in enumerate(rows):
        if labels:
            label = np.full((24, width), 255, np.uint8)
            cv2.putText(
                label, labels[index], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 90, 1, cv2.LINE_AA
            )
            parts.append(label)
        canvas = np.full((row.shape[0], width), 255, np.uint8)
        canvas[:, 8 : 8 + row.shape[1]] = row
        parts += [canvas, np.full((gap, width), 255, np.uint8)]
    return np.vstack(parts[:-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--passage", type=Path, default=Path("configs/passages/english_page1.txt"))
    parser.add_argument("--text", type=Path, default=Path("configs/passages/english_page2.txt"))
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated line numbers, from 1, to leave out: corrections and bad crops",
    )
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--keep", choices=("readable", "hand"), default="readable")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="a trained adapter to load onto the generator, from scripts/adapt_emuru.py",
    )
    parser.add_argument(
        "--style-by",
        choices=("width", "letters"),
        default="width",
        help="how each draw's style line is chosen from the page: by width, or by "
        "which line shows most of the characters this line needs.",
    )
    parser.add_argument("--selector", default="microsoft/trocr-small-handwritten")
    parser.add_argument(
        "--generator",
        choices=("emuru", "diffbrush"),
        default="emuru",
        help="the model that writes. Outputs of any but Emuru go to their own folder, "
        "so a second model never overwrites the first one's page.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)

    texts = (repo / args.passage).read_text(encoding="utf-8").splitlines()
    lines = read_page(args.photo, texts, height)
    skip = {int(n) for n in args.skip.split(",") if n.strip()}
    clean = [n for n in range(1, len(lines) + 1) if n not in skip]
    page, targets = choose(clean, args.targets)
    print(f"page      {len(lines)} lines; skipped {sorted(skip)}")
    print(f"learning  from lines {page}")
    print(f"targets   lines {targets}, written again without being shown")

    ensure_dirs(cfg, "outputs")
    suffix = "" if args.generator == "emuru" else f"_{args.generator}"
    if args.style_by != "width":
        suffix += f"_style{args.style_by}"
    if args.adapter is not None:
        suffix += f"_{args.adapter.stem}"
    out_dir = get_path(cfg, "outputs") / f"probe_{args.photo.stem}{suffix}"
    (out_dir / "blind").mkdir(parents=True, exist_ok=True)

    from nib.engine.metrics.recogniser import TrOcrRecogniser
    from nib.models.candidates import CandidateGenerator

    sys.path.insert(0, str(repo / "scripts"))
    from evaluate_generator import _embedder, load_generator

    hand = None
    if args.keep == "hand":
        hand, _ = _embedder(cfg, args.device)
    generator = CandidateGenerator(
        load_generator(args.generator, args.device, height, cfg=cfg, adapter=args.adapter),
        TrOcrRecogniser(model_name=args.selector, device=args.device, max_new_tokens=64),
        candidates=args.candidates,
        hand=hand,
        style_by=args.style_by,
    )
    style_images = [lines[n - 1] for n in page]
    style_texts = [texts[n - 1] for n in page]

    def write(text: str) -> np.ndarray | None:
        request = GenerationRequest(text=text, style_images=style_images, style_texts=style_texts)
        try:
            return generator.generate([request])[0]
        except EmptyGeneration:
            return None

    generated = {n: write(texts[n - 1]) for n in targets}
    print(f"targets   generated {sum(v is not None for v in generated.values())} of {len(targets)}")

    new_text = (repo / args.text).read_text(encoding="utf-8").splitlines()
    written = [write(text) for text in new_text]
    missing = [i + 1 for i, image in enumerate(written) if image is None]
    print(
        f"new text  {len(new_text) - len(missing)} of {len(new_text)} lines written; missing {missing}"
    )

    kept = [n for n in targets if generated[n] is not None]
    rows, labels = [], []
    for n in kept:
        rows += [lines[n - 1], generated[n]]
        labels += [f"line {n}, real", f"line {n}, generated"]
    cv2.imwrite(str(out_dir / "comparison.png"), stack(rows, gap=10, labels=labels))
    blank = np.full((height, 200), 255, np.uint8)
    cv2.imwrite(
        str(out_dir / "written.png"),
        stack([w if w is not None else blank for w in written], gap=20),
    )

    rng = random.Random(args.seed)
    key = {}
    for index, n in enumerate(kept, start=1):
        order = ["real", "generated"]
        rng.shuffle(order)
        pair = {"real": lines[n - 1], "generated": generated[n]}
        cv2.imwrite(
            str(out_dir / "blind" / f"pair_{index:02d}.png"),
            stack([pair[order[0]], pair[order[1]]], gap=30, labels=["A", "B"]),
        )
        key[f"pair_{index:02d}"] = {"line": n, "A": order[0], "B": order[1]}
    (out_dir / "blind" / "key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")

    judge = TrOcrRecogniser(device=args.device)
    target_texts = [texts[n - 1] for n in kept]
    real_cer = cer_mod.evaluate(
        judge, generated_images=[lines[n - 1] for n in kept], targets=target_texts
    )
    gen_cer = cer_mod.evaluate(
        judge, generated_images=[generated[n] for n in kept], targets=target_texts
    )
    real_ci = _interval(real_cer)
    gen_ci = _interval(gen_cer)
    print(
        f"\nCER on the targets   real {real_ci.format(as_percent=True)}   generated {gen_ci.format(as_percent=True)}"
    )
    print(generator.selection.summary())

    results = {
        "photo": args.photo.name,
        "page_lines": page,
        "targets": targets,
        "skipped": sorted(skip),
        "cer_real": [real_ci.value, real_ci.low, real_ci.high],
        "cer_generated": [gen_ci.value, gen_ci.low, gen_ci.high],
        "new_text_missing": missing,
        "selection": generator.selection.as_dict(),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\noutputs   {out_dir}")
    print("  comparison.png  real above, generated below, for each target")
    print("  written.png     the new passage in this hand")
    print("  blind/          pairs in random order, answers in key.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
