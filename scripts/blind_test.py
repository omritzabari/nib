"""Build the blind test from a finished run, and score the judges' answers.

    python scripts/blind_test.py build outputs/eval_emuru_lines_refs4_cand4_byhand_omission
    python scripts/blind_test.py score outputs/blind_<run> codes.txt

``build`` writes, for the page the judges use, ``trials/NNN.json`` -- three of the
writer's real lines, and the target text twice as A and B, one real and one
generated -- and ``index.json`` listing them; one file a trial, so a judge loads
only the trials they are given. And ``key.json``, which says where the real line
is and never leaves this machine. The generated line gets the writer's ink, as
the product does (see :mod:`nib.inference.calibrate`); both are cropped to their
ink (see :func:`nib.engine.metrics.blind.crop_to_ink`).

``score`` reads the codes the judges sent back (one a line, as the page shows them;
see :data:`nib.engine.metrics.blind.CODE_PREFIX`) and says whether nib works by
CLAUDE.md's bar.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2

from nib.config import get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.engine.metrics import blind
from nib.inference.calibrate import calibrate, measure_hand

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_run import saved_run

REFERENCES = 3


def _data_uri(image) -> str:
    ok, png = cv2.imencode(".png", blind.crop_to_ink(image))
    if not ok:
        raise ValueError("could not encode an image")
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def build(args, cfg, repo: Path) -> int:
    height = int(cfg.data.image_height)
    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{height}.lmdb")
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    held_out = [w for w in split.writers["test"] if w in pack.writers()]
    requests, truths, generated, _ = saved_run(
        args.run, pack, held_out, args.samples, args.style_refs, int(cfg.seed)
    )
    sides = blind.assign_sides(len(truths), int(cfg.seed))
    if args.repair is not None:
        from nib.models import repair

        network = repair.load(args.repair)
        generated = [repair.repair(network, image) for image in generated]

    trials, targets = [], {}
    for trial, (request, truth, image, side) in enumerate(
        zip(requests, truths, generated, sides, strict=True)
    ):
        toned = calibrate(image, measure_hand(request.style_images, height=height))
        real, fake = _data_uri(truth.image), _data_uri(toned)
        trials.append(
            {
                "trial": trial,
                "writer": truth.writer_id,
                "references": [_data_uri(line) for line in request.style_images[:REFERENCES]],
                "A": real if side == "A" else fake,
                "B": fake if side == "A" else real,
            }
        )
        targets[str(trial)] = truth.key
    pack.close()

    out = args.out or get_path(cfg, "outputs") / f"blind_{args.run.name}"
    (out / "trials").mkdir(parents=True, exist_ok=True)
    for entry in trials:
        path = out / "trials" / f"{entry['trial']:03d}.json"
        path.write_text(json.dumps(entry), encoding="utf-8")
    index = [{"trial": t["trial"], "writer": t["writer"]} for t in trials]
    (out / "index.json").write_text(json.dumps(index) + "\n", encoding="utf-8")
    key = {"run": args.run.name, "sides": dict(enumerate(sides)), "targets": targets}
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in (out / "trials").iterdir()) / len(trials) / 1e3
    print(f"trials    {len(trials)} from {args.run.name}, {size:.0f} KB each -> {out / 'trials'}")
    print(f"key       {out / 'key.json'}  (keep it here; the page never sees it)")
    return 0


def score(args) -> int:
    key = json.loads((args.test / "key.json").read_text(encoding="utf-8"))["sides"]
    answers = blind.parse_codes(args.answers.read_text(encoding="utf-8"))
    result = blind.score(answers, key)
    low, high = result["interval"]
    print(f"judgments {result['answers']} by {len(result['per_judge'])} judges")
    for judge, accuracy in result["per_judge"].items():
        print(f"  {judge:<20} picked the real line {accuracy:.0%}")
    print(
        f"real line picked  {result['accuracy']:.1%} [{low:.1%}, {high:.1%}]   (50% = cannot tell)"
    )
    verdict = "WORKS" if result["passes"] else "NOT YET"
    print(f"  -> {verdict}: the bar is at most {blind.PASS:.0%} (CLAUDE.md)")
    (args.test / "score.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build", help="trials and key from a finished run")
    make.add_argument("run", type=Path)
    make.add_argument("--samples", type=int, default=150, help="as the run was made")
    make.add_argument("--style-refs", type=int, default=4, help="as the run was made")
    make.add_argument("--out", type=Path, default=None)
    make.add_argument("--repair", type=Path, default=None, help="mend strokes with these weights")
    judge = commands.add_parser("score", help="the judges' answers against the key")
    judge.add_argument("test", type=Path, help="the directory build wrote")
    judge.add_argument("answers", type=Path, help="a text file of the judges' codes, one a line")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    if args.command == "score":
        return score(args)
    return build(args, load_config(repo / "configs" / "base.yaml", overrides=overrides), repo)


if __name__ == "__main__":
    sys.exit(main())
