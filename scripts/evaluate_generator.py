"""Generate handwriting in real writers' hands, and score it against the references.

    python scripts/check_metrics.py --pack data/processed/cvl_lines_64.lmdb
    python scripts/evaluate_generator.py --generator emuru --samples 300

This produces the project's first real numbers. Everything before it built the
ruler; this is the first thing measured with it.

**Run check_metrics.py on the same pack first.** It measures what *real*
handwriting scores on that pack and writes the figures to
``references/references_<pack>.json``, which this script reads. Without that file
the only baseline available is the phase-1 one, measured on word crops -- and a
generated line held against a word-level FID floor is being compared to a
different distribution. The fallback still runs, and says so in every line of
output it touches.

**Held-out writers only.** Style references and target texts come from the test
side of the committed split -- 94 writers no model in this project has trained on.
Scoring on training writers would flatter every number and answer a question
nobody asked.

**Lines, not words.** Emuru generates lines natively, and fixing a word crop to a
common height destroys relative scale, which is part of how a hand looks. The
default unit is therefore ``lines``.

**Each writer's own text, and never the target's own image.** The style sample is
a real line by that writer; the text to generate is a *different* line by the
same writer, so a real image of exactly that text in exactly that hand exists to
compare against. The generator never sees it, and it is kept out of the retrieval
gallery too -- otherwise a match would partly be shared content rather than a
recognised hand.

The references are not targets to beat. They are what *real handwriting* scores:
the ceiling, and the distance from them is the result.
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
from nib.engine.metrics import cer as cer_mod
from nib.engine.metrics.fid import InceptionFeatures, compute_fid
from nib.engine.metrics.writer import WriterRetrieval
from nib.models.generator import (
    EmptyGeneration,
    GenerationRequest,
    check_output,
    to_uint8,
)

PHASE1_WORD_REFERENCE = {
    "fid_floor": 33.72,
    "cer_real": 0.1233,
    "retrieval_real": 0.669,
    "pack": "cvl_words_64.lmdb (phase 1, WORD crops)",
}
"""The phase-1 numbers, kept only as a last resort.

They were measured on word crops. A generated *line* compared against them is
being compared to a different distribution, so `scripts/check_metrics.py` must be
run on the same pack first -- it writes the real figures next to the outputs and
this script reads them. These constants exist so a run without that file still
produces something, loudly labelled.
"""


def load_references(cfg, pack_name: str) -> tuple[dict, str]:
    """The baseline for this pack, and where it came from.

    The source string is printed with every result. A number whose provenance is
    not stated beside it is one nobody can check.
    """
    from nib.engine.metrics import references as ref_mod

    measured = ref_mod.load(get_path(cfg, "references"), pack_name)
    if measured is None:
        return PHASE1_WORD_REFERENCE, (
            f"NO measured references for {pack_name}. Falling back to phase-1 "
            "WORD-level numbers, which are not a valid baseline for lines. Run "
            f"scripts/check_metrics.py --pack .../{pack_name} first."
        )
    absent = ref_mod.missing(measured)
    note = f"measured on {measured.get('pack', pack_name)}"
    if absent:
        note += f" -- incomplete, missing {', '.join(absent)}"
    return {**PHASE1_WORD_REFERENCE, **measured}, note


def build_requests(pack, writers, style_refs, count, seed):
    """One request per target sample: that writer's other samples as style.

    Returns the requests alongside the real image of each target, which is what
    the comparison needs -- and which the generator is never shown.
    """
    rng = random.Random(seed)
    by_writer = pack.writers()
    requests, truths = [], []

    eligible = [w for w in writers if len(by_writer.get(w, [])) >= style_refs + 2]
    if not eligible:
        raise RuntimeError(f"no held-out writer has {style_refs + 2} samples")

    while len(requests) < count:
        writer = rng.choice(eligible)
        keys = rng.sample(sorted(by_writer[writer]), style_refs + 1)
        target = pack[keys[0]]
        refs = [pack[k] for k in keys[1:]]

        requests.append(
            GenerationRequest(
                text=target.text,
                style_images=[r.image for r in refs],
                style_texts=[r.text for r in refs],
            )
        )
        truths.append(target)
    return requests, truths


def load_generator(name: str, device: str, height: int, failure_rate: float = 0.0):
    if name == "emuru":
        from nib.models.emuru import EmuruGenerator

        return EmuruGenerator(device=device, output_height=height)
    if name == "fake":
        # Not a model. It draws the target text in a typeface, so every number is
        # meaningless and every shape is right -- which is what a run of this is
        # for: proving the harness works before spending an hour of GPU on it.
        from nib.models.fake import FakeGenerator

        return FakeGenerator(output_height=height, failure_rate=failure_rate)
    raise SystemExit(f"unknown generator {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator",
        default="emuru",
        choices=("emuru", "fake"),
        help="fake draws the target text in a typeface: every number it gives is "
        "meaningless and every shape is right, which proves the harness before an "
        "hour of GPU is spent on it.",
    )
    parser.add_argument(
        "--fake-failure-rate",
        type=float,
        default=0.0,
        help="share of requests the fake generator declines, to exercise the "
        "exclusion path on purpose rather than at request 72 of 300.",
    )
    parser.add_argument(
        "--unit",
        choices=("lines", "words"),
        default="lines",
        help="which pack to draw from. Lines: Emuru generates lines natively, and "
        "fixing a word to a common height destroys the relative scale that is part "
        "of how a hand looks.",
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument(
        "--style-refs",
        type=int,
        default=1,
        help="style samples per request. Emuru uses the first and ignores the rest; "
        "the flag exists for generators that take several.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-images", type=int, default=32)
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    device = args.device or ("cuda" if _cuda() else "cpu")
    height = int(cfg.data.image_height)

    ensure_dirs(cfg, "outputs")
    out_dir = get_path(cfg, "outputs") / f"eval_{args.generator}_{args.unit}"
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    pack = PackReader(get_path(cfg, "processed") / f"cvl_{args.unit}_{height}.lmdb")
    print(f"pack               {pack.path.name}  ({pack.header.source})")
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    held_out = [w for w in split.writers["test"] if w in pack.writers()]
    print(f"held-out writers   {len(held_out)}  (never trained on by anything here)")

    requests, truths = build_requests(pack, held_out, args.style_refs, args.samples, int(cfg.seed))
    print(f"requests           {len(requests)}, {args.style_refs} style samples each")

    print(f"\nloading {args.generator} on {device} ...")
    generator = load_generator(args.generator, device, height, args.fake_failure_rate)
    print(f"  {generator.name}, output height {generator.output_height}px")

    print("\ngenerating")
    started = time.perf_counter()
    generated: list[np.ndarray] = []
    kept: list = []
    excluded: list[str] = []

    # One request at a time. Emuru's own generate() loops internally anyway, so
    # batching bought nothing but a coarser failure unit -- and a single request
    # the model declined to write took the whole run down with it at 72 of 300.
    for index, request in enumerate(requests, start=1):
        try:
            image = to_uint8(generator.generate([request])[0])
        except EmptyGeneration as failure:
            # Excluded in pairs. Dropping the image alone would shift every later
            # pairing of generated to real, and CER would then be scoring the
            # wrong text against the wrong picture.
            excluded.append(request.text)
            print(f"  excluded {request.text[:40]!r}: {failure}", flush=True)
            continue

        check_output([image], [request], expected_height=generator.output_height)
        generated.append(image)
        kept.append(truths[index - 1])

        if index % args.batch_size == 0 or index == len(requests):
            rate = index / (time.perf_counter() - started)
            eta = (len(requests) - index) / max(rate, 1e-9)
            print(
                f"  {index:>4} / {len(requests)}   {rate:5.2f}/s   "
                f"eta {eta / 60:.0f} min   kept {len(generated)}",
                flush=True,
            )

    truths = kept

    elapsed = time.perf_counter() - started
    print(
        f"\ngenerated {len(generated)} in {elapsed / 60:.1f} min ({len(generated) / elapsed:.2f}/s)"
    )

    # A truncated line is a real output with its ending cut off, so it is scored
    # rather than dropped -- and a CER read over silently truncated text would be
    # measuring the budget, not the model.
    results_run: dict = {"requested": len(requests), "excluded": len(excluded)}
    if excluded:
        print(
            f"\nexcluded        {len(excluded)} of {len(requests)} requests the model "
            "wrote nothing for,\n                together with their ground truth, so "
            "the pairing never shifts."
        )

    for attribute in ("truncations", "empties"):
        log = getattr(generator, attribute, None)
        if log is not None:
            print("\n" + log.summary())
    log = getattr(generator, "truncations", None)
    if log is not None:
        results_run |= {"truncated": len(log.events), "truncation_rate": log.rate}
    empties = getattr(generator, "empties", None)
    if empties is not None:
        results_run |= {"retried_after_empty": empties.retried}

    for i in range(min(args.save_images, len(generated))):
        pair = np.full(
            (height * 2 + 8, max(generated[i].shape[1], truths[i].image.shape[1])),
            255,
            dtype=np.uint8,
        )
        pair[:height, : truths[i].image.shape[1]] = truths[i].image
        pair[height + 8 :, : generated[i].shape[1]] = generated[i]
        cv2.imwrite(str(out_dir / "samples" / f"{i:03d}_{truths[i].text}.png"), pair)
    print(f"samples            {out_dir / 'samples'}  (real on top, generated below)")

    results = _measure(cfg, generated, truths, held_out, pack, device, out_dir)
    results.update(results_run)
    pack.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults            {out_dir / 'results.json'}")
    return 0


def _measure(cfg, generated, truths, held_out, pack, device, out_dir):
    real = [t.image for t in truths]
    results: dict = {"count": len(generated)}

    reference, provenance = load_references(cfg, pack.path.name)
    results["reference_source"] = provenance
    print("\n" + "=" * 62)
    print(f"baseline   {provenance}")

    print("\n" + "=" * 62)
    print("FID -- does it look like handwriting at all")
    inception = InceptionFeatures(device=device)
    fid = compute_fid(inception(real), inception(generated))
    results["fid"] = fid.value
    print(f"  generated  {fid.value:8.2f}")
    print(f"  reference  {reference['fid_floor']:8.2f}   two halves of real handwriting")
    print(f"  -> {fid.value / reference['fid_floor']:.1f}x the floor")

    print("\n" + "=" * 62)
    print("writer retrieval -- is it in the RIGHT hand")
    embedder, source = _embedder(cfg, device)
    print(f"  embedder: {source}")
    # The gallery must not contain the very samples that were generated from.
    # Their real images say the same words as the generated ones, so a match
    # would partly be the recogniser noticing shared content rather than the
    # embedding recognising a hand -- which is the thing being measured.
    targets = {truth.key for truth in truths}

    gallery, gids = [], []
    by_writer = pack.writers()
    rng = random.Random(int(cfg.seed))
    for writer in held_out:
        keys = sorted(key for key in by_writer.get(writer, []) if key not in targets)
        if len(keys) >= 12:
            for key in rng.sample(keys, 12):
                gallery.append(pack[key].image)
                gids.append(writer)

    retrieval = WriterRetrieval(embedder).fit(gallery, gids)
    scored = retrieval.evaluate(generated, [t.writer_id for t in truths])
    results["retrieval_top1"] = scored.top1
    results["retrieval_top5"] = scored.topk
    print(f"  generated  {scored.top1:7.1%} top-1   {scored.topk:.1%} top-5")
    print(f"  reference  {reference['retrieval_real']:7.1%}   real handwriting")
    print(f"  chance     {scored.chance:7.1%}")
    if scored.top1 < scored.chance * 3:
        print("  -> the style did NOT carry. The model is ignoring its style input.")
    elif scored.top1 > reference["retrieval_real"] * 0.5:
        print("  -> the style carried.")
    else:
        print("  -> the style partly carried.")

    print("\n" + "=" * 62)
    print("CER -- is it readable as the intended text")
    from nib.engine.metrics.recogniser import TrOcrRecogniser

    recogniser = TrOcrRecogniser(device=device)
    subset = min(64, len(generated))
    scored_cer = cer_mod.evaluate(
        recogniser,
        generated_images=generated[:subset],
        targets=[t.text for t in truths[:subset]],
        real_images=real[:subset],
        real_targets=[t.text for t in truths[:subset]],
    )
    results["cer_generated"] = scored_cer.generated
    results["cer_real"] = scored_cer.real
    print("\n" + scored_cer.summary())
    print("\n  NOTE: TrOCR reads isolated words badly (53% on real words against 11%")
    print("  on lines). Both numbers above share that handicap, so the *gap* is the")
    print("  meaningful figure, not either value on its own.")

    print("\n" + "=" * 62)
    print("SUMMARY -- generated vs real")
    print(f"  FID            {results['fid']:8.2f}   vs {reference['fid_floor']:.2f} for real")
    print(
        f"  writer top-1   {results['retrieval_top1']:8.1%}   vs {reference['retrieval_real']:.1%} for real"
    )
    print(f"  CER gap        {(scored_cer.gap or 0):+8.1%}   generated minus real")
    return results


def _embedder(cfg, device):
    from nib.engine import checkpoint as ckpt
    from nib.models.writer_embedder import TorchEmbedderAdapter, WriterEmbedder

    path = get_path(cfg, "checkpoints") / "writer_embedder.pt"
    if not path.is_file():
        raise SystemExit(
            f"no trained embedding at {path}. Retrieval would score 3.7% on real "
            "handwriting and could not tell a styled generator from an unstyled one. "
            "Train it first: scripts/train_writer_embedder.py"
        )
    model = WriterEmbedder()
    ckpt.load(path, models={"embedder": model})
    return TorchEmbedderAdapter(model, device=device), f"trained, {path.name}"


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
