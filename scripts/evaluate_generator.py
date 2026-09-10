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
from nib.engine.metrics import bootstrap
from nib.engine.metrics import cer as cer_mod
from nib.engine.metrics.fid import InceptionFeatures, compute_fid
from nib.engine.metrics.writer import WriterRetrieval
from nib.models.generator import (
    EmptyGeneration,
    GenerationRequest,
    check_output,
    to_uint8,
)

GALLERY_DEPTH = 12
"""Real images per writer in the retrieval gallery, where the writer has that
many to spare. More is a better description of a hand and also more distractors,
so the figure only compares across runs that used the same one."""

GALLERY_MINIMUM = 6
"""Below this a writer gets no gallery entry at all, and their queries are
withheld from the score rather than counted as misses. A query whose writer is
absent from the gallery cannot match, so scoring it would measure how the samples
were drawn rather than how well the model copies a hand."""

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


def load_references(cfg, pack_name: str, records: int) -> tuple[dict, str]:
    """The baseline for this pack, and where it came from.

    The source string is printed with every result, and with every result in
    results.json. A number whose provenance is not stated beside it is one nobody
    can check.

    ``records`` is what makes the check real. A pack rebuilt under the same name
    -- which is what excluding the German passage did -- leaves a reference file
    that looks current and describes different data.
    """
    from nib.engine.metrics import references as ref_mod

    measured = ref_mod.load(get_path(cfg, "references"), pack_name)
    if measured is None:
        return PHASE1_WORD_REFERENCE, (
            f"NO measured references for {pack_name}. Falling back to phase-1 "
            "WORD-level numbers, which are not a valid baseline for lines. Run "
            f"scripts/check_metrics.py --pack .../{pack_name} first."
        )

    note = f"measured on {measured.get('pack', pack_name)}"
    if ref_mod.stale(measured, records):
        note = (
            f"STALE: measured on a {measured['pack_records']}-record {pack_name}, "
            f"this pack holds {records}. The pack was rebuilt and the references "
            "were not. Re-run scripts/check_metrics.py before trusting any "
            "comparison below."
        )
    elif "pack_records" not in measured:
        note += " (record count not recorded, so a rebuild cannot be detected)"

    absent = ref_mod.missing(measured)
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
    if name == "eruku":
        from nib.models.eruku import ErukuGenerator

        return ErukuGenerator(device=device, output_height=height)
    if name == "eruku-no-style-text":
        # The deployable case, measured rather than assumed: what the system
        # scores when nobody has transcribed the style page, which is the
        # situation a real user is always in.
        from nib.models.eruku import ErukuGenerator

        return ErukuGenerator(device=device, output_height=height, use_style_text=False)
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
        choices=("emuru", "eruku", "eruku-no-style-text", "fake"),
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
    parser.add_argument(
        "--cer-samples",
        type=int,
        default=0,
        help="how many samples to read for CER. 0 means all of them, which is "
        "the default because a headline number measured on a fifth of the data "
        "carries a sampling error nobody quantified.",
    )
    parser.add_argument(
        "--allow-stale-references",
        action="store_true",
        help="run even when the baseline was measured on a different version of "
        "this pack. The numbers will not mean what they appear to.",
    )
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    device = args.device or ("cuda" if _cuda() else "cpu")
    height = int(cfg.data.image_height)

    ensure_dirs(cfg, "outputs")
    out_dir = get_path(cfg, "outputs") / f"eval_{args.generator}_{args.unit}"
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    pack = PackReader(get_path(cfg, "processed") / f"cvl_{args.unit}_{height}.lmdb")
    print(f"pack               {pack.path.name}  ({pack.header.source}, {len(pack)} records)")

    # Checked here, before a 2.9 GB checkpoint download and an hour of GPU, and
    # not at the end beside the results. The two packs this project has built
    # carry the same filename and differ by 1,720 records, so nothing visible
    # distinguishes a stale baseline from a current one -- and a run scored
    # against the wrong one produces numbers that look entirely reasonable.
    reference, provenance = load_references(cfg, pack.path.name, len(pack))
    print(f"baseline           {provenance}")
    if provenance.startswith("STALE") and not args.allow_stale_references:
        pack.close()
        print(
            "\nRefusing to run. Re-measure with:\n"
            f"  python scripts/check_metrics.py --pack {pack.path} --samples 300\n"
            "or pass --allow-stale-references if you meant it."
        )
        return 1

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

    results = _measure(
        cfg,
        generated,
        truths,
        held_out,
        pack,
        device,
        out_dir,
        reference,
        provenance,
        args.cer_samples,
    )
    results.update(results_run)
    pack.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults            {out_dir / 'results.json'}")
    return 0


def _measure(
    cfg, generated, truths, held_out, pack, device, out_dir, reference, provenance, cer_samples
):
    real = [t.image for t in truths]
    results: dict = {"count": len(generated)}

    # Loaded and checked in main(), before the checkpoint download, so a stale
    # baseline costs seconds rather than an hour. Carried in rather than re-read.
    results["reference_source"] = provenance

    print("\n" + "=" * 62)
    print("FID -- does it look like handwriting at all")
    inception = InceptionFeatures(device=device)
    # Kept rather than discarded. Resampling 2048 floats per image is what turns
    # a bare number into one with a spread, and it costs a few megabytes against
    # the hour of GPU it would take to learn the same thing by running again.
    features_real = inception(real)
    features_generated = inception(generated)
    fid = compute_fid(features_real, features_generated)
    fid_ci = bootstrap.fid_interval(features_real, features_generated)
    results["fid"] = fid.value
    results["fid_ci"] = [fid_ci.low, fid_ci.high]
    print(f"  generated  {fid_ci.format()}")
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
    thin: list[str] = []
    by_writer = pack.writers()
    rng = random.Random(int(cfg.seed))
    for writer in held_out:
        keys = sorted(key for key in by_writer.get(writer, []) if key not in targets)
        if len(keys) < GALLERY_MINIMUM:
            thin.append(writer)
            continue
        for key in rng.sample(keys, min(GALLERY_DEPTH, len(keys))):
            gallery.append(pack[key].image)
            gids.append(writer)

    # A query whose writer is not in the gallery cannot possibly match, so
    # leaving it in the score would measure our sampling rather than the model.
    # Withheld and counted, never dropped in silence.
    galleried = set(gids)
    pairs = [
        (image, truth.writer_id)
        for image, truth in zip(generated, truths, strict=True)
        if truth.writer_id in galleried
    ]
    withheld = len(generated) - len(pairs)
    results["retrieval_scored"] = len(pairs)

    print(f"  gallery    {len(gallery)} images over {len(galleried)} writers")
    if thin:
        print(
            f"  withheld   {withheld} queries from {len(thin)} writers with fewer "
            f"than {GALLERY_MINIMUM} non-target samples, who cannot be in the gallery"
        )

    retrieval = WriterRetrieval(embedder).fit(gallery, gids)
    scored = retrieval.evaluate([image for image, _ in pairs], [w for _, w in pairs])
    results["retrieval_top1"] = scored.top1
    results["retrieval_top5"] = scored.topk
    top1_ci = bootstrap.rate_interval(scored.hits_top1)
    top5_ci = bootstrap.rate_interval(scored.hits_topk)
    results["retrieval_top1_ci"] = [top1_ci.low, top1_ci.high]
    print(f"  generated  {top1_ci.format(as_percent=True)} top-1")
    print(f"             {top5_ci.format(as_percent=True)} top-5")
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
    # Every sample by default. It used to be the first 64 of 298, which put a
    # headline number on a fifth of the data while FID used all of it -- and the
    # sampling error in that fifth was never quantified.
    subset = len(generated) if cer_samples <= 0 else min(cer_samples, len(generated))
    if subset < len(generated):
        print(f"  reading {subset} of {len(generated)} on request")
    scored_cer = cer_mod.evaluate(
        recogniser,
        generated_images=generated[:subset],
        targets=[t.text for t in truths[:subset]],
        real_images=real[:subset],
        real_targets=[t.text for t in truths[:subset]],
    )
    results["cer_generated"] = scored_cer.generated
    results["cer_real"] = scored_cer.real
    cer_ci = bootstrap.cer_interval(scored_cer.errors, scored_cer.lengths)
    real_cer_ci = bootstrap.cer_interval(scored_cer.real_errors, scored_cer.lengths)
    results["cer_generated_ci"] = [cer_ci.low, cer_ci.high]
    print("\n" + scored_cer.summary())
    print(f"\n  generated  {cer_ci.format(as_percent=True)}")
    print(f"  real       {real_cer_ci.format(as_percent=True)}")
    print("\n  NOTE: TrOCR reads isolated words badly (53% on real words against 11%")
    print("  on lines). Both numbers above share that handicap, so the *gap* is the")
    print("  meaningful figure, not either value on its own.")

    _save_analysis(
        out_dir,
        truths=truths,
        generated=generated,
        features_real=features_real,
        features_generated=features_generated,
        hits_top1=scored.hits_top1,
        hits_topk=scored.hits_topk,
        cer_errors=scored_cer.errors,
        cer_lengths=scored_cer.lengths,
        cer_real_errors=scored_cer.real_errors,
    )

    print("\n" + "=" * 62)
    print("SUMMARY -- generated vs real, with 95% intervals")
    print(f"  FID            {fid_ci.format():>28}   vs {reference['fid_floor']:.2f} for real")
    print(
        f"  writer top-1   {top1_ci.format(as_percent=True):>28}   "
        f"vs {reference['retrieval_real']:.1%} for real"
    )
    print(
        f"  CER            {cer_ci.format(as_percent=True):>28}   "
        f"vs {real_cer_ci.value:.1%} for real"
    )
    print(f"  CER gap        {(scored_cer.gap or 0):+8.1%}   generated minus real")
    print("\n  Two results whose intervals overlap cannot be told apart.")
    return results


def _save_analysis(out_dir, *, truths, generated, **arrays) -> None:
    """Write what the metrics were computed from, so they can be re-examined
    without a GPU.

    Two files. ``analysis.npz`` holds the numeric terms -- Inception features,
    per-query hits, per-sample edit distances -- which is what a confidence
    interval, an ablation or a second opinion needs. ``per_sample.json`` holds
    what each sample *was*: its key, its writer, its text and the width it came
    out at, so a number can always be traced back to the lines that produced it.

    A run that reports only its conclusions has to be repeated to be questioned.
    """
    np.savez_compressed(out_dir / "analysis.npz", **{k: np.asarray(v) for k, v in arrays.items()})

    records = [
        {
            "key": truth.key,
            "writer_id": truth.writer_id,
            "text": truth.text,
            "real_width": int(truth.image.shape[1]),
            "generated_width": int(image.shape[1]),
        }
        for truth, image in zip(truths, generated, strict=True)
    ]
    (out_dir / "per_sample.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nanalysis          {out_dir / 'analysis.npz'}  (re-examine without a GPU)")


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
