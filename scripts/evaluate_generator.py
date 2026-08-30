"""Generate handwriting in real writers' hands, and score it against the references.

    python scripts/evaluate_generator.py --generator emuru --samples 300

This produces the project's first real numbers. Everything before it built the
ruler; this is the first thing measured with it.

**Held-out writers only.** Style references and target texts come from the test
side of the committed split -- 94 writers no model in this project has trained on.
Scoring on training writers would flatter every number and answer a question
nobody asked.

**Each writer's own words, and never the target's own image.** The style samples
are real words by that writer; the text to generate is a *different* word by the
same writer, so a real image of exactly that word in exactly that hand exists to
compare against. The generator never sees it.

The three numbers land beside the references measured in phase 1::

    FID              33.72   two disjoint halves of real handwriting
    CER              12.33%  the recogniser's own error rate on real lines
    writer top-1     66.9%   the embedding's accuracy on real writers

Those are not targets to beat -- they are what *real handwriting* scores. They are
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
from nib.models.generator import GenerationRequest, check_output, to_uint8

REFERENCE = {"fid_floor": 33.72, "cer_real": 0.1233, "retrieval_real": 0.669}


def build_requests(pack, writers, style_refs, count, seed):
    """One request per target word: that writer's other words as style.

    Returns the requests alongside the real image of each target word, which is
    what the comparison needs -- and which the generator is never shown.
    """
    rng = random.Random(seed)
    by_writer = pack.writers()
    requests, truths = [], []

    eligible = [w for w in writers if len(by_writer.get(w, [])) >= style_refs + 2]
    if not eligible:
        raise RuntimeError(f"no held-out writer has {style_refs + 2} words")

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


def load_generator(name: str, device: str, height: int):
    if name == "emuru":
        from nib.models.emuru import EmuruGenerator

        return EmuruGenerator(device=device, output_height=height)
    raise SystemExit(f"unknown generator {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="emuru")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--style-refs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-images", type=int, default=32)
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    device = args.device or ("cuda" if _cuda() else "cpu")
    height = int(cfg.data.image_height)

    ensure_dirs(cfg, "outputs")
    out_dir = get_path(cfg, "outputs") / f"eval_{args.generator}"
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    pack = PackReader(get_path(cfg, "processed") / f"cvl_words_{height}.lmdb")
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    held_out = [w for w in split.writers["test"] if w in pack.writers()]
    print(f"held-out writers   {len(held_out)}  (never trained on by anything here)")

    requests, truths = build_requests(pack, held_out, args.style_refs, args.samples, int(cfg.seed))
    print(f"requests           {len(requests)}, {args.style_refs} style samples each")

    print(f"\nloading {args.generator} on {device} ...")
    generator = load_generator(args.generator, device, height)
    print(f"  {generator.name}, output height {generator.output_height}px")

    print("\ngenerating")
    started = time.perf_counter()
    generated: list[np.ndarray] = []
    for start in range(0, len(requests), args.batch_size):
        chunk = requests[start : start + args.batch_size]
        images = [to_uint8(im) for im in generator.generate(chunk)]
        check_output(images, chunk, expected_height=generator.output_height)
        generated.extend(images)
        done = len(generated)
        rate = done / (time.perf_counter() - started)
        eta = (len(requests) - done) / max(rate, 1e-9)
        print(f"  {done:>4} / {len(requests)}   {rate:5.2f}/s   eta {eta / 60:.0f} min", flush=True)

    elapsed = time.perf_counter() - started
    print(
        f"\ngenerated {len(generated)} in {elapsed / 60:.1f} min ({len(generated) / elapsed:.2f}/s)"
    )

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
    pack.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults            {out_dir / 'results.json'}")
    return 0


def _measure(cfg, generated, truths, held_out, pack, device, out_dir):
    real = [t.image for t in truths]
    results: dict = {"count": len(generated)}

    print("\n" + "=" * 62)
    print("FID -- does it look like handwriting at all")
    inception = InceptionFeatures(device=device)
    fid = compute_fid(inception(real), inception(generated))
    results["fid"] = fid.value
    print(f"  generated  {fid.value:8.2f}")
    print(f"  reference  {REFERENCE['fid_floor']:8.2f}   two halves of real handwriting")
    print(f"  -> {fid.value / REFERENCE['fid_floor']:.1f}x the floor")

    print("\n" + "=" * 62)
    print("writer retrieval -- is it in the RIGHT hand")
    embedder, source = _embedder(cfg, device)
    print(f"  embedder: {source}")
    gallery, gids = [], []
    by_writer = pack.writers()
    rng = random.Random(int(cfg.seed))
    for writer in held_out:
        keys = sorted(by_writer.get(writer, []))
        if len(keys) >= 12:
            for key in rng.sample(keys, 12):
                gallery.append(pack[key].image)
                gids.append(writer)

    retrieval = WriterRetrieval(embedder).fit(gallery, gids)
    scored = retrieval.evaluate(generated, [t.writer_id for t in truths])
    results["retrieval_top1"] = scored.top1
    results["retrieval_top5"] = scored.topk
    print(f"  generated  {scored.top1:7.1%} top-1   {scored.topk:.1%} top-5")
    print(f"  reference  {REFERENCE['retrieval_real']:7.1%}   real handwriting")
    print(f"  chance     {scored.chance:7.1%}")
    if scored.top1 < scored.chance * 3:
        print("  -> the style did NOT carry. The model is ignoring its style input.")
    elif scored.top1 > REFERENCE["retrieval_real"] * 0.5:
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
    print(f"  FID            {results['fid']:8.2f}   vs {REFERENCE['fid_floor']:.2f} for real")
    print(
        f"  writer top-1   {results['retrieval_top1']:8.1%}   vs {REFERENCE['retrieval_real']:.1%} for real"
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
