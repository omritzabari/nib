"""How much of writer retrieval survives a degradation that keeps the hand?

    python scripts/calibrate_retrieval.py

Writer retrieval scores 85.8% on real lines and 22.1% on Emuru's output, and the
project has spent a day treating that gap as the model's failure. That reading
assumes the metric responds to *whose hand it is* and to nothing else, and the
assumption has never been tested.

So: take real lines -- unquestionably the right hand, unquestionably the right
writer -- and damage them in ways a generative model damages an image without
changing whose handwriting it is. Blur. Resample through a smaller size. Shift
the contrast. Add noise. Then measure retrieval again.

If mild damage takes 85.8% down to the twenties, the metric is substantially
measuring *image fidelity* rather than identity, and the 22.1% attributed to
Emuru is partly a property of our ruler. If retrieval barely moves, the metric
is sound and the gap is real.

Either answer is worth more than another hour of GPU spent trying to move a
number nobody has validated. It costs nothing: no generation, no model download
beyond the embedding already on disk.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from nib.config import get_path, load_config
from nib.data.pack import PackReader
from nib.data.split import WriterSplit
from nib.engine.metrics.bootstrap import rate_interval
from nib.engine.metrics.writer import WriterRetrieval

GALLERY_DEPTH = 12


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def resample(image: np.ndarray, factor: float) -> np.ndarray:
    """Down and back up. Every generative model does this through its decoder."""
    height, width = image.shape
    small = cv2.resize(
        image, (max(1, int(width * factor)), max(1, int(height * factor))), cv2.INTER_AREA
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def soften_contrast(image: np.ndarray, amount: float) -> np.ndarray:
    """Grey the ink without moving a stroke. The hand is untouched."""
    mid = 255.0 * amount
    return np.clip(image.astype(np.float32) * (1 - amount) + mid, 0, 255).astype(np.uint8)


def speckle(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


DAMAGE: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "untouched": lambda im: im,
    "blur 0.8px": lambda im: blur(im, 0.8),
    "blur 1.5px": lambda im: blur(im, 1.5),
    "resample 1/2": lambda im: resample(im, 0.5),
    "resample 1/4": lambda im: resample(im, 0.25),
    "contrast -30%": lambda im: soften_contrast(im, 0.3),
    "noise sigma 12": lambda im: speckle(im, 12.0),
    "blur+resample+noise": lambda im: speckle(resample(blur(im, 1.0), 0.5), 8.0),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args, overrides = parser.parse_known_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "base.yaml", overrides=overrides)
    height = int(cfg.data.image_height)

    pack = PackReader(get_path(cfg, "processed") / f"cvl_lines_{height}.lmdb")
    split = WriterSplit.load(repo / "configs" / "splits" / "cvl-writer-disjoint.json")
    by_writer = pack.writers()
    held_out = [w for w in split.writers["test"] if w in by_writer]

    # The same construction the evaluation uses, so the numbers are comparable:
    # a gallery of real lines, and queries from lines the gallery does not hold.
    rng = random.Random(int(cfg.seed))
    gallery, gallery_ids, queries, query_ids = [], [], [], []
    for writer in held_out:
        keys = sorted(by_writer[writer])
        if len(keys) < GALLERY_DEPTH + 2:
            continue
        chosen = rng.sample(keys, GALLERY_DEPTH + 2)
        for key in chosen[:GALLERY_DEPTH]:
            gallery.append(pack[key].image)
            gallery_ids.append(writer)
        for key in chosen[GALLERY_DEPTH:]:
            queries.append(pack[key].image)
            query_ids.append(writer)
    pack.close()

    order = list(range(len(queries)))
    rng.shuffle(order)
    order = order[: args.queries]
    queries = [queries[i] for i in order]
    query_ids = [query_ids[i] for i in order]

    print(f"gallery   {len(gallery)} real lines over {len(set(gallery_ids))} held-out writers")
    print(f"queries   {len(queries)} real lines, none of them in the gallery")
    print("\nEvery query below is a real line by the writer it claims. Anything the")
    print("damage costs is the metric reacting to image quality, not to identity.\n")

    embedder, source = _embedder(cfg, args.device)
    print(f"embedder  {source}\n")
    retrieval = WriterRetrieval(embedder).fit(gallery, gallery_ids)

    print(f"{'damage':<22} {'top-1':>22}   {'kept':>6}")
    baseline = None
    for label, damage in DAMAGE.items():
        scored = retrieval.evaluate([damage(image) for image in queries], query_ids)
        interval = rate_interval(scored.hits_top1)
        baseline = baseline or interval.value
        share = interval.value / baseline if baseline else 0.0
        print(f"  {label:<20} {interval.format(as_percent=True):>22}   {share:>5.0%}")

    print("\nA metric that loses most of its accuracy to blur is measuring")
    print("cleanliness as much as identity, and the figure it gives a generator")
    print("is not only about whose hand the generator copied.")
    return 0


def _embedder(cfg, device):
    from nib.engine import checkpoint as ckpt
    from nib.models.writer_embedder import TorchEmbedderAdapter, WriterEmbedder

    path = get_path(cfg, "checkpoints") / "writer_embedder.pt"
    if not path.is_file():
        raise SystemExit(f"no trained embedding at {path}")
    model = WriterEmbedder()
    ckpt.load(path, models={"embedder": model})
    return TorchEmbedderAdapter(model, device=device), f"trained, {path.name}"


if __name__ == "__main__":
    sys.exit(main())
