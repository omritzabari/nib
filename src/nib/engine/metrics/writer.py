"""Writer retrieval: is it in the *right person's* hand?

The third metric, and the only one that checks the thing the project is actually
about. FID asks whether the output looks like handwriting in general; CER asks
whether it is readable. Neither notices a model that produces beautiful, legible
handwriting in a style belonging to nobody in particular -- which is precisely
the failure mode of a generator that has learned to ignore its style input.

The test: take a generated image, and ask which of the real writers it most
resembles. If the answer is the writer it was conditioned on, the style carried.

Implementation is nearest-neighbour retrieval over an embedding, not a trained
classifier, and that is deliberate. A classifier has a fixed set of output
classes, so it cannot be asked about a writer it never saw -- and *unseen writers
are the entire point*. Retrieval handles a new person by embedding a few of their
real samples and comparing distances, which is the same operation the product
itself performs.

CVL was built for exactly this task, which is the main reason it is our
evaluation set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

RETRIEVAL_FLOOR = 0.30
"""Top-1 this metric must reach on REAL handwriting before it is worth trusting.

Not a threshold over chance. Wired to ImageNet Inception it scored 3.7% against
0.9% chance -- four times chance, and useless. The metric exists to notice a
generator that ignores its style input, and it can only do that if it identifies
real writers reliably first. So the bar is what the job needs, not what beats a
coin toss.
"""


class Embedder(Protocol):
    """Anything that turns handwriting images into vectors."""

    def __call__(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


class RetrievalError(RuntimeError):
    pass


@dataclass
class RetrievalResult:
    top1: float
    topk: float
    k: int
    num_queries: int
    num_writers: int
    chance: float = 0.0
    per_writer: dict[str, float] = field(default_factory=dict)

    @property
    def lift_over_chance(self) -> float:
        """How far above guessing. With 94 writers, chance is about 1%, so a top-1
        of 40% is a strong result -- and reporting the raw 40% without the 1%
        beside it hides that."""
        return self.top1 - self.chance

    def summary(self) -> str:
        return "\n".join(
            [
                f"writer top-1     {self.top1:.1%}   (chance {self.chance:.1%})",
                f"writer top-{self.k:<4}    {self.topk:.1%}",
                f"queries          {self.num_queries} against {self.num_writers} writers",
            ]
        )


class WriterRetrieval:
    """A gallery of real handwriting, queryable by image.

    Each writer is represented by the mean of their sample embeddings. Averaging
    rather than keeping every sample is what makes the comparison about the hand
    and not about which particular word happened to be nearest.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.writer_ids: list[str] = []
        self.gallery: np.ndarray | None = None

    def fit(
        self,
        images: Sequence[np.ndarray],
        writer_ids: Sequence[str],
        batch_size: int = 64,
    ) -> WriterRetrieval:
        if len(images) != len(writer_ids):
            raise RetrievalError(f"{len(images)} images for {len(writer_ids)} writer ids")
        if not images:
            raise RetrievalError("cannot build a gallery from nothing")

        features = _embed(self.embedder, images, batch_size)
        by_writer: dict[str, list[np.ndarray]] = {}
        for feature, writer_id in zip(features, writer_ids, strict=True):
            by_writer.setdefault(writer_id, []).append(feature)

        self.writer_ids = sorted(by_writer)
        self.gallery = _normalise(
            np.stack([np.mean(by_writer[w], axis=0) for w in self.writer_ids])
        )
        return self

    def evaluate(
        self,
        images: Sequence[np.ndarray],
        writer_ids: Sequence[str],
        k: int = 5,
        batch_size: int = 64,
    ) -> RetrievalResult:
        """Score queries against the gallery.

        Queries whose true writer is not in the gallery are refused rather than
        counted as failures: they are unanswerable, and folding them in would
        depress the score for a reason unrelated to style.
        """
        if self.gallery is None:
            raise RetrievalError("fit() must be called before evaluate()")
        if len(images) != len(writer_ids):
            raise RetrievalError(f"{len(images)} images for {len(writer_ids)} writer ids")
        if not images:
            raise RetrievalError("nothing to evaluate")

        unknown = set(writer_ids) - set(self.writer_ids)
        if unknown:
            raise RetrievalError(
                f"{len(unknown)} query writers are not in the gallery: {sorted(unknown)[:5]}. "
                "A query whose writer is absent cannot be answered; build the gallery "
                "from the same writers, or drop those queries deliberately."
            )

        queries = _normalise(_embed(self.embedder, images, batch_size))
        similarity = queries @ self.gallery.T  # cosine, both sides unit length

        k = max(1, min(k, len(self.writer_ids)))
        ranking = np.argsort(-similarity, axis=1)
        index_of = {w: i for i, w in enumerate(self.writer_ids)}
        truth = np.array([index_of[w] for w in writer_ids])

        hits1 = ranking[:, 0] == truth
        hitsk = np.any(ranking[:, :k] == truth[:, None], axis=1)

        per_writer: dict[str, list[bool]] = {}
        for writer_id, hit in zip(writer_ids, hits1, strict=True):
            per_writer.setdefault(writer_id, []).append(bool(hit))

        return RetrievalResult(
            top1=float(hits1.mean()),
            topk=float(hitsk.mean()),
            k=k,
            num_queries=len(images),
            num_writers=len(self.writer_ids),
            chance=1.0 / len(self.writer_ids),
            per_writer={w: float(np.mean(v)) for w, v in per_writer.items()},
        )


def _embed(embedder: Embedder, images: Sequence[np.ndarray], batch_size: int) -> np.ndarray:
    chunks = [
        np.asarray(embedder(list(images[start : start + batch_size])), dtype=np.float64)
        for start in range(0, len(images), batch_size)
    ]
    features = np.concatenate(chunks, axis=0)
    if features.ndim != 2 or len(features) != len(images):
        raise RetrievalError(
            f"the embedder returned {features.shape} for {len(images)} images; expected (n, d)"
        )
    return features


def _normalise(features: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is cosine similarity.

    Without this, a writer whose embeddings happen to have larger magnitude wins
    every comparison regardless of style.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)
