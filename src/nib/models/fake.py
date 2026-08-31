"""A stand-in generator, for testing the evaluation rather than a model.

Not a model and not pretending to be one. It draws the requested text in a
typeface and returns that, so every number it produces is meaningless -- and
every *shape* it produces is right.

The project already works this way one level down: the Colab smoke run trains a
dummy network on purpose, because that run is a test of the plumbing and not of
anything that learns. This is the same idea for the evaluation. Loading Emuru
costs a 2.9 GB download and generating three hundred lines costs the better part
of an hour on a T4, and the first attempt at that spent twelve minutes before
dying on a code path nobody had exercised. Thirty seconds with this first is
cheap insurance.

Two behaviours make it useful rather than merely fast:

**Deterministic.** The same text always produces the same image, so a pipeline
run is reproducible and a difference between two runs is a real difference.

**It can decline to write.** ``failure_rate`` makes it raise
:class:`~nib.models.generator.EmptyGeneration` for a share of requests, which is
the path that killed the first real evaluation. A path that is only exercised by
accident, in production, at request 72 of 300, is not tested.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import cv2
import numpy as np

from nib.models.generator import EmptyGeneration, GenerationRequest

PIXELS_PER_CHAR = 22
"""Roughly what a real CVL line runs to at 64px, so widths land in the same
range as the real thing and the metrics see plausible shapes."""


class FakeGenerator:
    """Draws the requested text in a typeface. Fills the interface, models nothing."""

    def __init__(
        self,
        output_height: int = 64,
        failure_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._output_height = output_height
        self.failure_rate = failure_rate
        self.seed = seed

    @property
    def name(self) -> str:
        return "fake"

    @property
    def output_height(self) -> int:
        return self._output_height

    def generate(self, requests: Sequence[GenerationRequest]) -> list[np.ndarray]:
        return [self._draw(request) for request in requests]

    def _draw(self, request: GenerationRequest) -> np.ndarray:
        # Hashed rather than random, so the same request fails on every run and a
        # failure can be reproduced instead of chased.
        digest = hashlib.sha256(f"{self.seed}:{request.text}".encode()).digest()
        roll = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        if roll < self.failure_rate:
            raise EmptyGeneration(
                f"fake generator declining {request.text[:30]!r}, to exercise the "
                "exclusion path deliberately rather than at request 72 of 300"
            )

        height = self._output_height
        width = max(height, len(request.text) * PIXELS_PER_CHAR * height // 64)
        canvas = np.full((height, width), 255, dtype=np.uint8)
        cv2.putText(
            canvas,
            request.text,
            (4, int(height * 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            height / 90.0,
            0,
            1,
            cv2.LINE_AA,
        )
        return canvas
