"""The interface every handwriting generator must satisfy.

The project brief made modularity a requirement from day one, and this is where
that requirement is paid. Everything downstream -- the layout engine, the metrics,
the API, the deception study -- talks to this interface and never to a particular
model. Swapping the generator is then a change in one file rather than a rewrite.

That matters more here than it usually would. The architecture survey found that
the model chosen at planning time had already been superseded twice, and the
field is still moving. Whatever is behind this interface today is unlikely to be
what is behind it at the end.

The interface itself is deliberately narrow::

    generate(style_images, texts) -> list of images

Style comes in as *images*, never as a writer id. A model conditioned on an
identifier can only reproduce writers it was trained on, and every writer that
matters is one it has never seen. Anything that cannot be conditioned on images
does not belong behind this interface at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class GeneratorError(RuntimeError):
    pass


class EmptyGeneration(GeneratorError):
    """The model produced no image at all for one request.

    Lives here rather than beside a particular model because it is an outcome any
    generator can have, and because the caller has to be able to tell it apart
    from a programming error: an evaluation over three hundred samples should
    survive one request the model declined to write, and must not survive a
    misconfigured one.

    Raised rather than answered with a blank image. A blank would pass through
    FID as a legitimate sample and pull the score toward whatever an empty canvas
    scores -- a lie the metric has no way to detect. The honest handling is to
    exclude the request *and its ground truth together*, so the pairing of
    generated to real never shifts, and to report the count.
    """


@dataclass(frozen=True)
class GenerationRequest:
    """One thing to write, and the hand to write it in."""

    text: str
    style_images: list[np.ndarray]
    """Samples of the target hand. Grayscale, ink dark on light paper."""

    style_texts: list[str] | None = None
    """What the style samples say, where the model wants to know.

    Some models need it and some do not, which is itself worth recording: a model
    that requires transcribed references is harder to deploy, because a user
    uploading a photo has not transcribed anything.
    """

    def __post_init__(self) -> None:
        if not self.text:
            raise GeneratorError("nothing to write")
        if not self.style_images:
            raise GeneratorError(
                "no style samples. Generating without them is a different task -- "
                "this interface exists for few-shot style transfer."
            )
        if self.style_texts is not None and len(self.style_texts) != len(self.style_images):
            raise GeneratorError(
                f"{len(self.style_texts)} style texts for {len(self.style_images)} images"
            )


@runtime_checkable
class Generator(Protocol):
    """Anything that writes text in a given hand."""

    @property
    def name(self) -> str:
        """Identifies the model in reports and experiment logs."""
        ...

    @property
    def output_height(self) -> int:
        """Pixel height of what it produces. Metrics need to match it."""
        ...

    def generate(self, requests: Sequence[GenerationRequest]) -> list[np.ndarray]:
        """Grayscale images, ink dark on light paper, one per request.

        Same length and order as the input, always. A generator that silently
        drops a failed request would misalign every downstream pairing of image
        to target text, and CER would then be measuring the wrong pairs.
        """
        ...


def check_output(
    images: Sequence[np.ndarray],
    requests: Sequence[GenerationRequest],
    expected_height: int | None = None,
) -> None:
    """Validate what a generator returned, before anything measures it.

    Cheap, and it catches the failures that would otherwise surface as a strange
    metric value rather than as an error: wrong count, wrong orientation, colour
    where grayscale was promised, or an image that is entirely one shade.
    """
    if len(images) != len(requests):
        raise GeneratorError(
            f"{len(images)} images for {len(requests)} requests. Order and count "
            "must be preserved, or every image is paired with the wrong text."
        )

    for index, (image, request) in enumerate(zip(images, requests, strict=True)):
        array = np.asarray(image)
        if array.ndim != 2:
            raise GeneratorError(
                f"request {index} ({request.text!r}): expected a grayscale image, "
                f"got shape {array.shape}"
            )
        if array.size == 0:
            raise GeneratorError(f"request {index} ({request.text!r}): empty image")
        if expected_height is not None and array.shape[0] != expected_height:
            raise GeneratorError(
                f"request {index}: height {array.shape[0]}, expected {expected_height}"
            )
        if float(array.max()) - float(array.min()) < 1e-6:
            raise GeneratorError(
                f"request {index} ({request.text!r}): the image is a single flat "
                "shade. The model produced nothing."
            )


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise a generator's output to uint8, ink dark on light paper.

    Models emit [0,1], [-1,1] or uint8 depending on their head, and getting this
    wrong inverts the image -- which every metric would happily score without
    complaint.
    """
    array = np.asarray(image)
    if array.dtype == np.uint8:  # checked before the cast, or it can never be true
        return array

    array = array.astype(np.float32)
    low, high = float(array.min()), float(array.max())
    if low < -0.01:  # tanh head
        array = (array + 1.0) / 2.0
    elif high > 1.01:  # already 0..255, stored as float
        array = array / 255.0
    return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
