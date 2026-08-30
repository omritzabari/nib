"""Tests for the generator interface.

No model here. The point of putting an interface in front of the generator is
that its contract can be checked without one -- and the contract is where the
quiet failures live. A generator that returns images in a different order, or
inverted, or one short, produces metrics that are wrong rather than errors that
are visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.models.generator import (
    GenerationRequest,
    Generator,
    GeneratorError,
    check_output,
    to_uint8,
)


def style(n=3, width=40, height=64):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (height, width), dtype=np.uint8) for _ in range(n)]


def request(text="handwriting", **kwargs):
    kwargs.setdefault("style_images", style())
    return GenerationRequest(text=text, **kwargs)


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------


def test_a_request_needs_something_to_write():
    with pytest.raises(GeneratorError, match="nothing to write"):
        GenerationRequest(text="", style_images=style())


def test_a_request_needs_style_samples():
    """Generating without them is a different task. This interface exists for
    few-shot style transfer, and an empty list would quietly produce generic
    handwriting that scores fine on FID and fails the point entirely."""
    with pytest.raises(GeneratorError, match="few-shot"):
        GenerationRequest(text="hello", style_images=[])


def test_style_texts_must_match_the_images_when_given():
    with pytest.raises(GeneratorError, match="2 style texts for 3"):
        GenerationRequest(text="hello", style_images=style(3), style_texts=["a", "b"])


def test_style_texts_are_optional():
    """Whether a model needs them is worth recording: one that requires
    transcribed references is harder to deploy, since a user uploading a photo
    has transcribed nothing."""
    assert request().style_texts is None


# ---------------------------------------------------------------------------
# validating what came back
# ---------------------------------------------------------------------------


def image(width=100, height=64, low=0, high=255):
    array = np.full((height, width), high, dtype=np.uint8)
    array[:, : width // 2] = low
    return array


def test_a_correct_batch_passes():
    requests = [request("a"), request("bb")]
    check_output([image(), image()], requests, expected_height=64)


def test_a_missing_image_is_caught():
    """The dangerous one. A generator that drops a failed request shifts every
    later pairing of image to text, and CER then scores the wrong pairs -- with
    no error anywhere."""
    with pytest.raises(GeneratorError, match="Order and count"):
        check_output([image()], [request("a"), request("b")])


def test_a_colour_image_is_caught():
    with pytest.raises(GeneratorError, match="grayscale"):
        check_output([np.zeros((64, 40, 3), np.uint8)], [request()])


def test_an_empty_image_is_caught():
    with pytest.raises(GeneratorError, match="empty image"):
        check_output([np.zeros((0, 40), np.uint8)], [request()])


def test_a_wrong_height_is_caught():
    with pytest.raises(GeneratorError, match="height 32, expected 64"):
        check_output([image(height=32)], [request()], expected_height=64)


def test_a_blank_image_is_caught():
    """A model that has failed often returns a uniform field rather than raising.
    Every metric would score it without complaint."""
    with pytest.raises(GeneratorError, match="single flat"):
        check_output([np.full((64, 40), 255, np.uint8)], [request()])


def test_the_failing_request_is_named():
    """An error that says which text failed is worth far more than one that says
    a batch failed, when the batch is three hundred long."""
    with pytest.raises(GeneratorError, match="handwriting"):
        check_output([np.zeros((64, 40, 3), np.uint8)], [request("handwriting")])


# ---------------------------------------------------------------------------
# output ranges
# ---------------------------------------------------------------------------


def test_uint8_passes_through_unchanged():
    array = np.array([[0, 128, 255]], dtype=np.uint8)
    assert np.array_equal(to_uint8(array), array)


def test_a_tanh_head_is_mapped_not_clipped():
    """A model with a tanh head emits [-1, 1]. Treating that as [0, 1] would clip
    every dark pixel to black -- the samples would look wrong for a reason that
    has nothing to do with the model."""
    assert to_uint8(np.array([[-1.0, 0.0, 1.0]], np.float32)).tolist() == [[0, 127, 255]]


def test_a_zero_to_one_head_is_scaled():
    assert to_uint8(np.array([[0.0, 0.5, 1.0]], np.float32)).tolist() == [[0, 127, 255]]


def test_floats_already_in_zero_to_255_are_not_scaled_twice():
    assert to_uint8(np.array([[0.0, 127.0, 255.0]], np.float32)).tolist() == [[0, 127, 255]]


def test_ink_stays_dark_through_conversion():
    """The orientation that matters. Inverting here would make every metric score
    a negative of the handwriting, and none of them would object."""
    dark_ink = np.array([[0.0, 0.9, 1.0]], np.float32)  # ink at 0, paper near 1
    out = to_uint8(dark_ink)
    assert out[0, 0] < out[0, -1], "ink and paper were swapped"


# ---------------------------------------------------------------------------
# the protocol
# ---------------------------------------------------------------------------


class Fake:
    name = "fake"
    output_height = 64

    def generate(self, requests):
        return [image(width=8 * len(r.text)) for r in requests]


def test_a_conforming_object_satisfies_the_protocol():
    assert isinstance(Fake(), Generator)


def test_a_fake_generator_round_trips_through_validation():
    requests = [request("a"), request("longer text")]
    images = Fake().generate(requests)
    check_output(images, requests, expected_height=64)
    assert images[1].shape[1] > images[0].shape[1], "width should follow the text"
