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


# ---------------------------------------------------------------------------
# Emuru's token budget
#
# No model is loaded here either. token_budget is a module-level pure function
# precisely so the arithmetic that decides whether a line can finish is testable
# without a 3 GB download -- and that arithmetic is where the last bug lived.
# ---------------------------------------------------------------------------


def test_the_budget_grows_with_the_text():
    from nib.models.emuru import token_budget

    assert token_budget("a" * 20) < token_budget("a" * 60)


def test_an_average_real_line_gets_room_to_finish():
    """The regression this whole change exists for.

    A real line of the English pack at 64px averages 42 characters and 910
    pixels. The old flat budget of 96 tokens was 768 pixels, so the model ran out
    of canvas before the sentence ended and the truncation was recorded as the
    model failing to stop.
    """
    from nib.models.emuru import PIXELS_PER_TOKEN, token_budget

    assert token_budget("x" * 42) * PIXELS_PER_TOKEN > 910


def test_the_longest_real_line_still_fits_under_the_cap():
    """87 characters and 1762 pixels is the widest line in the pack."""
    from nib.models.emuru import MAX_TOKENS, PIXELS_PER_TOKEN, token_budget

    assert token_budget("x" * 87) * PIXELS_PER_TOKEN >= 1762
    assert token_budget("x" * 87) <= MAX_TOKENS


def test_the_budget_fits_almost_every_real_line():
    """Checked against the data the budget was derived from, not asserted.

    TOKENS_PER_CHAR is set from the 99th percentile of tokens per character, so
    at least 99% of real lines should fit the budget their length earns. That is
    the property the setting is supposed to buy, and it is the one that decides
    whether a line can finish -- a question no synthetic string can answer,
    because length and pixels per character are anticorrelated in real
    handwriting: short lines are written large.

    The first full evaluation truncated 10.7% of its output at 4.0 tokens per
    character, which is roughly what this test would have predicted.
    """
    from nib.config import find_repo_root
    from nib.data.pack import PackReader, is_complete
    from nib.models.emuru import PIXELS_PER_TOKEN, token_budget

    pack_path = find_repo_root() / "data" / "processed" / "cvl_lines_64.lmdb"
    if not is_complete(pack_path):
        pytest.skip(f"no line pack at {pack_path}")

    with PackReader(pack_path) as pack:
        step = max(1, len(pack) // 1000)
        lines = [pack[i] for i in range(0, len(pack), step)]

    fits = [token_budget(line.text) * PIXELS_PER_TOKEN >= line.image.shape[1] for line in lines]
    share = sum(fits) / len(fits)
    assert share >= 0.99, f"only {share:.1%} of real lines fit their budget"


def test_the_budget_is_clamped_at_both_ends():
    from nib.models.emuru import MAX_TOKENS, MIN_TOKENS, token_budget

    assert token_budget("a") == MIN_TOKENS
    assert token_budget("a" * 10_000) == MAX_TOKENS


def test_an_empty_log_reports_nothing_rather_than_dividing_by_zero():
    from nib.models.emuru import TruncationLog

    log = TruncationLog()
    assert log.rate == 0.0
    assert "nothing generated" in log.summary()


def test_a_clean_run_says_so_explicitly():
    """Silence would be indistinguishable from the counter never running."""
    from nib.models.emuru import TruncationLog

    log = TruncationLog(generated=10)
    assert log.rate == 0.0
    assert "0 of 10" in log.summary()


def test_truncations_are_counted_and_named():
    from nib.models.emuru import Truncation, TruncationLog

    log = TruncationLog(
        generated=4,
        events=[
            Truncation(text="a short one", width=760, budget=96),
            Truncation(text="a considerably longer target line", width=1520, budget=192),
        ],
    )

    assert log.rate == 0.5
    assert "2 of 4" in log.summary()
    assert "33 chars" in log.summary()


# ---------------------------------------------------------------------------
# writing nothing at all
#
# Emuru returns imgs[style_width : stop*8], and its stopping criterion scans the
# style prefix too. When it fires at or before the end of the style image the
# slice is empty -- which killed the first Colab evaluation at request 72 of 300.
# The style latents are sampled, so a re-draw is a real second chance; these
# tests cover the retry and the give-up, with a stand-in for the model.
# ---------------------------------------------------------------------------


class FakeEmuru:
    """Returns an empty image for the first `empties` calls, then a real one."""

    def __init__(self, empties: int):
        self.empties = empties
        self.calls = 0

    def generate(self, style_text, gen_text, style_img, max_new_tokens):
        from PIL import Image

        self.calls += 1
        if self.calls <= self.empties:
            return Image.new("L", (0, 64), 255)
        return Image.new("L", (240, 64), 128)


def emuru_stub(empties: int, retries: int = 3):
    """An EmuruGenerator with a fake model and no checkpoint download.

    Built without __init__ on purpose: the retry logic is the thing under test,
    and making it wait on a 3 GB download would mean it never got tested.
    """
    from nib.models.emuru import EmptyOutputLog, EmuruGenerator, TruncationLog

    generator = object.__new__(EmuruGenerator)
    generator.model = FakeEmuru(empties)
    generator._output_height = 64
    generator.max_new_tokens = None
    generator.tokens_per_char = 4.0
    generator.empty_retries = retries
    generator.truncations = TruncationLog()
    generator.empties = EmptyOutputLog()
    generator._as_tensor = lambda image: None  # the tensor conversion is not the subject
    return generator


def styled_request(text="a line of handwriting"):
    return GenerationRequest(text=text, style_images=style(1), style_texts=["reference"])


def test_a_request_that_writes_nothing_is_redrawn_until_it_does():
    generator = emuru_stub(empties=2)

    images = generator.generate([styled_request()])

    assert len(images) == 1
    assert images[0].shape == (64, 240)
    assert generator.model.calls == 3
    assert generator.empties.retried == 1
    assert generator.empties.extra_draws == 2


def test_a_request_that_never_writes_anything_raises_rather_than_returning_a_blank():
    """A blank would pass through FID as a legitimate sample and pull the score
    toward whatever an empty canvas scores -- undetectable from inside the metric."""
    from nib.models.generator import EmptyGeneration

    generator = emuru_stub(empties=99)

    with pytest.raises(EmptyGeneration, match="wrote nothing"):
        generator.generate([styled_request("unwritable")])

    assert generator.empties.failed == ["unwritable"]
    assert generator.model.calls == 4  # the first attempt plus three re-draws


def test_the_empty_failure_is_still_a_generator_error():
    """Callers that only know the interface must keep catching it."""
    from nib.models.generator import EmptyGeneration, GeneratorError

    assert issubclass(EmptyGeneration, GeneratorError)


def test_a_clean_run_says_so_rather_than_staying_silent():
    from nib.models.emuru import EmptyOutputLog

    assert "none" in EmptyOutputLog().summary()


def test_the_empty_log_names_what_was_excluded():
    from nib.models.emuru import EmptyOutputLog

    log = EmptyOutputLog(retried=2, extra_draws=3, failed=["a line nobody wrote"])
    summary = log.summary()

    assert "2 needed a retry" in summary
    assert "3 extra draws" in summary
    assert "a line nobody wrote" in summary


# ---------------------------------------------------------------------------
# the stand-in generator
# ---------------------------------------------------------------------------


def test_the_fake_generator_satisfies_the_interface():
    from nib.models.fake import FakeGenerator

    assert isinstance(FakeGenerator(), Generator)


def test_the_fake_generator_is_deterministic():
    """A failure must be reproducible rather than chased across runs."""
    from nib.models.fake import FakeGenerator

    first = FakeGenerator(failure_rate=0.5).generate([request("repeatable")])
    second = FakeGenerator(failure_rate=0.5).generate([request("repeatable")])

    assert np.array_equal(first[0], second[0])


def test_the_fake_generator_output_passes_validation():
    from nib.models.fake import FakeGenerator

    requests = [request("a line of text"), request("another")]
    images = FakeGenerator(output_height=64).generate(requests)

    check_output(images, requests, expected_height=64)


def test_the_fake_generator_declines_when_asked_to():
    """The exclusion path deserves to be exercised on purpose."""
    from nib.models.fake import FakeGenerator
    from nib.models.generator import EmptyGeneration

    generator = FakeGenerator(failure_rate=1.0)

    with pytest.raises(EmptyGeneration):
        generator.generate([request("anything")])


def test_the_fake_generator_writes_nothing_off_by_default():
    from nib.models.fake import FakeGenerator

    texts = [f"line number {i}" for i in range(50)]
    images = FakeGenerator().generate([request(t) for t in texts])

    assert len(images) == 50


# ---------------------------------------------------------------------------
# Eruku, and the failure that would look like success
# ---------------------------------------------------------------------------


class FakeEruku:
    """Stands in for the checkpoint. Records what it was called with."""

    def __init__(self, output: np.ndarray | None = None, empties: int = 0):
        self.output = output
        self.empties = empties
        self.calls: list[dict] = []

    def generate_handwriting(self, style_image, gen_text, style_text, cfg_scale, max_new_tokens):
        from PIL import Image

        self.calls.append({"gen_text": gen_text, "style_text": style_text, "cfg": cfg_scale})
        if len(self.calls) <= self.empties:
            return Image.new("L", (0, 64), 255)
        if self.output is not None:
            return Image.fromarray(self.output)
        return Image.fromarray(np.full((64, 200), 128, dtype=np.uint8))


def eruku_stub(model, use_style_text: bool = True):
    """An ErukuGenerator with a fake model and no 3 GB download."""
    from nib.models.emuru import EmptyOutputLog, TruncationLog
    from nib.models.eruku import DEFAULT_CFG_SCALE, ErukuGenerator

    generator = object.__new__(ErukuGenerator)
    generator.model = model
    generator._output_height = 64
    generator.max_new_tokens = None
    generator.tokens_per_char = 5.5
    generator.cfg_scale = DEFAULT_CFG_SCALE
    generator.empty_retries = 3
    generator.use_style_text = use_style_text
    generator.truncations = TruncationLog()
    generator.empties = EmptyOutputLog()
    generator._prefix_checked = False
    return generator


def test_eruku_output_that_begins_with_its_style_image_is_refused():
    """The failure that moves every metric the way success moves.

    An output carrying its style prefix contains a real crop of the writer's
    hand: writer retrieval rises, FID improves, and nothing looks wrong.
    """
    style_image = np.random.default_rng(0).integers(0, 255, (64, 90), dtype=np.uint8)
    leaked = np.concatenate([style_image, np.full((64, 150), 128, dtype=np.uint8)], axis=1)
    generator = eruku_stub(FakeEruku(output=leaked))

    request = GenerationRequest(text="a line", style_images=[style_image], style_texts=["ref"])
    with pytest.raises(GeneratorError, match="starts with its own style image"):
        generator.generate([request])


def test_eruku_output_that_does_not_repeat_the_style_passes():
    rng = np.random.default_rng(1)
    style_image = rng.integers(0, 255, (64, 90), dtype=np.uint8)
    fresh = rng.integers(0, 255, (64, 240), dtype=np.uint8)
    generator = eruku_stub(FakeEruku(output=fresh))

    request = GenerationRequest(text="a line", style_images=[style_image], style_texts=["ref"])
    images = generator.generate([request])

    assert images[0].shape == (64, 240)


def test_eruku_can_be_asked_to_ignore_the_style_transcription():
    """The deployable case. A user photographing a page has transcribed nothing,
    and Emuru could not be run this way at all."""
    model = FakeEruku()
    generator = eruku_stub(model, use_style_text=False)

    generator.generate([request("some text")])

    assert model.calls[0]["style_text"] == ""
    assert generator.name == "eruku-no-style-text"


def test_eruku_passes_the_transcription_when_it_has_one():
    model = FakeEruku()
    generator = eruku_stub(model, use_style_text=True)

    generator.generate(
        [GenerationRequest(text="target", style_images=style(1), style_texts=["the reference"])]
    )

    assert model.calls[0]["style_text"] == "the reference"
    assert generator.name == "eruku"


def test_eruku_retries_an_empty_output_like_emuru_does():
    """Kept identical on purpose: two generators whose failure handling differs
    cannot be compared on their failure counts."""
    model = FakeEruku(empties=2)
    generator = eruku_stub(model)

    images = generator.generate([request("a line")])

    assert len(images) == 1
    assert generator.empties.retried == 1
    assert len(model.calls) == 3


# ---------------------------------------------------------------------------
# several style lines as one reference
# ---------------------------------------------------------------------------


def test_one_style_image_passes_through_untouched():
    """The single-reference case must stay bit-identical, or a change measured
    against it is a change in the plumbing rather than in the references."""
    from nib.models.style import join_style

    only = style(1)[0]
    joined, text = join_style([only], ["a line"])

    assert np.array_equal(joined, only)
    assert text == "a line"


def test_joining_lays_them_side_by_side_with_a_gap():
    from nib.models.style import GAP_RATIO, join_style

    parts = [np.zeros((64, 100), np.uint8), np.zeros((64, 150), np.uint8)]
    joined, _ = join_style(parts)

    assert joined.shape == (64, 100 + 150 + round(64 * GAP_RATIO))


def test_the_gap_is_paper_and_not_ink():
    """Butted together, the last word of one line and the first of the next read
    as a single word and teach a letter join the writer never made."""
    from nib.models.style import PAPER, join_style

    joined, _ = join_style([np.zeros((64, 40), np.uint8)] * 2)

    assert joined.max() == PAPER, "the separator is darker than paper"


def test_the_text_is_joined_to_match_the_image():
    """Both models are told what the style says. Joining the images without the
    texts leaves the model reading one line while looking at four."""
    from nib.models.style import join_style

    _, text = join_style(style(3), ["first", "second", "third"])

    assert text == "first second third"


def test_joining_without_texts_returns_none():
    """Eruku does not require a transcription, so absent must stay absent rather
    than becoming an empty string that looks like one."""
    from nib.models.style import join_style

    assert join_style(style(2))[1] is None


def test_mismatched_counts_are_refused():
    from nib.models.style import join_style

    with pytest.raises(ValueError, match="style texts for"):
        join_style(style(3), ["only", "two"])


def test_joining_nothing_is_refused():
    from nib.models.style import join_style

    with pytest.raises(ValueError, match="no style images"):
        join_style([])


def test_differing_heights_are_reconciled():
    from nib.models.style import join_style

    joined, _ = join_style([np.zeros((64, 80), np.uint8), np.zeros((32, 80), np.uint8)])

    assert joined.shape[0] == 64
