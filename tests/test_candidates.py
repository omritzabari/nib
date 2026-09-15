"""Tests for drawing several candidates and keeping a readable one.

No model and no recogniser: a scripted generator returns images whose single
pixel value names the draw, and a scripted recogniser reads back whatever the
test says that draw reads as.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.models.candidates import CandidateGenerator, style_order
from nib.models.emuru import Truncation, TruncationLog
from nib.models.generator import EmptyGeneration, GenerationRequest, GeneratorError


def _line(width: int, value: int = 0) -> np.ndarray:
    image = np.full((64, width), 255, dtype=np.uint8)
    image[20:40, 5:15] = value
    return image


class ScriptedGenerator:
    """Each call returns the next scripted outcome: a reading, None for empty,
    or ("truncated", reading)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[GenerationRequest] = []
        self.truncations = TruncationLog()

    @property
    def name(self):
        return "scripted"

    @property
    def output_height(self):
        return 64

    def generate(self, requests):
        (request,) = requests
        self.calls.append(request)
        outcome = self.script[len(self.calls) - 1]
        if outcome is None:
            raise EmptyGeneration("scripted empty")
        truncated = isinstance(outcome, tuple)
        reading = outcome[1] if truncated else outcome
        # The draw's number is written into the image, and the reader maps it back.
        image = _line(300, value=len(self.calls))
        READINGS[len(self.calls)] = reading
        self.truncations.generated += 1
        if truncated:
            self.truncations.events.append(Truncation(request.text, 300, 99))
        return [image]


READINGS: dict[int, str] = {}


class ScriptedReader:
    def read(self, images):
        return [READINGS[int(image[30, 10])] for image in images]


def _request(widths=(800, 800, 800), text="hello world"):
    return GenerationRequest(
        text=text,
        style_images=[_line(w) for w in widths],
        style_texts=[f"style {i}" for i in range(len(widths))],
    )


@pytest.fixture(autouse=True)
def _clear():
    READINGS.clear()


def test_style_lines_in_the_working_width_range_come_first():
    images = [_line(300), _line(1500), _line(820), _line(600)]

    # 820 is nearest the middle of 500-1100, then 600; outside it, 300 is nearer
    # the range than 1500.
    assert style_order(images) == [2, 3, 0, 1]


def test_width_is_judged_at_the_reference_height():
    tall = np.full((128, 1600), 255, dtype=np.uint8)  # 800px at 64px high
    assert style_order([_line(300), tall]) == [1, 0]


def test_a_readable_first_draw_is_kept_without_drawing_again():
    base = ScriptedGenerator(["hello world", "never used"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    (image,) = wrapper.generate([_request()])

    assert len(base.calls) == 1
    assert int(image[30, 10]) == 1
    assert wrapper.selection.first_draw_accepted == 1
    assert wrapper.selection.draws == 1


def test_an_unreadable_draw_is_redrawn_and_the_readable_one_kept():
    base = ScriptedGenerator(["xxxxxxxxxxxxxx", "hello world"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    (image,) = wrapper.generate([_request()])

    assert len(base.calls) == 2
    assert int(image[30, 10]) == 2
    assert wrapper.selection.none_accepted == 0


def test_when_nothing_is_readable_the_best_of_all_draws_is_kept_and_counted():
    base = ScriptedGenerator(["zzzzzzzzzz", "hellozzzzzz", "zzzzzzzzzzzz"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=3, accept_cer=0.1)

    (image,) = wrapper.generate([_request()])

    assert len(base.calls) == 3
    assert int(image[30, 10]) == 2
    assert wrapper.selection.none_accepted == 1
    assert wrapper.selection.draws_per_request == 3


def test_every_draw_carries_exactly_one_style_line_with_its_own_text():
    """Two lines joined side by side broke Emuru's text; a draw never joins."""
    base = ScriptedGenerator(["bad", "bad", "bad"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=3, accept_cer=0.0)

    wrapper.generate([_request(widths=(300, 820, 600))])

    for call in base.calls:
        assert len(call.style_images) == 1
        assert len(call.style_texts) == 1
    used = [call.style_texts[0] for call in base.calls]
    assert set(used[:2]) == {"style 1", "style 2"}
    assert used[2] == "style 0"


def test_draws_cycle_through_the_style_lines_when_there_are_more_draws_than_lines():
    base = ScriptedGenerator(["bad"] * 4)
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4, accept_cer=0.0)

    wrapper.generate([_request(widths=(800, 700))])

    used = [call.style_texts[0] for call in base.calls]
    assert used == ["style 0", "style 1", "style 0", "style 1"]


def test_an_empty_draw_costs_a_draw_and_does_not_end_the_request():
    base = ScriptedGenerator([None, "hello world"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    (image,) = wrapper.generate([_request()])

    assert int(image[30, 10]) == 2
    assert wrapper.empties.retried == 1
    assert wrapper.empties.extra_draws == 1
    assert wrapper.empties.failed == []


def test_a_request_with_no_image_from_any_draw_is_an_empty_generation():
    base = ScriptedGenerator([None, None])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=2)

    with pytest.raises(EmptyGeneration):
        wrapper.generate([_request(text="lost line")])

    assert wrapper.empties.failed == ["lost line"]


def test_truncation_is_counted_over_kept_images_not_over_rejected_draws():
    base = ScriptedGenerator([("truncated", "zzzzzzzzzzzz"), "hello world"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    wrapper.generate([_request()])

    assert len(base.truncations.events) == 1
    assert wrapper.truncations.generated == 1
    assert wrapper.truncations.events == []


def test_a_kept_image_that_ran_to_its_budget_is_counted_as_truncated():
    base = ScriptedGenerator([("truncated", "hello world")])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    wrapper.generate([_request()])

    assert len(wrapper.truncations.events) == 1


# ---------------------------------------------------------------------------
# Keeping the draw closest to the hand
# ---------------------------------------------------------------------------


class HandEmbedder:
    """Style lines embed to the writer's direction; a draw embeds to whatever
    direction the test assigns its draw number."""

    def __init__(self, directions):
        self.directions = directions
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        out = []
        for image in images:
            if image.shape[1] != 300:  # a style line, not a scripted draw
                out.append([1.0, 0.0])
            else:
                out.append(self.directions[int(image[30, 10])])
        return np.array(out, dtype=float)


def test_by_hand_every_draw_is_made_and_the_closest_readable_one_kept():
    base = ScriptedGenerator(["hello world", "hello world", "hello world"])
    hand = HandEmbedder({1: [0.0, 1.0], 2: [1.0, 0.1], 3: [0.5, 0.5]})
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=3, hand=hand)

    (image,) = wrapper.generate([_request()])

    assert len(base.calls) == 3, "no early stop when choosing by hand"
    assert int(image[30, 10]) == 2
    assert wrapper.selection.moved_by_hand == 1
    assert wrapper.name == "scripted+best-of-3-by-hand"


def test_by_hand_an_unreadable_draw_is_never_kept_however_close():
    base = ScriptedGenerator(["zzzzzzzzzzzzzz", "hello world"])
    hand = HandEmbedder({1: [1.0, 0.0], 2: [0.0, 1.0]})
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=2, hand=hand)

    (image,) = wrapper.generate([_request()])

    assert int(image[30, 10]) == 2
    assert wrapper.selection.moved_by_hand == 0


def test_by_hand_with_nothing_readable_the_best_read_is_kept():
    base = ScriptedGenerator(["zzzzzzzzzz", "hellozzzzzz"])
    hand = HandEmbedder({1: [1.0, 0.0], 2: [0.0, 1.0]})
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=2, accept_cer=0.1, hand=hand)

    (image,) = wrapper.generate([_request()])

    assert int(image[30, 10]) == 2
    assert wrapper.selection.none_accepted == 1


def test_the_page_is_embedded_once_for_requests_sharing_its_lines():
    base = ScriptedGenerator(["hello world"] * 4)
    hand = HandEmbedder({1: [1.0, 0.0], 2: [1.0, 0.0], 3: [1.0, 0.0], 4: [1.0, 0.0]})
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=2, hand=hand)

    wrapper.generate([_request(text="hello world")])
    wrapper.generate([_request(text="hello world")])

    # One call for the page, then one per request for its readable draws.
    assert hand.calls == 3
    assert wrapper.selection.as_dict()["mode"] == "hand"


def test_zero_candidates_is_refused():
    with pytest.raises(GeneratorError):
        CandidateGenerator(ScriptedGenerator([]), ScriptedReader(), candidates=0)


def test_it_names_itself_after_the_model_underneath():
    wrapper = CandidateGenerator(ScriptedGenerator([]), ScriptedReader(), candidates=3)

    assert wrapper.name == "scripted+best-of-3"
    assert wrapper.output_height == 64
