"""Tests for drawing several candidates and keeping a readable one.

No model and no recogniser: a scripted generator returns images whose single
pixel value names the draw, and a scripted recogniser reads back whatever the
test says that draw reads as.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.models.candidates import CandidateGenerator, overrun, style_order, underrun
from nib.models.emuru import Truncation, TruncationLog
from nib.models.generator import EmptyGeneration, GenerationRequest, GeneratorError

PIXELS_PER_CHARACTER = 28
"""What the scripted hand takes per character.

The width check compares a draw against what the style lines say this writer's
hand costs, so a scripted draw and the style lines it came from have to carry
coherent geometry or every test would trip it. Both sides are built from this
one number, which puts an ordinary scripted draw at a ratio of 1.0.
"""


def _line(width: int, value: int = 0) -> np.ndarray:
    image = np.full((64, width), 255, dtype=np.uint8)
    image[20:40, 5:15] = value
    return image


class ScriptedGenerator:
    """Each call returns the next scripted outcome: a reading, None for empty,
    or ("truncated", reading).

    ``widths`` overrides the drawn width for each call in turn, for the tests
    that need a draw too narrow or too wide for its text; by default a draw is
    as wide as the scripted hand would write its request.
    """

    def __init__(self, script, widths=None):
        self.script = list(script)
        self.widths = list(widths) if widths is not None else None
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
        if self.widths is not None:
            width = self.widths[len(self.calls) - 1]
        else:
            width = PIXELS_PER_CHARACTER * len(request.text)
        # The draw's number is written into the image, and the reader maps it back.
        image = _line(width, value=len(self.calls))
        READINGS[len(self.calls)] = reading
        self.truncations.generated += 1
        if truncated:
            self.truncations.events.append(Truncation(request.text, 300, 99))
        return [image]


READINGS: dict[int, str] = {}


class ScriptedReader:
    def read(self, images):
        return [READINGS[int(image[30, 10])] for image in images]


def _style_name(index: int) -> str:
    return f"style {index}"


def _style_text(index: int, width: int) -> str:
    """A style line's transcription: it names its line, and it is as long as the
    line is wide.

    A transcription that did not match its line's width would make the width
    check measure the fixture rather than the draw.
    """
    name = _style_name(index)
    return name.ljust(max(len(name), round(width / PIXELS_PER_CHARACTER)), "s")


def _style_used(base) -> list[str]:
    """Which style line each draw was handed, by name."""
    return [call.style_texts[0][: len(_style_name(0))] for call in base.calls]


def _request(widths=(800, 800, 800), text="hello world"):
    return GenerationRequest(
        text=text,
        style_images=[_line(w) for w in widths],
        style_texts=[_style_text(i, w) for i, w in enumerate(widths)],
    )


@pytest.fixture(autouse=True)
def _clear():
    READINGS.clear()


def test_style_lines_in_the_working_width_range_come_first():
    images = [_line(300), _line(1500), _line(820), _line(600)]

    # 820 is nearest the middle of 500-1100, then 600; outside it, 300 is nearer
    # the range than 1500.
    assert style_order(images) == [2, 3, 0, 1]


# ---------------------------------------------------------------------------
# Choosing the style line by what it shows
#
# The model sees one of the writer's lines per draw. Ordering by width alone
# ignores that a line showing the letters the target needs is better evidence of
# how this writer forms them.
# ---------------------------------------------------------------------------


def test_by_letters_the_line_showing_most_of_the_target_is_first():
    images = [_line(800), _line(800), _line(800)]
    texts = ["oooo oooo", "quick zephyr", "aeiou"]

    order = style_order(images, style_texts=texts, target="quiz zephyr", by="letters")

    assert order[0] == 1


def test_by_letters_a_line_of_the_wrong_width_still_loses():
    """Width was measured: lines under 500px read at 69% CER against 26-28% inside
    the range. A perfect letter match on a short line is still a bad draw."""
    images = [_line(300), _line(800)]
    texts = ["quiz zephyr", "nothing alike"]

    order = style_order(images, style_texts=texts, target="quiz zephyr", by="letters")

    assert order[0] == 1


def test_without_style_texts_it_falls_back_to_width():
    images = [_line(300), _line(820), _line(600)]

    assert style_order(images, target="anything", by="letters") == style_order(images)


def test_an_unknown_ordering_is_refused():
    with pytest.raises(GeneratorError, match="letters"):
        style_order([_line(800)], by="colour")


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
    used = _style_used(base)
    assert set(used[:2]) == {"style 1", "style 2"}
    assert used[2] == "style 0"


def test_draws_cycle_through_the_style_lines_when_there_are_more_draws_than_lines():
    base = ScriptedGenerator(["bad"] * 4)
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4, accept_cer=0.0)

    wrapper.generate([_request(widths=(800, 700))])

    used = _style_used(base)
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
# Writing beyond the text
#
# A draw can write the whole line and then keep going -- "about the t.t. 1/",
# "cafe: Te-Te" -- and still read under the CER threshold, because the extra
# characters are few against the line's length.
# ---------------------------------------------------------------------------


def test_writing_after_the_text_is_counted():
    assert overrun("hello world te Te", "hello world") == 4


def test_a_stray_mark_before_the_text_is_counted():
    assert overrun(". hello world", "hello world") == 1


def test_the_larger_end_is_what_counts():
    assert overrun("xx hello world yyy", "hello world") == 3


def test_the_space_a_reader_puts_before_punctuation_costs_nothing():
    assert overrun("his dream .", "his dream.") == 0


def test_a_misread_letter_inside_the_line_is_not_writing_beyond_it():
    assert overrun("hellu world", "hello world") == 0


# ---------------------------------------------------------------------------
# Writing *less* than the text. The mirror of the block above, and the failure
# that had nothing watching it: on Amri's page "warm at noon" came out "walm
# noon" and "Order #378 at Lior's Cafe" came out "Order 3 t Lior's Cafe". Each
# reads well under the CER threshold with an overrun of zero.
# ---------------------------------------------------------------------------


def test_a_dropped_word_is_counted():
    assert underrun("warm noon", "warm at noon") == 2


def test_the_longest_run_is_what_counts_not_the_total():
    """Three separate misses of one character are a reader's noise; one run of
    three is a word gone. The sum cannot tell them apart and CER already has it."""
    assert underrun("ello worl tday", "hello world today") == 1


def test_a_run_inside_the_line_is_found():
    assert underrun("Order 3 t Lior", "Order #378 at Lior") == 3


def test_a_missing_beginning_is_counted():
    """Spaces are removed first, so the run is the five letters, not six."""
    assert underrun("world", "hello world") == 5


def test_a_missing_end_is_counted():
    assert underrun("hello", "hello world") == 5


def test_a_misread_letter_is_not_a_missing_one():
    """The alignment that substitutes is preferred to the one that deletes, so
    ambiguity is charged to CER rather than counted here."""
    assert underrun("hellu world", "hello world") == 0


def test_writing_beyond_the_text_is_not_writing_less_of_it():
    assert underrun("hello world te Te", "hello world") == 0


def test_the_space_a_reader_drops_costs_nothing():
    assert underrun("helloworld", "hello world") == 0


def test_a_perfect_reading_has_none():
    assert underrun("hello world", "hello world") == 0


def test_an_empty_reading_is_the_whole_text():
    assert underrun("", "hello world") == 10


def test_a_draw_that_keeps_writing_after_the_text_is_redrawn():
    # 35% CER: readable by that measure alone.
    base = ScriptedGenerator(["hello world again te Te", "hello world again"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    (image,) = wrapper.generate([_request(text="hello world again")])

    assert len(base.calls) == 2
    assert int(image[30, 10]) == 2
    assert wrapper.selection.rejected_for_overrun == 1
    assert "1 read well but wrote beyond the text" in wrapper.selection.summary()


def test_a_single_extra_character_is_tolerated():
    """Readers add a full stop or a quote to real lines too."""
    base = ScriptedGenerator(["hello world.", "never used"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    wrapper.generate([_request()])

    assert len(base.calls) == 1
    assert wrapper.selection.rejected_for_overrun == 0


def test_when_every_draw_writes_beyond_the_text_the_best_read_is_kept_and_counted():
    # The first reads at 35% CER and writes on; the second stays inside the text
    # and reads at 59%. Neither is acceptable, and the better-read one is kept.
    base = ScriptedGenerator(["hello world again te Te", "zzzzz zzzzz again"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=2)

    (image,) = wrapper.generate([_request(text="hello world again")])

    assert int(image[30, 10]) == 1
    assert wrapper.selection.none_accepted == 1


# ---------------------------------------------------------------------------
# Writing less than the text, wired -- and off by default
#
# The failure is real: on Amri's page the generator wrote "warm noon" for "warm
# at noon". But TrOCR-small, the reader, drops short words from complete lines on
# its own, so by default a draw is not judged on what the reader left out. The
# check stays available for a reader that can be trusted with it.
# ---------------------------------------------------------------------------


def test_by_default_a_draw_is_not_rejected_for_what_the_reader_left_out():
    """Read end to end on the fake generator, whose lines are complete by
    construction, TrOCR-small read "not the rapid calculation" as "not rapid
    calculation". A default that rejected that would reject sound draws."""
    base = ScriptedGenerator(["not rapid calculation", "never used"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    wrapper.generate([_request(text="not the rapid calculation")])

    assert len(base.calls) == 1
    assert wrapper.selection.rejected_for_underrun == 0


def test_turned_on_a_draw_that_leaves_a_word_out_is_redrawn():
    # "hello world again" without "world": reads at 35%, well under the CER
    # threshold, with no overrun.
    base = ScriptedGenerator(["hello again", "hello world again"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4, accept_underrun=1)

    (image,) = wrapper.generate([_request(text="hello world again")])

    assert len(base.calls) == 2
    assert int(image[30, 10]) == 2
    assert wrapper.selection.rejected_for_underrun == 1
    assert "1 read well but left part of the text out" in wrapper.selection.summary()


def test_turned_on_one_dropped_letter_is_tolerated():
    """Readers drop a letter from real lines too; CER counts that already."""
    base = ScriptedGenerator(["hello wold", "never used"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4, accept_underrun=1)

    wrapper.generate([_request()])

    assert len(base.calls) == 1
    assert wrapper.selection.rejected_for_underrun == 0


# ---------------------------------------------------------------------------
# The width the text should take
#
# A line too narrow for its text has lost some of it, and one far too wide has
# repeated itself or run into a smear. Both are visible without a recogniser,
# from what the style lines say this writer's hand costs per character. Measured
# over three runs: outside [0.70, 1.40) the mean CER of a draw is 29%, 31% and
# 75% against 14%, 12% and 31% over all draws, and only 13-28% are rejected.
# ---------------------------------------------------------------------------


def test_a_draw_far_too_narrow_for_its_text_is_redrawn():
    # Half the width the hand would need: something was left out, whatever the
    # reader made of it.
    base = ScriptedGenerator(["hello world", "hello world"], widths=(150, 308))
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    (image,) = wrapper.generate([_request()])

    assert len(base.calls) == 2
    assert int(image[30, 10]) == 2
    assert wrapper.selection.rejected_for_width == 1
    assert "1 read well but was the wrong width" in wrapper.selection.summary()


def test_a_draw_far_too_wide_for_its_text_is_redrawn():
    base = ScriptedGenerator(["hello world", "hello world"], widths=(700, 308))
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    wrapper.generate([_request()])

    assert len(base.calls) == 2
    assert wrapper.selection.rejected_for_width == 1


def test_an_ordinary_width_passes():
    base = ScriptedGenerator(["hello world", "never used"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    wrapper.generate([_request()])

    assert len(base.calls) == 1
    assert wrapper.selection.rejected_for_width == 0


def test_without_style_transcriptions_the_width_is_not_judged():
    """DiffBrush needs no transcription of its style line, so there is nothing to
    predict a width from. The check reports nothing rather than guessing."""
    request = GenerationRequest(text="hello world", style_images=[_line(800)])
    base = ScriptedGenerator(["hello world"], widths=(150,))
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    wrapper.generate([request])

    assert len(base.calls) == 1
    assert wrapper.selection.rejected_for_width == 0


def test_the_reading_of_every_kept_draw_is_recorded():
    """Nothing ever stored what the selector read, which is why the thresholds
    here cannot be calibrated from any run already on disk."""
    base = ScriptedGenerator(["hello world"])
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=4)

    wrapper.generate([_request()])

    assert wrapper.selection.chosen_readings == ["hello world"]


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
            if int(image[30, 10]) == 0:  # a style line, not a scripted draw
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


def test_by_hand_a_draw_that_writes_beyond_the_text_is_never_kept_however_close():
    base = ScriptedGenerator(["hello world again te Te", "hello world again"])
    hand = HandEmbedder({1: [1.0, 0.0], 2: [0.0, 1.0]})
    wrapper = CandidateGenerator(base, ScriptedReader(), candidates=2, hand=hand)

    (image,) = wrapper.generate([_request(text="hello world again")])

    assert int(image[30, 10]) == 2
    assert wrapper.selection.rejected_for_overrun == 1


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
