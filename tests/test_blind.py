"""Tests for the blind test's building blocks: cropping, sides, scoring."""

from __future__ import annotations

import numpy as np
import pytest

from nib.engine.metrics import blind


def test_a_line_is_cropped_to_its_ink_with_a_thin_margin():
    image = np.full((64, 300), 255, dtype=np.uint8)
    image[20:40, 50:250] = 0

    crop = blind.crop_to_ink(image, margin=4)

    assert crop.shape == (20 + 8, 200 + 8)
    assert (crop[:4] == 255).all() and (crop[4:-4, 4:-4] == 0).all()


def test_a_line_without_ink_is_left_alone():
    blank = np.full((64, 100), 255, dtype=np.uint8)

    assert blind.crop_to_ink(blank) is blank


def test_the_real_line_is_on_each_side_half_the_time():
    sides = blind.assign_sides(40, seed=1)

    assert sides.count("A") == sides.count("B") == 20
    assert sides == blind.assign_sides(40, seed=1)
    assert sides != blind.assign_sides(40, seed=2)


def test_a_judge_who_always_answers_a_scores_half():
    sides = blind.assign_sides(40, seed=0)
    key = {str(i): side for i, side in enumerate(sides)}
    answers = [{"judge": "x", "trial": i, "choice": "A"} for i in range(40)]

    result = blind.score(answers, key)

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["per_judge"] == {"x": pytest.approx(0.5)}
    assert result["passes"]


def test_a_judge_who_always_finds_the_real_line_fails_the_system():
    key = {"0": "A", "1": "B", "2": "A"}
    answers = [{"judge": "x", "trial": t, "choice": key[str(t)]} for t in range(3)]

    result = blind.score(answers, key)

    assert result["accuracy"] == 1.0 and not result["passes"]


def test_the_codes_judges_send_back_become_answers():
    text = "hello\nNIB1;Dana;12A 7B\n\nNIB1;Omri K;3A\n"

    answers = blind.parse_codes(text)

    assert answers == [
        {"judge": "Dana", "trial": 12, "choice": "A"},
        {"judge": "Dana", "trial": 7, "choice": "B"},
        {"judge": "Omri K", "trial": 3, "choice": "A"},
    ]
