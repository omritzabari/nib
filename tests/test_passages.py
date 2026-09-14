"""The dictated passages: what a writer copies out by hand to enrol.

A passage is designed rather than found, and these tests hold it to the design.
Every character the system can write appears on each page, so the enrolment
shows how this person forms a Q, a 7 and a semicolon instead of hoping they turn
up. Nothing the system cannot write appears, so a transcription never has to be
filtered. And no line outgrows what the generator writes in one go.

Page 1 is the style a writer enrols with; page 2 is written out too, and held
back, so the system's version of page 2 can be set beside the real one.
"""

from __future__ import annotations

import pytest

from nib.config import find_repo_root
from nib.data import charset

PASSAGES = find_repo_root() / "configs" / "passages"
PAGES = sorted(PASSAGES.glob("english_page*.txt"))

MAX_LINE = 44
"""Characters per line. Full lines on Amri's sample page held 39 to 45, and the
generator's token budget -- 5.5 tokens per character, capped at 384 -- only binds
past 69."""

MIN_LETTER = 3
"""Each lowercase letter at least this often, so a letter's shape is seen in more
than one word and more than one neighbour."""


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_there_is_an_enrolment_page_and_a_held_back_page():
    assert [path.name for path in PAGES] == ["english_page1.txt", "english_page2.txt"]


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.stem)
def test_every_character_the_system_writes_is_on_the_page_and_nothing_else(path):
    alphabet = set(charset.get("english").characters)
    text = "".join(_lines(path))

    assert alphabet - set(text) == set(), "missing from the page"
    assert set(text) - alphabet == set(), "not writable by the system"


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.stem)
def test_every_lowercase_letter_is_seen_several_times(path):
    text = "".join(_lines(path))

    thin = {c: text.count(c) for c in "abcdefghijklmnopqrstuvwxyz" if text.count(c) < MIN_LETTER}
    assert thin == {}


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.stem)
def test_no_line_is_too_long_to_write_or_to_generate(path):
    long = [
        (number, len(line))
        for number, line in enumerate(_lines(path), start=1)
        if len(line) > MAX_LINE
    ]

    assert long == []
