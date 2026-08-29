"""Tests for the synthetic fixture.

Two things must hold or the fixture is worthless as a development stand-in:

1. It is *deterministic* -- identical output for the same seed, on any machine.
   A fixture that drifts cannot serve as a test baseline.
2. Writers are self-consistent and mutually distinguishable. Without that, any
   style-related test downstream passes or fails for the wrong reason.

The layout assertions guard the contract the parser (T4) will be written against.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np
import pytest

from nib.data import charset as cs
from nib.data import fixture


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture")
    summary = fixture.build(root=root, num_writers=4, words_per_writer=20, height=64, seed=0)
    return summary


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_pixels():
    style = fixture.style_for_writer(3, seed=0)
    a = fixture.render_word("handwriting", style, height=64, seed=0)
    b = fixture.render_word("handwriting", style, height=64, seed=0)
    assert np.array_equal(a, b)


def test_style_is_stable_and_independent_of_generation_order():
    """Writer 5 must be identical whether generated alone or as part of a batch."""
    alone = fixture.style_for_writer(5, seed=0)
    in_batch = [fixture.style_for_writer(i, seed=0) for i in range(10)][5]
    assert alone == in_batch


def test_different_seeds_give_different_styles():
    assert fixture.style_for_writer(0, seed=0) != fixture.style_for_writer(0, seed=1)


# --------------------------------------------------------------------------
# the two properties the pipeline actually depends on
# --------------------------------------------------------------------------


def test_writers_are_distinguishable_from_each_other():
    """Different writers rendering the same word must not produce the same image."""
    word = "handwriting"
    images = [
        fixture.render_word(word, fixture.style_for_writer(i, seed=0), height=64, seed=0)
        for i in range(6)
    ]
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            same_shape = images[i].shape == images[j].shape
            assert not (same_shape and np.array_equal(images[i], images[j])), (
                f"writers {i} and {j} render identically -- style has no effect"
            )


def test_a_writer_is_consistent_with_themselves():
    """Crude but real: one writer's ink density across different words should vary
    less than ink density across different writers on the same word."""
    words = ["the", "would", "London", "water", "think"]

    style = fixture.style_for_writer(2, seed=0)
    within = [(fixture.render_word(w, style, height=64, seed=0) < 128).mean() for w in words]

    across = [
        (
            fixture.render_word("the", fixture.style_for_writer(i, seed=0), height=64, seed=0) < 128
        ).mean()
        for i in range(12)
    ]

    assert np.std(within) < np.std(across), (
        "within-writer variation is not smaller than between-writer variation"
    )


# --------------------------------------------------------------------------
# image properties
# --------------------------------------------------------------------------


def test_image_height_is_exact_and_width_varies_with_text():
    style = fixture.style_for_writer(0, seed=0)
    short = fixture.render_word("the", style, height=64, seed=0)
    long = fixture.render_word("government", style, height=64, seed=0)
    assert short.shape[0] == long.shape[0] == 64
    assert long.shape[1] > short.shape[1], "width must follow the text"


def test_images_are_grayscale_uint8_with_ink_and_background():
    image = fixture.render_word("the", fixture.style_for_writer(0, seed=0), height=64)
    assert image.dtype == np.uint8
    assert image.ndim == 2
    assert image.min() < 128, "no dark ink was drawn"
    assert image.max() > 200, "no light background remains"


def test_requested_height_is_honoured():
    style = fixture.style_for_writer(0, seed=0)
    for height in [32, 64, 96]:
        assert fixture.render_word("test", style, height=height).shape[0] == height


def test_empty_word_raises():
    with pytest.raises(ValueError, match="empty word"):
        fixture.render_word("", fixture.style_for_writer(0, seed=0), height=64)


# --------------------------------------------------------------------------
# on-disk layout -- this is the contract the T4 parser is written against
# --------------------------------------------------------------------------


def test_directory_layout_mirrors_iam(built):
    root = built.root
    assert (root / "words").is_dir()
    assert (root / "xml").is_dir()
    assert (root / "ascii" / "words.txt").is_file()

    for form_id in built.forms:
        assert (root / "xml" / f"{form_id}.xml").is_file()
    pngs = list((root / "words").rglob("*.png"))
    assert len(pngs) == built.num_words


def test_word_images_are_nested_prefix_form_word(built):
    """IAM nests as words/<prefix>/<form-id>/<word-id>.png."""
    png = next((built.root / "words").rglob("*.png"))
    assert png.parent.parent.parent.name == "words"
    assert png.stem.startswith(png.parent.name)


def test_xml_carries_writer_id_and_transcriptions(built):
    tree = ET.parse(built.root / "xml" / f"{built.forms[0]}.xml")
    form = tree.getroot()
    assert form.tag == "form"
    assert form.get("writer-id") is not None

    words = form.findall(".//word")
    assert words, "no <word> elements"
    for word in words:
        assert word.get("id")
        assert word.get("text")
        assert word.find("cmp") is not None


def test_every_xml_word_has_a_matching_image(built):
    """The join the parser will perform. If ids and filenames disagree, every
    downstream sample is mislabelled."""
    for form_id in built.forms:
        tree = ET.parse(built.root / "xml" / f"{form_id}.xml")
        prefix = form_id.split("-")[0]
        for word in tree.getroot().findall(".//word"):
            path = built.root / "words" / prefix / form_id / f"{word.get('id')}.png"
            assert path.is_file(), f"missing image for {word.get('id')}"


def test_writer_ids_are_unique_across_forms(built):
    ids = set()
    for form_id in built.forms:
        writer = ET.parse(built.root / "xml" / f"{form_id}.xml").getroot().get("writer-id")
        assert writer not in ids, "duplicate writer id would break the writer-disjoint split"
        ids.add(writer)
    assert len(ids) == built.num_writers


def test_ascii_index_has_one_row_per_word(built):
    lines = (built.root / "ascii" / "words.txt").read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if not line.startswith("#")]
    assert len(rows) == built.num_words
    assert len(rows[0].split()) >= 9


def test_all_transcriptions_are_in_the_charset(built):
    alphabet = cs.get("english")
    for form_id in built.forms:
        for word in ET.parse(built.root / "xml" / f"{form_id}.xml").getroot().findall(".//word"):
            text = word.get("text")
            assert alphabet.supports(text), f"{text!r} is outside the charset"


def test_rebuilding_is_idempotent(tmp_path):
    """Regenerating over an existing fixture must reproduce it, not accumulate."""
    first = fixture.build(root=tmp_path, num_writers=2, words_per_writer=10, seed=0)
    before = sorted(p.name for p in (tmp_path / "words").rglob("*.png"))
    second = fixture.build(root=tmp_path, num_writers=2, words_per_writer=10, seed=0)
    after = sorted(p.name for p in (tmp_path / "words").rglob("*.png"))
    assert before == after
    assert first.num_words == second.num_words


def test_build_is_fast_enough_to_stay_in_the_test_suite(tmp_path):
    """The fixture's value depends on being cheap. 20x50 is the configured default."""
    import time

    started = time.perf_counter()
    fixture.build(root=tmp_path, num_writers=20, words_per_writer=50, seed=0)
    elapsed = time.perf_counter() - started
    assert elapsed < 30, f"fixture build took {elapsed:.1f}s -- too slow to keep running"
