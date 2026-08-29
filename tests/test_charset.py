"""Tests for the charset.

The failure modes worth guarding: a roundtrip that silently loses characters, an
index layout that drifts (which would invalidate every saved artefact), and an
unknown character being encoded as something plausible instead of as unk.
"""

import pytest

from nib.config import load_config
from nib.data import charset as cs


def test_index_layout_is_fixed():
    """pad=0 and unk=1 are baked into saved checkpoints and packed datasets.
    If this test ever fails, every existing artefact is invalid."""
    assert cs.PAD_INDEX == 0
    assert cs.UNK_INDEX == 1
    english = cs.get("english")
    assert english.encode("A")[0] >= 2


def test_english_matches_the_iam_alphabet():
    english = cs.get("english")
    assert len(english.characters) == 79, "IAM's alphabet is 79 characters"
    assert len(english) == 81, "vocabulary is the alphabet plus pad and unk"
    for expected in [" ", "!", '"', "#", "&", "'", "?", "0", "9", "A", "Z", "a", "z"]:
        assert expected in english, f"{expected!r} missing from the IAM alphabet"


def test_roundtrip_is_lossless():
    english = cs.get("english")
    text = "The quick brown fox, jumps over 13 lazy dogs! (Really?)"
    assert english.decode(english.encode(text)) == text


def test_empty_text():
    english = cs.get("english")
    assert english.encode("") == []
    assert english.decode([]) == ""


def test_unknown_characters_become_unk_not_something_plausible():
    english = cs.get("english")
    assert english.encode("é") == [cs.UNK_INDEX]
    assert english.unsupported("café ☕") == {"é", "☕"}
    assert not english.supports("café")
    assert english.supports("cafe")


def test_decode_drops_padding():
    english = cs.get("english")
    encoded = english.encode("hi") + [cs.PAD_INDEX] * 5
    assert english.decode(encoded) == "hi"


def test_decode_can_show_unknowns_when_asked():
    english = cs.get("english")
    encoded = english.encode("a?b")
    encoded[1] = cs.UNK_INDEX
    assert english.decode(encoded) == "ab"
    assert english.decode(encoded, keep_unknown=True) == "a\ufffdb"


def test_out_of_range_index_raises():
    english = cs.get("english")
    with pytest.raises(ValueError, match="outside charset"):
        english.decode([9999])


def test_filter_drops_unsupported():
    english = cs.get("english")
    assert english.filter("café") == "caf"


def test_duplicate_characters_rejected():
    """Duplicates would make decode ambiguous, so they must fail at construction."""
    with pytest.raises(ValueError, match="duplicate"):
        cs.Charset(name="broken", characters="abca")


def test_empty_charset_rejected():
    with pytest.raises(ValueError, match="empty"):
        cs.Charset(name="broken", characters="")


def test_unknown_charset_name_raises():
    with pytest.raises(KeyError, match="unknown charset"):
        cs.get("klingon")


def test_config_charset_name_resolves():
    """The name in configs/base.yaml must actually exist. Catches a typo in the
    config before it becomes a crash mid-run."""
    cfg = load_config()
    assert cfg.data.charset in cs.available()
    assert cs.get(cfg.data.charset) is not None


def test_adding_an_alphabet_is_a_data_change_only():
    """The design claim, tested: registering a new script requires no edit to
    charset.py. Uses a throwaway alphabet so the test asserts the mechanism
    rather than any particular language.
    """
    greek = cs.register("greek_test", "αβγδε")
    try:
        assert len(greek) == 7  # five letters plus pad and unk
        assert greek.decode(greek.encode("αβγ")) == "αβγ"
        assert "greek_test" in cs.available()
        assert greek.encode("a") == [cs.UNK_INDEX]  # separate alphabets stay separate
    finally:
        cs._REGISTRY.pop("greek_test", None)


def test_registering_over_an_existing_name_requires_intent():
    with pytest.raises(KeyError, match="already registered"):
        cs.register("english", "abc")
