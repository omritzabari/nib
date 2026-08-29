"""Tests for the CVL word-level reader.

The whole label depends on parsing a filename correctly, so a parsing slip does
not raise -- it silently mislabels a training pair. The three traps that CVL's
naming actually contains each get a test: hyphens inside the transcription,
non-contiguous indices, and German words that fall outside an English charset.

The tests at the bottom run against the real database once it is extracted.
"""

from __future__ import annotations

import pytest

from nib.config import find_repo_root
from nib.data.cvl_words import (
    DROP_OUT_OF_CHARSET,
    DROP_UNPARSED_NAME,
    EXCLUDED_WRITERS,
    group_by_page,
    group_by_writer,
    official_split,
    parse_word_filename,
    scan_words,
)

CVL_ROOT = find_repo_root() / "data" / "raw" / "cvl"
FULL_ROOT = CVL_ROOT / "cvl-database-1-1"

EXPECTED_WRITERS = 310
EXPECTED_WORD_IMAGES = 99904


def _word_images_on_disk() -> int:
    if not FULL_ROOT.is_dir():
        return 0
    return sum(1 for p in FULL_ROOT.rglob("*.tif") if "words" in p.parts)


# A partly extracted archive is not the same as an absent one, and must not be
# reported as a data error. Extraction takes minutes; skip until it is complete.
_ON_DISK = _word_images_on_disk()
_REAL = _ON_DISK >= EXPECTED_WORD_IMAGES

needs_cvl = pytest.mark.skipif(
    not _REAL,
    reason=(
        f"full CVL release not ready: {_ON_DISK} of {EXPECTED_WORD_IMAGES} word images "
        "present (extraction still running, or the archive was never extracted)"
    ),
)


def make_words(tmp_path, names, split="testset"):
    d = tmp_path / "cvl-database-1-1" / split / "words" / "0052"
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        (d / f"{name}.tif").write_bytes(b"")
    return tmp_path


# --------------------------------------------------------------------------
# filename parsing -- the whole label rests on this
# --------------------------------------------------------------------------


def test_parses_the_five_fields():
    assert parse_word_filename("0052-1-0-0-Imagine") == ("0052", "1", 0, 0, "Imagine")


def test_a_hyphen_inside_the_word_is_kept():
    """The trap. About 875 of CVL's 99,904 filenames have more than five fields,
    and a naive split on '-' truncates every one of them."""
    assert parse_word_filename("0052-1-3-7-well-known")[4] == "well-known"
    assert parse_word_filename("0052-1-3-7-a-b-c-d")[4] == "a-b-c-d"


def test_indices_are_returned_as_integers_for_ordering():
    _, _, line, word, _ = parse_word_filename("0052-1-10-7-higher")
    assert (line, word) == (10, 7)
    assert isinstance(line, int)


def test_an_unrecognisable_name_returns_none_rather_than_guessing():
    assert parse_word_filename("thumbnail") is None
    assert parse_word_filename("0052-1-0") is None
    assert parse_word_filename("52-1-0-0-word") is None  # writer id is four digits


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------


def test_scan_reads_writer_text_line_word_and_transcription(tmp_path):
    make_words(tmp_path, ["0052-1-0-0-Imagine", "0052-1-0-1-a"])
    words, report = scan_words(tmp_path)
    assert len(words) == 2
    assert words[0].writer_id == "0052"
    assert words[0].text_id == "1"
    assert words[0].text == "Imagine"
    assert words[0].word_id == "0052-1-0-0"
    assert words[0].page_id == "0052-1"
    assert report.kept == 2


def test_non_contiguous_indices_are_fine(tmp_path):
    """Words whose segmentation failed were dropped, so the numbering has gaps.
    Anything assuming a dense range would lose data or crash."""
    make_words(tmp_path, ["0052-1-1-0-straight", "0052-1-1-2-Lines", "0052-1-10-7-views"])
    words, _ = scan_words(tmp_path)
    assert [(w.line_index, w.word_index) for w in words] == [(1, 0), (1, 2), (10, 7)]


def test_words_are_sorted_deterministically(tmp_path):
    make_words(tmp_path, ["0052-1-10-0-b", "0052-1-2-0-a", "0052-1-2-1-c"])
    words, _ = scan_words(tmp_path)
    assert [(w.line_index, w.word_index) for w in words] == [(2, 0), (2, 1), (10, 0)]


def test_german_words_are_excluded_and_counted_not_dropped_silently(tmp_path):
    """One of CVL's seven passages is Goethe's Faust. Those words fall outside an
    English charset, and the count is how you find out how many were lost."""
    make_words(tmp_path, ["0052-5-0-0-Imagine", "0052-5-0-1-Mailüfterl", "0052-5-0-2-schön"])
    words, report = scan_words(tmp_path)
    assert [w.text for w in words] == ["Imagine"]
    assert report.dropped[DROP_OUT_OF_CHARSET] == 2


def test_out_of_charset_words_can_be_kept_deliberately(tmp_path):
    make_words(tmp_path, ["0052-5-0-1-Mailüfterl"])
    words, _ = scan_words(tmp_path, keep_out_of_charset=True)
    assert len(words) == 1


def test_the_excluded_writer_from_the_readme_is_skipped(tmp_path):
    """CVL's readme says writer 0431 was left out of the published evaluation,
    although the files are still shipped. Excluding it keeps our writer-retrieval
    numbers comparable with the paper's."""
    assert "0431" in EXCLUDED_WRITERS
    d = tmp_path / "cvl-database-1-1" / "testset" / "words" / "0431"
    d.mkdir(parents=True)
    (d / "0431-1-0-0-word.tif").write_bytes(b"")
    make_words(tmp_path, ["0052-1-0-0-Imagine"])

    words, report = scan_words(tmp_path)
    assert [w.writer_id for w in words] == ["0052"]
    assert report.excluded_writers == {"0431"}
    assert "0431" in report.summary()


def test_unparsable_filenames_are_counted(tmp_path):
    make_words(tmp_path, ["0052-1-0-0-Imagine", "thumbs", "notes"])
    _, report = scan_words(tmp_path)
    assert report.dropped[DROP_UNPARSED_NAME] == 2


def test_the_report_accounts_for_everything_seen(tmp_path):
    make_words(tmp_path, ["0052-1-0-0-good", "0052-1-0-1-schön", "junk"])
    _, report = scan_words(tmp_path)
    assert report.total_seen == 3
    assert report.kept == 1


def test_only_files_under_a_words_directory_are_read(tmp_path):
    """Line and page images live beside the words and must not be mistaken for them."""
    make_words(tmp_path, ["0052-1-0-0-Imagine"])
    lines = tmp_path / "cvl-database-1-1" / "testset" / "lines" / "0052"
    lines.mkdir(parents=True)
    (lines / "0052-1-0.tif").write_bytes(b"")
    words, _ = scan_words(tmp_path)
    assert len(words) == 1


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no CVL directory"):
        scan_words(tmp_path / "nope")


def test_grouping_helpers_partition_everything(tmp_path):
    make_words(tmp_path, ["0052-1-0-0-a", "0052-1-0-1-b", "0052-2-0-0-c"])
    words, _ = scan_words(tmp_path)
    by_writer = group_by_writer(words)
    by_page = group_by_page(words)
    assert set(by_writer) == {"0052"}
    assert set(by_page) == {"0052-1", "0052-2"}
    assert sum(len(v) for v in by_page.values()) == 3


def test_official_split_is_recorded_per_word(tmp_path):
    make_words(tmp_path, ["0052-1-0-0-a"], split="testset")
    make_words(tmp_path, ["0100-1-0-0-b"], split="trainset")
    words, _ = scan_words(tmp_path)
    parts = official_split(words)
    assert [w.writer_id for w in parts["trainset"]] == ["0100"]
    assert [w.writer_id for w in parts["testset"]] == ["0052"]


# --------------------------------------------------------------------------
# against the real database
# --------------------------------------------------------------------------


@needs_cvl
def test_real_release_has_the_expected_number_of_word_images():
    assert _ON_DISK == EXPECTED_WORD_IMAGES, (
        f"{_ON_DISK} word images, expected {EXPECTED_WORD_IMAGES}"
    )


@needs_cvl
def test_real_scan_keeps_most_words_and_reports_the_rest():
    words, report = scan_words(CVL_ROOT)
    print("\n" + report.summary())
    assert words
    assert report.kept / report.total_seen > 0.7, (
        f"kept only {report.kept / report.total_seen:.1%}:\n{report.summary()}"
    )
    assert len(report.writers) == EXPECTED_WRITERS - len(EXCLUDED_WRITERS)


@needs_cvl
def test_every_file_on_disk_is_accounted_for():
    """kept + dropped + deliberately excluded must equal the file count. A total
    that quietly disagrees with the disk is how small losses go unnoticed."""
    _, report = scan_words(CVL_ROOT)
    assert report.total_seen == _ON_DISK, (
        f"{_ON_DISK} files on disk but {report.total_seen} accounted for; "
        f"{_ON_DISK - report.total_seen} vanished"
    )


@needs_cvl
def test_real_transcriptions_are_all_inside_the_charset():
    from nib.data import charset as cs

    alphabet = cs.get("english")
    words, _ = scan_words(CVL_ROOT)
    for word in words:
        assert alphabet.supports(word.text), f"{word.text!r} slipped through the filter"


@needs_cvl
def test_real_official_split_is_writer_disjoint():
    words, _ = scan_words(CVL_ROOT)
    parts = official_split(words)
    train = {w.writer_id for w in parts["trainset"]}
    test = {w.writer_id for w in parts["testset"]}
    assert not (train & test), "CVL's own split leaks writers"
    assert len(train) == 27, f"expected 27 train writers, got {len(train)}"


@needs_cvl
def test_real_writers_have_enough_words_to_condition_a_style():
    """Style conditioning needs several sample words. A writer with three usable
    words cannot be styled from, and that should surface now."""
    words, _ = scan_words(CVL_ROOT)
    grouped = group_by_writer(words)
    thin = {w: len(v) for w, v in grouped.items() if len(v) < 30}
    assert len(thin) < 0.05 * len(grouped), f"writers with too few words: {thin}"


@needs_cvl
def test_real_word_images_open_and_are_reasonable():
    from PIL import Image

    words, _ = scan_words(CVL_ROOT)
    for word in words[:20]:
        with Image.open(word.image_path) as image:
            assert image.width > 5 and image.height > 5
            assert image.width < 4000 and image.height < 1000
