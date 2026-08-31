"""Tests for the CVL line-level reader.

Every failure this module can have is silent. A line paired with the wrong text
does not raise -- it teaches the CER metric that the recogniser got a word wrong
when in fact the ground truth was incomplete. So the tests below are mostly about
transcriptions that look fine and are not.

Three properties carry the weight:

* words are joined by **index**, not by filename order, or word 10 lands before
  word 2 and the sentence is scrambled;
* a line with missing word crops is **dropped**, because its image holds ink its
  text does not describe;
* the drop reasons are attributed in a fixed order, so the reported counts mean
  what the module docstring says they mean.

The tests at the bottom run against the real database once it is extracted.
"""

from __future__ import annotations

import pytest

from nib.config import find_repo_root
from nib.data.cvl_lines import (
    DROP_INCOMPLETE_TRANSCRIPTION,
    DROP_NO_WORDS,
    DROP_OUT_OF_CHARSET,
    DROP_UNPARSED_NAME,
    group_by_writer,
    index_words_by_line,
    scan_lines,
)

CVL_ROOT = find_repo_root() / "data" / "raw" / "cvl"
FULL_ROOT = CVL_ROOT / "cvl-database-1-1"

# Measured on the extracted database, then written down here -- not the other way
# round. A change in any of these numbers means the data or the filtering moved,
# and either is worth stopping for.
EXPECTED_LINE_IMAGES = 13473
EXPECTED_KEPT = 10862
EXPECTED_WRITERS = 309  # 310 minus writer 0431, whom CVL's own readme excludes
EXPECTED_HELD_OUT_LINES = 3264


def _line_images_on_disk() -> int:
    if not FULL_ROOT.is_dir():
        return 0
    return sum(1 for p in FULL_ROOT.rglob("*.tif") if "lines" in p.parts)


_ON_DISK = _line_images_on_disk()

needs_cvl = pytest.mark.skipif(
    _ON_DISK < EXPECTED_LINE_IMAGES,
    reason=(
        f"full CVL release not ready: {_ON_DISK} of {EXPECTED_LINE_IMAGES} line images "
        "present (extraction still running, or the archive was never extracted)"
    ),
)


def make_tree(tmp_path, line_stems=(), word_stems=(), split="testset"):
    """A miniature CVL tree. The images are empty: nothing here reads them."""
    root = tmp_path / "cvl-database-1-1" / split
    for kind, stems in (("lines", line_stems), ("words", word_stems)):
        for stem in stems:
            directory = root / kind / stem[:4]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{stem}.tif").write_bytes(b"")
    return tmp_path


# --------------------------------------------------------------------------
# reassembling the transcription
# --------------------------------------------------------------------------


def test_joins_words_by_index_not_by_filename_order(tmp_path):
    """Filename order is a string sort, in which "10" comes before "2"."""
    words = [f"0052-1-0-{i}-w{i}" for i in range(11)]
    root = make_tree(tmp_path, ["0052-1-0"], words)

    lines, report = scan_lines(root)

    assert report.kept == 1
    assert lines[0].text == " ".join(f"w{i}" for i in range(11))
    assert lines[0].word_count == 11


def test_hyphens_inside_a_word_survive(tmp_path):
    root = make_tree(tmp_path, ["0052-1-0"], ["0052-1-0-0-well-being", "0052-1-0-1-there"])

    lines, _ = scan_lines(root)

    assert lines[0].text == "well-being there"


def test_records_carry_their_identity_and_split(tmp_path):
    root = make_tree(tmp_path, ["0052-3-7"], ["0052-3-7-0-Imagine"], split="trainset")

    lines, _ = scan_lines(root)

    assert lines[0].writer_id == "0052"
    assert lines[0].text_id == "3"
    assert lines[0].line_index == 7
    assert lines[0].line_id == "0052-3-7"
    assert lines[0].page_id == "0052-3"
    assert lines[0].split == "trainset"
    assert lines[0].complete is True


# --------------------------------------------------------------------------
# the trap: a line whose transcription is missing a word
# --------------------------------------------------------------------------


def test_drops_a_line_whose_word_indices_have_gaps(tmp_path):
    """Word 1's crop failed segmentation, but its ink is still in the line."""
    root = make_tree(tmp_path, ["0052-1-0"], ["0052-1-0-0-Imagine", "0052-1-0-2-sheet"])

    lines, report = scan_lines(root)

    assert lines == []
    assert report.dropped[DROP_INCOMPLETE_TRANSCRIPTION] == 1


def test_drops_a_line_whose_words_do_not_start_at_zero(tmp_path):
    """A missing *first* word leaves no gap in the middle, and is just as wrong."""
    root = make_tree(tmp_path, ["0052-1-0"], ["0052-1-0-1-vast", "0052-1-0-2-sheet"])

    _, report = scan_lines(root)

    assert report.dropped[DROP_INCOMPLETE_TRANSCRIPTION] == 1


def test_keep_incomplete_flags_the_line_rather_than_dropping_it(tmp_path):
    root = make_tree(tmp_path, ["0052-1-0"], ["0052-1-0-0-Imagine", "0052-1-0-2-sheet"])

    lines, report = scan_lines(root, keep_incomplete=True)

    assert len(lines) == 1
    assert lines[0].complete is False
    assert report.incomplete_kept == 1
    assert DROP_INCOMPLETE_TRANSCRIPTION not in report.dropped


# --------------------------------------------------------------------------
# the other exclusions, and the order they are attributed in
# --------------------------------------------------------------------------


def test_out_of_charset_is_attributed_before_incompleteness(tmp_path):
    """A German line that *also* has a gap counts as German.

    The order is the documented one and it decides what the numbers mean: German
    is an English-only scope decision, while incompleteness is a data defect. If
    the two were swapped, the "incomplete" figure would be inflated by passages
    we were never going to use.
    """
    root = make_tree(tmp_path, ["0052-3-0"], ["0052-3-0-0-Binär", "0052-3-0-2-Zahl"])

    _, report = scan_lines(root)

    assert report.dropped[DROP_OUT_OF_CHARSET] == 1
    assert DROP_INCOMPLETE_TRANSCRIPTION not in report.dropped


def test_keep_out_of_charset_lets_german_through(tmp_path):
    root = make_tree(tmp_path, ["0052-3-0"], ["0052-3-0-0-Binär"])

    lines, _ = scan_lines(root, keep_out_of_charset=True)

    assert lines[0].text == "Binär"


def test_drops_a_line_with_no_word_files(tmp_path):
    root = make_tree(tmp_path, ["0052-1-0"], [])

    lines, report = scan_lines(root)

    assert lines == []
    assert report.dropped[DROP_NO_WORDS] == 1


def test_counts_a_line_filename_that_does_not_parse(tmp_path):
    """A word image filed under lines/ has five fields, not three."""
    root = make_tree(tmp_path, ["0052-1-0-0-Imagine"], ["0052-1-0-0-Imagine"])

    lines, report = scan_lines(root)

    assert lines == []
    assert report.dropped[DROP_UNPARSED_NAME] == 1


def test_excluded_writer_is_counted_apart_from_the_drops(tmp_path):
    """Writer 0431 is a deliberate exclusion, not a data failure. Reporting it as
    a drop would mix a decision in with the defects."""
    root = make_tree(tmp_path, ["0431-1-0"], ["0431-1-0-0-Imagine"])

    lines, report = scan_lines(root)

    assert lines == []
    assert report.excluded_lines == 1
    assert report.excluded_writers == {"0431"}
    assert sum(report.dropped.values()) == 0


def test_every_line_image_is_accounted_for(tmp_path):
    """total_seen must equal the files on disk. A total that quietly disagrees
    with `ls | wc -l` is how small losses go unnoticed."""
    root = make_tree(
        tmp_path,
        ["0052-1-0", "0052-1-1", "0052-3-0", "0052-1-2", "0431-1-0"],
        [
            "0052-1-0-0-Imagine",  # kept
            "0052-1-1-1-vast",  # incomplete
            "0052-3-0-0-Binär",  # out of charset
            "0431-1-0-0-Imagine",  # excluded writer
            # 0052-1-2 gets no words at all
        ],
    )

    _, report = scan_lines(root)

    assert report.total_seen == 5
    assert report.kept == 1
    assert sum(report.dropped.values()) == 3
    assert report.excluded_lines == 1


# --------------------------------------------------------------------------
# ordering, grouping, and the word index
# --------------------------------------------------------------------------


def test_order_is_deterministic_and_numeric_on_the_line_index(tmp_path):
    stems = ["0052-1-10", "0052-1-2", "0052-2-0", "0051-1-0"]
    root = make_tree(tmp_path, stems, [f"{s}-0-w" for s in stems])

    lines, _ = scan_lines(root)

    assert [line.line_id for line in lines] == ["0051-1-0", "0052-1-2", "0052-1-10", "0052-2-0"]


def test_group_by_writer(tmp_path):
    stems = ["0052-1-0", "0052-1-1", "0053-1-0"]
    root = make_tree(tmp_path, stems, [f"{s}-0-w" for s in stems])

    grouped = group_by_writer(scan_lines(root)[0])

    assert sorted(grouped) == ["0052", "0053"]
    assert len(grouped["0052"]) == 2


def test_word_index_is_unfiltered(tmp_path):
    """The index must keep German and excluded writers, or the gap they leave
    behind becomes invisible and an incomplete line looks complete."""
    root = make_tree(tmp_path, [], ["0052-3-0-0-Binär", "0431-1-0-0-Imagine"])

    index = index_words_by_line(root)

    assert index[("0052", "3", 0)] == [(0, "Binär")]
    assert index[("0431", "1", 0)] == [(0, "Imagine")]


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_lines(tmp_path / "nowhere")


# --------------------------------------------------------------------------
# the real database
# --------------------------------------------------------------------------


@needs_cvl
def test_real_cvl_counts_are_stable():
    _, report = scan_lines(CVL_ROOT)

    assert report.total_seen == EXPECTED_LINE_IMAGES
    assert report.kept == EXPECTED_KEPT
    assert len(report.writers) == EXPECTED_WRITERS
    assert report.excluded_writers == {"0431"}


@needs_cvl
def test_real_cvl_transcriptions_read_as_english():
    lines, _ = scan_lines(CVL_ROOT)

    first = lines[0]
    assert first.line_id == "0001-1-0"
    assert first.text == "Imagine a vast sheet of paper on which straight"


@needs_cvl
def test_every_held_out_writer_has_enough_lines_to_evaluate():
    """The evaluation needs a style reference, a target, and a retrieval gallery
    from each held-out writer. If one writer fell below that after filtering, the
    evaluation would quietly cover 93 writers instead of 94."""
    split_file = find_repo_root() / "configs" / "splits" / "cvl-writer-disjoint.json"
    from nib.data.split import WriterSplit

    split = WriterSplit.load(split_file)
    grouped = group_by_writer(scan_lines(CVL_ROOT)[0])

    held_out = {w: grouped.get(w, []) for w in split.writers["test"]}

    assert all(len(v) >= 12 for v in held_out.values())
    assert sum(len(v) for v in held_out.values()) == EXPECTED_HELD_OUT_LINES
