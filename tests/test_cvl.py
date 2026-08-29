"""Tests for the CVL reader.

Filename parsing is the whole load-bearing mechanism here: in the cropped release
the writer id exists *only* in the filename, so a parsing slip does not raise, it
silently produces the wrong writer -- and every writer-disjoint guarantee
downstream is then meaningless.

The tests at the bottom run against the real database when it is present and skip
when it is not.
"""

from __future__ import annotations

import pytest

from nib.config import find_repo_root
from nib.data import cvl
from nib.data.split import counts_from_records, make_split

CVL_ROOT = find_repo_root() / "data" / "raw" / "cvl"
_REAL = CVL_ROOT.is_dir() and any(CVL_ROOT.rglob("*.tif"))

needs_cvl = pytest.mark.skipif(
    not _REAL, reason=f"CVL not present under {CVL_ROOT}; these checks activate once it is"
)

# Published figures for CVL, used as sanity bounds.
EXPECTED_WRITERS = 310
EXPECTED_PAGES = 1604


def make_tree(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"")
    return root


# --------------------------------------------------------------------------
# filename parsing
# --------------------------------------------------------------------------


def test_parses_writer_and_text_from_the_cropped_naming(tmp_path):
    make_tree(tmp_path, ["0001-1-cropped.tif", "0001-2-cropped.tif", "0042-6-cropped.tif"])
    inv = cvl.scan(tmp_path)
    assert len(inv.pages) == 3
    assert inv.writers == {"0001", "0042"}
    page = next(p for p in inv.pages if p.page_id == "0042-6")
    assert page.writer_id == "0042"
    assert page.text_id == "6"


def test_parses_the_full_release_naming_too(tmp_path):
    make_tree(tmp_path, ["0007-3.tif", "0008-4.tif"])
    inv = cvl.scan(tmp_path)
    assert inv.writers == {"0007", "0008"}


def test_files_are_found_at_any_depth(tmp_path):
    """It must not matter whether the archive was extracted into a subdirectory."""
    make_tree(tmp_path / "cvl-database-cropped-1-1", ["0001-1-cropped.tif"])
    assert cvl.scan(tmp_path).writers == {"0001"}


def test_unrecognised_filenames_are_reported_not_silently_skipped(tmp_path):
    make_tree(tmp_path, ["0001-1-cropped.tif", "readme_thumbnail.png", "scan.tif"])
    inv = cvl.scan(tmp_path)
    assert len(inv.pages) == 1
    assert len(inv.unparsed) == 2
    assert "unparsed" in inv.summary()


def test_the_same_page_from_two_releases_is_counted_once(tmp_path):
    """Extracting both archives side by side must not double a writer's page count
    and skew the split's balancing."""
    make_tree(tmp_path / "cropped", ["0001-1-cropped.tif"])
    make_tree(tmp_path / "full", ["0001-1.tif"])
    inv = cvl.scan(tmp_path)
    assert len(inv.pages) == 1


def test_pages_are_returned_in_a_stable_order(tmp_path):
    make_tree(tmp_path, [f"{w:04d}-{t}-cropped.tif" for w in range(5) for t in (1, 2)])
    ids = [p.page_id for p in cvl.scan(tmp_path).pages]
    assert ids == sorted(ids)


def test_word_and_line_crops_are_not_mistaken_for_unparsed_pages(tmp_path):
    """The full release ships 113,000 word and line crops alongside the pages.
    Flagging them as unparsed would make the warning meaningless."""
    pages = tmp_path / "cvl-database-1-1" / "testset" / "pages"
    make_tree(pages, ["0052-1.tif"])
    make_tree(
        tmp_path / "cvl-database-1-1" / "testset" / "words" / "0052", ["0052-1-0-0-Imagine.tif"]
    )
    make_tree(tmp_path / "cvl-database-1-1" / "testset" / "lines" / "0052", ["0052-1-0.tif"])

    inv = cvl.scan(tmp_path)
    assert len(inv.pages) == 1
    assert inv.unparsed == [], f"false alarm on {inv.unparsed}"


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no CVL directory"):
        cvl.scan(tmp_path / "nope")


def test_the_cropped_only_release_is_reported_as_such(tmp_path):
    """has_annotations means "the full release is here", not "transcriptions are
    here" -- CVL's XML carries line geometry and no text at all."""
    make_tree(tmp_path, ["0001-1-cropped.tif"])
    inv = cvl.scan(tmp_path)
    assert inv.has_annotations is False
    assert "pages only" in inv.summary()


def test_group_by_writer_partitions_everything(tmp_path):
    make_tree(tmp_path, [f"{w:04d}-{t}-cropped.tif" for w in range(4) for t in (1, 2, 3)])
    pages = cvl.scan(tmp_path).pages
    grouped = cvl.group_by_writer(pages)
    assert len(grouped) == 4
    assert sum(len(v) for v in grouped.values()) == len(pages)


# --------------------------------------------------------------------------
# against the real database
# --------------------------------------------------------------------------


@needs_cvl
def test_real_cvl_matches_its_published_shape():
    inv = cvl.scan(CVL_ROOT)
    print("\n" + inv.summary())
    assert len(inv.pages) == EXPECTED_PAGES
    assert len(inv.writers) == EXPECTED_WRITERS
    assert not inv.unparsed, f"unparsed CVL files: {inv.unparsed[:5]}"


@needs_cvl
def test_real_page_counts_per_writer_match_the_documentation():
    """CVL documents 27 writers producing 7 texts and 283 producing 5."""
    inv = cvl.scan(CVL_ROOT)
    spread = {}
    for count in inv.pages_per_writer().values():
        spread[count] = spread.get(count, 0) + 1
    assert spread == {5: 283, 7: 27}, spread


@needs_cvl
def test_real_text_numbering_skips_five():
    """Undocumented, and worth failing loudly on if a future release changes it --
    code that assumes contiguous numbering would break quietly."""
    inv = cvl.scan(CVL_ROOT)
    assert sorted(inv.texts, key=int) == ["1", "2", "3", "4", "6", "7", "8"]


@needs_cvl
def test_real_images_exist_and_open():
    from PIL import Image

    inv = cvl.scan(CVL_ROOT)
    for page in inv.pages[:5]:
        with Image.open(page.image_path) as image:
            assert image.width > 500 and image.height > 200


@needs_cvl
def test_the_committed_split_still_matches_the_real_data():
    """The split file is committed. If the data on disk ever stops matching it,
    every reported number becomes incomparable -- so fail here, loudly."""
    from nib.data.split import WriterSplit

    path = find_repo_root() / "configs" / "splits" / "cvl-writer-disjoint.json"
    if not path.is_file():
        pytest.skip("no committed split yet")

    inv = cvl.scan(CVL_ROOT)
    split = WriterSplit.load(path)
    assert split.all_writers == inv.writers, "committed split does not cover the data on disk"

    parts = split.partition(inv.pages)
    assert sum(len(v) for v in parts.values()) == len(inv.pages)
    assert not set(split.writers["train"]) & set(split.writers["test"])


@needs_cvl
def test_rebuilding_the_split_from_real_data_reproduces_the_committed_one():
    """Determinism, end to end, on the real database."""
    from nib.data.split import WriterSplit

    path = find_repo_root() / "configs" / "splits" / "cvl-writer-disjoint.json"
    if not path.is_file():
        pytest.skip("no committed split yet")

    inv = cvl.scan(CVL_ROOT)
    committed = WriterSplit.load(path)
    rebuilt = make_split(
        counts_from_records(inv.pages, key=lambda p: p.writer_id),
        ratios=committed.ratios,
        seed=committed.seed,
        name=committed.name,
    )
    assert rebuilt.writers == committed.writers
