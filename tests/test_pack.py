"""Tests for the LMDB pack.

The pack is the artefact that gets copied to Colab and trained from. Two classes
of failure matter here: a record coming back different from how it went in, which
would corrupt training silently; and a half-written file looking complete, which
would corrupt a run that starts from it.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from nib.config import find_repo_root, load_config
from nib.data.pack import (
    HEADER_KEY,
    PackedSample,
    PackError,
    PackHeader,
    PackReader,
    PackWriter,
    compact,
    is_complete,
)

PACK = find_repo_root() / "data" / "processed" / "cvl_words_64.lmdb"
# "exists" is not the same as "finished": a pack being built exists but has no
# header yet, and treating that as a data error is noise, not a finding.
needs_pack = pytest.mark.skipif(
    not is_complete(PACK), reason=f"no complete pack at {PACK}; run scripts/build_index.py"
)


def word(key="0001-1-0-0", writer="0001", text="Imagine", split="testset", height=64, width=100):
    rng = np.random.default_rng(abs(hash(key)) % 2**32)
    return PackedSample(
        key=key,
        writer_id=writer,
        text=text,
        split=split,
        image=rng.integers(0, 255, (height, width), dtype=np.uint8),
    )


def build(path, words, height=64):
    with PackWriter(path, PackHeader(height=height)) as writer:
        for w in words:
            writer.add(w)
    return path


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def test_a_record_comes_back_exactly_as_it_went_in(tmp_path):
    """PNG is lossless, and it must stay that way -- a silently lossy codec would
    degrade every training sample without ever raising."""
    original = word()
    path = build(tmp_path / "p.lmdb", [original])

    with PackReader(path) as reader:
        restored = reader[0]
    assert restored.key == original.key
    assert restored.writer_id == original.writer_id
    assert restored.text == original.text
    assert restored.split == original.split
    assert np.array_equal(restored.image, original.image), "the image changed in storage"


def test_variable_widths_survive(tmp_path):
    """Width carries the aspect ratio of the word. If the pack normalised it away,
    the variable-width collate path downstream would be testing nothing."""
    words = [word(key=f"k{i}", width=w) for i, w in enumerate([40, 180, 95])]
    path = build(tmp_path / "p.lmdb", words)
    with PackReader(path) as reader:
        assert sorted(r.image.shape[1] for r in reader) == [40, 95, 180]


def test_records_can_be_fetched_by_key_as_well_as_index(tmp_path):
    path = build(tmp_path / "p.lmdb", [word(key="a"), word(key="b")])
    with PackReader(path) as reader:
        assert reader["b"].key == "b"
        with pytest.raises(KeyError, match="not in"):
            reader["missing"]


def test_insertion_order_is_preserved(tmp_path):
    """Order is the basis of a reproducible split and a reproducible epoch."""
    keys = [f"k{i:03d}" for i in range(50)]
    path = build(tmp_path / "p.lmdb", [word(key=k) for k in keys])
    with PackReader(path) as reader:
        assert reader.keys == keys


def test_iteration_yields_every_record(tmp_path):
    path = build(tmp_path / "p.lmdb", [word(key=f"k{i}") for i in range(5)])
    with PackReader(path) as reader:
        assert len(list(reader)) == len(reader) == 5


# ---------------------------------------------------------------------------
# batching must not change behaviour
# ---------------------------------------------------------------------------


def test_more_records_than_one_write_batch(tmp_path):
    """Writes are batched for speed. The batch boundary is a place where records
    could be dropped, so cross it."""
    from nib.data.pack import WRITE_BATCH

    count = WRITE_BATCH + 137
    path = build(tmp_path / "p.lmdb", [word(key=f"k{i:06d}", width=20) for i in range(count)])
    with PackReader(path) as reader:
        assert len(reader) == count
        assert reader[count - 1].key == f"k{count - 1:06d}"


# ---------------------------------------------------------------------------
# refusing bad input
# ---------------------------------------------------------------------------


def test_a_wrong_height_is_refused(tmp_path):
    """Every record must match the declared height, or batches come out ragged in
    a way nothing downstream expects."""
    with pytest.raises(PackError, match="does not match"):
        build(tmp_path / "p.lmdb", [word(height=32)], height=64)


def test_a_colour_image_is_refused(tmp_path):
    coloured = PackedSample("k", "0001", "x", "testset", np.zeros((64, 20, 3), np.uint8))
    with pytest.raises(PackError, match="grayscale"):
        build(tmp_path / "p.lmdb", [coloured])


# ---------------------------------------------------------------------------
# a truncated pack must not look complete
# ---------------------------------------------------------------------------


def test_a_pack_without_a_header_is_rejected(tmp_path):
    """The header is written last on purpose. An interrupted build leaves a file
    that is detectably incomplete rather than quietly short by ten thousand
    records -- which would look like a training problem, not a build problem."""
    import lmdb

    path = tmp_path / "partial.lmdb"
    env = lmdb.open(str(path), map_size=10 * 1024**2, subdir=False)
    with env.begin(write=True) as txn:
        txn.put(b"0001-1-0-0", b"whatever")
    env.close()

    with pytest.raises(PackError, match="not closed properly"):
        PackReader(path)
    assert not is_complete(path)


def test_a_future_format_version_is_rejected(tmp_path):
    import json

    import lmdb

    path = build(tmp_path / "p.lmdb", [word()])
    env = lmdb.open(str(path), map_size=10 * 1024**2, subdir=False)
    with env.begin(write=True) as txn:
        header = json.loads(txn.get(HEADER_KEY).decode())
        header["format_version"] = 99
        txn.put(HEADER_KEY, json.dumps(header).encode())
    env.close()

    with pytest.raises(PackError, match="format 99"):
        PackReader(path)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no pack at"):
        PackReader(tmp_path / "nope.lmdb")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_reported_size_is_data_not_reserved_address_space(tmp_path):
    """stat() on the file reports the map size -- 8 GB of reservation, which read
    as a 40 GB pack for a 500-record test and sends you hunting a bug that is not
    there."""
    path = build(tmp_path / "p.lmdb", [word(key=f"k{i}", width=30) for i in range(200)])
    with PackReader(path) as reader:
        assert reader.data_size_bytes() < 50 * 1024**2
        assert "MB" in reader.summary()


def test_compaction_shrinks_the_file_and_keeps_the_data(tmp_path):
    """LMDB reserves its whole map size up front, so a pack holding 25 MB reports
    8 GB. The file is sparse, so that costs nothing locally -- and costs
    everything the moment it is uploaded to Drive, which transfers the apparent
    size. Compaction is therefore not optional for the shipped artefact.
    """
    path = build(tmp_path / "p.lmdb", [word(key=f"k{i:05d}", width=40) for i in range(300)])
    before = path.stat().st_size

    compact(path)
    after = path.stat().st_size
    assert after < before / 10, f"compaction barely helped: {before} -> {after}"

    with PackReader(path) as reader:
        assert len(reader) == 300
        assert reader[299].key == "k00299"
        assert reader.header.height == 64


def test_compaction_can_write_a_copy_instead(tmp_path):
    path = build(tmp_path / "p.lmdb", [word(key=f"k{i}", width=40) for i in range(50)])
    copy = compact(path, tmp_path / "shipped.lmdb")
    assert copy.exists() and path.exists()
    with PackReader(copy) as reader:
        assert len(reader) == 50


def test_header_records_what_the_pack_is(tmp_path):
    path = build(tmp_path / "p.lmdb", [word(key=f"k{i}") for i in range(7)])
    with PackReader(path) as reader:
        assert reader.header.count == 7
        assert reader.header.writers == 1
        assert reader.header.height == 64


def test_writers_index_groups_keys(tmp_path):
    words = [word(key=f"a{i}", writer="0001") for i in range(3)]
    words += [word(key=f"b{i}", writer="0002") for i in range(2)]
    path = build(tmp_path / "p.lmdb", words)
    with PackReader(path) as reader:
        grouped = reader.writers()
    assert {k: len(v) for k, v in grouped.items()} == {"0001": 3, "0002": 2}


# ---------------------------------------------------------------------------
# against the real pack
# ---------------------------------------------------------------------------


@needs_pack
def test_the_real_pack_holds_every_usable_word():
    with PackReader(PACK) as reader:
        print("\n" + reader.summary())
        assert len(reader) > 90000
        assert reader.header.writers == 309
        assert reader.header.height == 64


@needs_pack
def test_the_real_pack_agrees_with_the_config():
    """A pack built at one height and a config asking for another would silently
    produce wrongly-sized batches."""
    cfg = load_config(find_repo_root() / "configs" / "base.yaml")
    with PackReader(PACK) as reader:
        assert reader.header.height == int(cfg.data.image_height)
        assert reader.header.charset == str(cfg.data.charset)


@needs_pack
def test_the_real_pack_decodes_and_keeps_aspect_ratios():
    with PackReader(PACK) as reader:
        widths = set()
        for index in range(0, len(reader), max(1, len(reader) // 200)):
            record = reader[index]
            assert record.image.shape[0] == reader.header.height
            assert record.text
            assert record.writer_id
            widths.add(record.image.shape[1])
    assert len(widths) > 20, "every word came out the same width; aspect ratio was lost"


@needs_pack
def test_the_real_pack_covers_the_committed_split():
    """The pack and the writer split must describe the same population, or the
    split silently drops or invents writers."""
    from nib.data.split import WriterSplit

    split_path = find_repo_root() / "configs" / "splits" / "cvl-writer-disjoint.json"
    if not split_path.is_file():
        pytest.skip("no committed split")

    with PackReader(PACK) as reader:
        in_pack = set(reader.writers())
    split = WriterSplit.load(split_path)

    missing = in_pack - split.all_writers
    assert not missing, (
        f"{len(missing)} writers in the pack are absent from the split: {sorted(missing)[:5]}"
    )


# --------------------------------------------------------------------------
# the line pack
#
# Words and lines share a format, which is convenient and is also the way the
# two could be confused. The tests below check the property that actually
# separates them, not just the label: a line is several words wide.
# --------------------------------------------------------------------------

LINE_PACK = find_repo_root() / "data" / "processed" / "cvl_lines_64.lmdb"
EXPECTED_LINES = 10862

needs_line_pack = pytest.mark.skipif(
    not is_complete(LINE_PACK),
    reason=f"no complete pack at {LINE_PACK}; run scripts/build_index.py --unit lines",
)


@needs_line_pack
def test_the_line_pack_declares_what_it_holds():
    """The header is how a file says which unit it carries. A pack that does not
    say would have to be guessed at from its contents, and a wrong guess would
    reach the generator as a silently wrong style reference."""
    with PackReader(LINE_PACK) as reader:
        assert reader.header.source == "cvl-lines"
        assert reader.header.count == EXPECTED_LINES
        assert reader.header.height == 64


@needs_line_pack
def test_the_line_pack_really_holds_lines_and_not_words():
    """The discriminating property, checked rather than trusted.

    Building the line pack with the word scanner would produce a file that looks
    healthy, passes every other test, and quietly evaluates the generator on the
    unit it fails at. So: a line is several words wide and its text has spaces in
    it. A word pack averages 156px at this height; a line pack averages ~880.
    """
    with PackReader(LINE_PACK) as reader:
        step = max(1, len(reader) // 300)
        records = [reader[i] for i in range(0, len(reader), step)]

    widths = [r.image.shape[1] for r in records]
    multiword = [r for r in records if " " in r.text]

    assert sum(widths) / len(widths) > 400, "these are word crops, not lines"
    assert len(multiword) > 0.9 * len(records), "most 'lines' hold a single word"


@needs_line_pack
def test_the_line_pack_keys_are_line_ids():
    with PackReader(LINE_PACK) as reader:
        keys = reader.keys[:50]
    assert all(re.fullmatch(r"\d{4}-\d+-\d+", key) for key in keys), keys[:5]


@needs_line_pack
def test_the_line_pack_covers_every_held_out_writer():
    """The evaluation reports a number per held-out writer. One missing writer
    would shrink the population without saying so."""
    from nib.data.split import WriterSplit

    split_path = find_repo_root() / "configs" / "splits" / "cvl-writer-disjoint.json"
    if not split_path.is_file():
        pytest.skip("no committed split")

    with PackReader(LINE_PACK) as reader:
        in_pack = set(reader.writers())
    held_out = set(WriterSplit.load(split_path).writers["test"])

    assert held_out <= in_pack, f"missing from the pack: {sorted(held_out - in_pack)}"
