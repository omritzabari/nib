"""Tests for the writer-disjoint split.

The property under test is the one the project's central claim depends on: a
writer never appears on both sides. Everything else here is about making that
property reproducible and hard to break by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from nib.data.split import (
    SplitError,
    WriterSplit,
    counts_from_records,
    make_split,
)


@dataclass
class FakeRecord:
    writer_id: str
    value: int = 0


def even_counts(n_writers: int, per_writer: int = 10) -> dict[str, int]:
    return {f"{i:03d}": per_writer for i in range(n_writers)}


def uneven_counts(n_writers: int) -> dict[str, int]:
    """Writers differ a lot in productivity -- the case that breaks naive splitting."""
    return {f"{i:03d}": 1 + (i * 7) % 40 for i in range(n_writers)}


# --------------------------------------------------------------------------
# the property that matters
# --------------------------------------------------------------------------


def test_no_writer_appears_in_two_splits():
    split = make_split(uneven_counts(310), seed=1337)
    train = set(split.writers["train"])
    test = set(split.writers["test"])
    assert not (train & test), "writer leak between train and test"
    assert train and test


def test_every_writer_is_assigned_exactly_once():
    counts = uneven_counts(310)
    split = make_split(counts)
    assigned = [w for ids in split.writers.values() for w in ids]
    assert sorted(assigned) == sorted(counts)
    assert len(assigned) == len(set(assigned))


def test_construction_rejects_an_overlapping_split():
    """The guard fires even if a split is built by hand or loaded from a bad file."""
    with pytest.raises(SplitError, match="more than one split"):
        WriterSplit(
            name="broken",
            seed=0,
            ratios={"train": 0.5, "test": 0.5},
            writers={"train": ["001", "002"], "test": ["002", "003"]},
        )


def test_construction_rejects_a_duplicated_writer():
    with pytest.raises(SplitError, match="twice"):
        WriterSplit(
            name="broken",
            seed=0,
            ratios={"train": 1.0},
            writers={"train": ["001", "001"]},
        )


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_the_same_split():
    counts = uneven_counts(200)
    assert make_split(counts, seed=7).writers == make_split(counts, seed=7).writers


def test_different_seeds_give_different_splits():
    counts = uneven_counts(200)
    assert make_split(counts, seed=1).writers != make_split(counts, seed=2).writers


def test_input_dict_ordering_does_not_change_the_result():
    """A split that depended on dict insertion order would silently differ between
    a fresh parse and a reload."""
    counts = uneven_counts(150)
    shuffled = dict(reversed(list(counts.items())))
    assert make_split(counts, seed=5).writers == make_split(shuffled, seed=5).writers


# --------------------------------------------------------------------------
# balance
# --------------------------------------------------------------------------


def test_sample_shares_land_near_the_requested_ratios():
    """Splitting writers 70/30 does not split samples 70/30 when writers differ in
    productivity. This asserts the balancing actually works."""
    counts = uneven_counts(310)
    split = make_split(counts, ratios={"train": 0.7, "test": 0.3}, seed=1337)
    total = sum(counts.values())
    for name, target in [("train", 0.7), ("test", 0.3)]:
        share = sum(counts[w] for w in split.writers[name]) / total
        assert abs(share - target) < 0.02, f"{name} got {share:.1%}, wanted {target:.0%}"


def test_three_way_split():
    counts = uneven_counts(310)
    split = make_split(counts, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=3)
    assert set(split.writers) == {"train", "val", "test"}
    assert sum(len(v) for v in split.writers.values()) == len(counts)


def test_iam_scale_split_matches_the_planned_shape():
    """657 writers split 70/30 should land near the 340/160-ish shape the project
    brief describes for IAM."""
    split = make_split(even_counts(657), ratios={"train": 0.7, "test": 0.3}, seed=1337)
    assert 440 <= len(split.writers["train"]) <= 480
    assert 175 <= len(split.writers["test"]) <= 215


# --------------------------------------------------------------------------
# bad input
# --------------------------------------------------------------------------


def test_ratios_must_sum_to_one():
    with pytest.raises(SplitError, match="sum to 1"):
        make_split(even_counts(10), ratios={"train": 0.7, "test": 0.4})


def test_ratios_must_be_positive():
    with pytest.raises(SplitError, match="positive"):
        make_split(even_counts(10), ratios={"train": 1.2, "test": -0.2})


def test_no_writers_raises():
    with pytest.raises(SplitError, match="no writers"):
        make_split({})


def test_too_few_writers_raises_rather_than_producing_an_empty_split():
    with pytest.raises(SplitError, match="cannot divide"):
        make_split({"001": 5}, ratios={"train": 0.5, "test": 0.5})


# --------------------------------------------------------------------------
# partitioning records
# --------------------------------------------------------------------------


def test_partition_sends_each_record_to_its_writer_side():
    counts = even_counts(50)
    split = make_split(counts, seed=1)
    records = [FakeRecord(w) for w, n in counts.items() for _ in range(n)]

    parts = split.partition(records)
    assert sum(len(v) for v in parts.values()) == len(records)
    for name, group in parts.items():
        assert {r.writer_id for r in group} <= set(split.writers[name])


def test_partition_raises_on_a_writer_the_split_has_never_heard_of():
    """A stale split file silently dropping samples is exactly the failure mode
    this project refuses to allow."""
    split = make_split(even_counts(10), seed=1)
    records = [FakeRecord("999")]
    with pytest.raises(SplitError, match="not in split"):
        split.partition(records)


def test_partition_can_drop_unknown_writers_when_told_to():
    split = make_split(even_counts(10), seed=1)
    records = [FakeRecord("999"), FakeRecord("001")]
    parts = split.partition(records, strict=False)
    assert sum(len(v) for v in parts.values()) == 1


def test_counts_from_records():
    records = [FakeRecord("a"), FakeRecord("a"), FakeRecord("b")]
    assert counts_from_records(records) == {"a": 2, "b": 1}


# --------------------------------------------------------------------------
# persistence -- the split is a committed artefact
# --------------------------------------------------------------------------


def test_saved_split_reloads_identically(tmp_path):
    original = make_split(uneven_counts(120), seed=42, name="cvl")
    path = original.save(tmp_path / "split.json")
    reloaded = WriterSplit.load(path)
    assert reloaded.writers == original.writers
    assert reloaded.seed == original.seed
    assert reloaded.ratios == original.ratios
    assert reloaded.name == original.name


def test_saved_file_is_human_readable_and_stable(tmp_path):
    """It gets committed and diffed, so it must be sorted and not reorder itself
    between saves."""
    split = make_split(uneven_counts(60), seed=9)
    first = split.save(tmp_path / "a.json").read_text(encoding="utf-8")
    second = split.save(tmp_path / "b.json").read_text(encoding="utf-8")
    assert first == second

    data = json.loads(first)
    assert data["writers"]["train"] == sorted(data["writers"]["train"])
    assert data["counts"]["train"] == len(data["writers"]["train"])


def test_summary_reports_both_writer_and_sample_shares():
    counts = uneven_counts(100)
    split = make_split(counts, seed=1)
    text = split.summary(sample_counts=counts)
    assert "writers" in text and "samples" in text
