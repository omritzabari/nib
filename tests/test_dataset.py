"""Tests for the style-conditioned dataset and its collate.

Almost everything here guards a failure that does not raise. A style reference
drawn from the wrong writer, a target word appearing among its own references, a
black bar painted by padding -- each of them trains happily and produces a model
that is wrong in a way the loss curve will not show.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.data import charset as cs
from nib.data.dataset import (
    PAD_VALUE,
    Batch,
    DatasetError,
    Sample,
    WordDataset,
    collate,
)
from nib.data.pack import PackedSample, PackHeader, PackReader, PackWriter
from nib.data.split import WriterSplit

torch = pytest.importorskip("torch", reason="torch is an optional extra")

WORDS = ["Imagine", "a", "vast", "sheet", "of", "paper", "on", "which", "straight", "Lines"]


def make_pack(tmp_path, writers=4, per_writer=10, height=64):
    path = tmp_path / "p.lmdb"
    rng = np.random.default_rng(0)
    with PackWriter(path, PackHeader(height=height)) as writer:
        for w in range(writers):
            for i in range(per_writer):
                writer.add(
                    PackedSample(
                        key=f"{w:04d}-1-0-{i}",
                        writer_id=f"{w:04d}",
                        text=WORDS[i % len(WORDS)],
                        split="testset",
                        image=rng.integers(0, 255, (height, 30 + 11 * i), dtype=np.uint8),
                    )
                )
    return PackReader(path)


# ---------------------------------------------------------------------------
# the property the whole file exists for
# ---------------------------------------------------------------------------


def test_style_references_come_from_the_same_writer(tmp_path):
    """Wrong-writer references make the style input noise and the model learns to
    ignore it. Nothing raises; the samples simply stop being conditioned.

    An earlier version of this test compared image shapes and asserted nothing
    about writers -- it passed while checking nothing. Sample now carries the
    reference keys so the property can actually be read.
    """
    data = WordDataset(make_pack(tmp_path), num_style_refs=3)
    for index in range(len(data)):
        sample = data[index]
        assert len(sample.style_keys) == 3
        for ref_key in sample.style_keys:
            assert data.writer_of[ref_key] == sample.writer_id, (
                f"{sample.key} (writer {sample.writer_id}) was given a reference "
                f"from writer {data.writer_of[ref_key]}"
            )


def test_the_target_word_never_appears_among_its_own_references(tmp_path):
    """The most important assertion in the project's data layer. If a word can be
    its own style reference, the model copies rather than generalises: the loss
    drops, the samples look excellent, and the few-shot claim is hollow."""
    pack = make_pack(tmp_path, writers=3, per_writer=12)
    data = WordDataset(pack, num_style_refs=4, seed=7)

    for index in range(len(data)):
        sample = data[index]
        assert sample.key not in sample.style_keys, (
            f"{sample.key} was used as its own style reference"
        )


def test_every_sample_has_the_requested_number_of_references(tmp_path):
    data = WordDataset(make_pack(tmp_path), num_style_refs=5)
    for index in range(0, len(data), 7):
        assert len(data[index].style) == 5


def test_reference_choice_is_reproducible(tmp_path):
    """Seeded per key, not from a shared generator, so an epoch is identical
    whether the loader runs one worker or eight."""
    pack = make_pack(tmp_path)
    a = WordDataset(pack, num_style_refs=3, seed=11)[4]
    b = WordDataset(pack, num_style_refs=3, seed=11)[4]
    assert all(np.array_equal(x, y) for x, y in zip(a.style, b.style, strict=True))

    c = WordDataset(pack, num_style_refs=3, seed=12)[4]
    assert not all(np.array_equal(x, y) for x, y in zip(a.style, c.style, strict=True))


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------


def test_only_writers_from_the_requested_split_are_served(tmp_path):
    pack = make_pack(tmp_path, writers=6)
    split = WriterSplit(
        name="t",
        seed=0,
        ratios={"train": 0.5, "test": 0.5},
        writers={"train": ["0000", "0001", "0002"], "test": ["0003", "0004", "0005"]},
    )
    train = WordDataset(pack, writer_split=split, split="train")
    test = WordDataset(pack, writer_split=split, split="test")

    assert train.writers == ["0000", "0001", "0002"]
    assert test.writers == ["0003", "0004", "0005"]
    assert not set(train.keys) & set(test.keys), "a key is served by both sides"


def test_an_unknown_split_name_raises(tmp_path):
    split = WriterSplit("t", 0, {"train": 1.0}, {"train": ["0000"]})
    with pytest.raises(DatasetError, match="not in"):
        WordDataset(make_pack(tmp_path), writer_split=split, split="validation")


def test_a_split_naming_writers_the_pack_lacks_raises(tmp_path):
    split = WriterSplit("t", 0, {"train": 1.0}, {"train": ["9999"]})
    with pytest.raises(DatasetError, match="no writers"):
        WordDataset(make_pack(tmp_path), writer_split=split, split="train")


# ---------------------------------------------------------------------------
# writers with too little data
# ---------------------------------------------------------------------------


def test_writers_with_too_few_words_are_dropped_and_reported(tmp_path):
    """A writer with three words cannot supply five references. Dropping them here
    beats an IndexError at step 40,000 of a training run."""
    path = tmp_path / "p.lmdb"
    rng = np.random.default_rng(1)
    with PackWriter(path, PackHeader(height=64)) as writer:
        for w, count in [("0001", 20), ("0002", 2)]:
            for i in range(count):
                writer.add(
                    PackedSample(
                        f"{w}-1-0-{i}",
                        w,
                        "word",
                        "testset",
                        rng.integers(0, 255, (64, 40), dtype=np.uint8),
                    )
                )

    data = WordDataset(PackReader(path), num_style_refs=5)
    assert data.writers == ["0001"]
    assert data.dropped_writers == {"0002": 2}
    assert "0002" in data.summary()


def test_a_pack_where_no_writer_qualifies_raises(tmp_path):
    with pytest.raises(DatasetError, match="none can be styled from"):
        WordDataset(make_pack(tmp_path, writers=2, per_writer=2), num_style_refs=5)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def test_labels_encode_the_transcription(tmp_path):
    data = WordDataset(make_pack(tmp_path))
    sample = data[0]
    assert data.charset.decode(sample.label) == sample.text


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------


def sample(width=40, height=64, text="hi", refs=2, ref_width=30, writer="0001"):
    alphabet = cs.get("english")
    return Sample(
        key=f"k{width}",
        writer_id=writer,
        text=text,
        image=np.full((height, width), 0.5, dtype=np.float32),
        label=alphabet.encode(text),
        style=[np.full((height, ref_width), 0.5, dtype=np.float32) for _ in range(refs)],
        style_keys=[f"ref{i}" for i in range(refs)],
    )


def test_padding_is_white_not_black(tmp_path):
    """Zero is solid ink in this encoding. Padding with zeros paints a black bar
    beside every short word, and the model learns to draw it."""
    batch = collate([sample(width=20), sample(width=60)])
    assert PAD_VALUE == 1.0
    padded_region = batch.images[0, 0, :, 20:]
    assert torch.all(padded_region == 1.0), "padding is not white"


def test_masks_mark_exactly_the_real_pixels():
    batch = collate([sample(width=20), sample(width=60)])
    assert batch.images.shape == (2, 1, 64, 60)
    assert batch.image_mask[0].sum().item() == 20
    assert batch.image_mask[1].sum().item() == 60
    assert batch.image_widths.tolist() == [20, 60]


def test_labels_are_padded_with_the_pad_index():
    batch = collate([sample(text="hi"), sample(width=41, text="longer")])
    assert batch.labels.shape == (2, 6)
    assert batch.labels[0, 2:].tolist() == [cs.PAD_INDEX] * 4
    assert batch.label_lengths.tolist() == [2, 6]


def test_style_references_are_padded_and_masked():
    a = sample(refs=3, ref_width=25)
    b = sample(width=41, refs=3, ref_width=70)
    batch = collate([a, b])
    assert batch.style.shape == (2, 3, 1, 64, 70)
    assert batch.style_mask[0, 0].sum().item() == 25
    assert batch.style_mask[1, 0].sum().item() == 70
    assert torch.all(batch.style[0, 0, 0, :, 25:] == 1.0)


def test_metadata_survives_collation():
    batch = collate([sample(writer="0001"), sample(width=41, writer="0002")])
    assert batch.writer_ids == ["0001", "0002"]
    assert len(batch) == 2
    assert isinstance(batch, Batch)


def test_an_empty_batch_raises():
    with pytest.raises(DatasetError, match="empty batch"):
        collate([])


def test_mismatched_heights_raise():
    with pytest.raises(DatasetError, match="same height"):
        collate([sample(height=64), sample(width=41, height=32)])


def test_mismatched_reference_counts_raise():
    with pytest.raises(DatasetError, match="same number of style refs"):
        collate([sample(refs=2), sample(width=41, refs=3)])


# ---------------------------------------------------------------------------
# end to end through a DataLoader
# ---------------------------------------------------------------------------


def test_a_dataloader_iterates_the_whole_dataset(tmp_path):
    from torch.utils.data import DataLoader

    data = WordDataset(make_pack(tmp_path, writers=4, per_writer=10), num_style_refs=3)
    loader = DataLoader(data, batch_size=6, shuffle=False, collate_fn=collate, num_workers=0)

    seen = []
    for batch in loader:
        assert batch.images.shape[2] == data.height
        assert batch.style.shape[1] == 3
        seen.extend(batch.keys)
    assert sorted(seen) == sorted(data.keys)
