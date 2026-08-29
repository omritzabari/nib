"""Tests for the IAM parser, run against the synthetic fixture.

The failure modes worth guarding are the quiet ones: a writer id that comes back
empty, a word joined to the wrong image, or samples vanishing without anyone
noticing why. None of those raise on their own.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from nib.data import fixture
from nib.data.iam_parser import (
    DROP_BAD_SEGMENTATION,
    DROP_EMPTY_TEXT,
    DROP_MISSING_IMAGE,
    DROP_OUT_OF_CHARSET,
    SchemaMismatch,
    group_by_writer,
    parse_dataset,
    validate_schema,
    word_image_path,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("iam")
    summary = fixture.build(root=root, num_writers=6, words_per_writer=30, height=64, seed=0)
    records, report = parse_dataset(root)
    return summary, records, report


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_parses_records_with_every_field_populated(dataset):
    _, records, _ = dataset
    assert records
    for record in records[:50]:
        assert record.word_id
        assert record.form_id
        assert record.line_id
        assert record.writer_id, "empty writer id would silently break the split"
        assert record.text
        assert record.image_path.is_file()


def test_every_record_points_at_its_own_image(dataset):
    """A record joined to the wrong image mislabels a training sample and nothing
    raises. Check the id embedded in the path matches the record."""
    _, records, _ = dataset
    for record in records:
        assert record.image_path.stem == record.word_id
        assert record.image_path.parent.name == record.form_id


def test_all_writers_are_found(dataset):
    summary, _, report = dataset
    assert len(report.writers) == summary.num_writers
    assert report.forms == summary.num_writers


def test_ordering_is_stable_not_filesystem_dependent(dataset):
    """Two parses of the same directory must agree, or runs are not reproducible."""
    summary, records, _ = dataset
    again, _ = parse_dataset(summary.root)
    assert [r.word_id for r in records] == [r.word_id for r in again]
    assert [r.word_id for r in records] == sorted(r.word_id for r in records)


def test_group_by_writer_partitions_everything(dataset):
    _, records, _ = dataset
    grouped = group_by_writer(records)
    assert sum(len(v) for v in grouped.values()) == len(records)
    for writer_id, group in grouped.items():
        assert all(r.writer_id == writer_id for r in group)


# --------------------------------------------------------------------------
# exclusions -- counted, never silent
# --------------------------------------------------------------------------


def test_bad_segmentation_lines_are_dropped_and_counted(dataset):
    """The fixture marks about one line in ten as segmentation=err, mirroring IAM.
    Those words must be excluded and the exclusion must appear in the report."""
    _, _, report = dataset
    assert report.dropped[DROP_BAD_SEGMENTATION] > 0
    assert report.kept < report.total_seen


def test_bad_segmentation_can_be_kept_on_request(dataset):
    summary, _, report = dataset
    kept_all, report_all = parse_dataset(summary.root, skip_bad_segmentation=False)
    assert len(kept_all) > report.kept
    assert report_all.dropped[DROP_BAD_SEGMENTATION] == 0


def test_report_accounts_for_every_word_seen(dataset):
    """kept + dropped must equal what was in the XML. If they do not, samples are
    disappearing somewhere unaccounted for."""
    summary, _, report = dataset
    in_xml = sum(
        len(ET.parse(summary.root / "xml" / f"{f}.xml").getroot().findall(".//word"))
        for f in summary.forms
    )
    assert report.total_seen == in_xml


def test_out_of_charset_words_are_dropped(tmp_path):
    fixture.build(root=tmp_path, num_writers=1, words_per_writer=10, seed=0)
    xml_path = next((tmp_path / "xml").glob("*.xml"))
    tree = ET.parse(xml_path)
    word = tree.getroot().find(".//word")
    word.set("text", "café")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    _, report = parse_dataset(tmp_path)
    assert report.dropped[DROP_OUT_OF_CHARSET] == 1


def test_empty_transcriptions_are_dropped(tmp_path):
    fixture.build(root=tmp_path, num_writers=1, words_per_writer=10, seed=0)
    xml_path = next((tmp_path / "xml").glob("*.xml"))
    tree = ET.parse(xml_path)
    tree.getroot().find(".//word").set("text", "   ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    _, report = parse_dataset(tmp_path)
    assert report.dropped[DROP_EMPTY_TEXT] == 1


def test_missing_images_are_dropped_not_returned_as_broken_records(tmp_path):
    """Real IAM has records whose image is absent. Returning them would fail much
    later, inside the data loader, mid-training."""
    summary = fixture.build(root=tmp_path, num_writers=1, words_per_writer=10, seed=0)
    records, _ = parse_dataset(tmp_path)
    victim = records[0]
    victim.image_path.unlink()

    survivors, report = parse_dataset(tmp_path)
    assert report.dropped[DROP_MISSING_IMAGE] == 1
    assert victim.word_id not in {r.word_id for r in survivors}
    assert summary.num_words > 0


def test_report_summary_is_readable(dataset):
    _, _, report = dataset
    text = report.summary()
    assert "writers" in text
    assert "words kept" in text
    assert DROP_BAD_SEGMENTATION in text


# --------------------------------------------------------------------------
# schema validation -- the guard against the reconstruction being wrong
# --------------------------------------------------------------------------


def test_fixture_passes_schema_validation(dataset):
    summary, _, _ = dataset
    validate_schema(summary.root / "xml" / f"{summary.forms[0]}.xml")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("drop_writer_id", "writer-id"),
        ("rename_root", "expected <form>"),
        ("drop_words", "no <word> elements"),
        ("drop_word_text", "no text attribute"),
    ],
)
def test_schema_problems_are_reported_specifically(tmp_path, mutation, expected):
    """Each way the real schema could differ produces a message naming the actual
    difference, not a generic parse failure."""
    fixture.build(root=tmp_path, num_writers=1, words_per_writer=10, seed=0)
    xml_path = next((tmp_path / "xml").glob("*.xml"))
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if mutation == "drop_writer_id":
        del root.attrib["writer-id"]
    elif mutation == "rename_root":
        root.tag = "document"
    elif mutation == "drop_words":
        for line in root.findall(".//line"):
            for word in line.findall("word"):
                line.remove(word)
    elif mutation == "drop_word_text":
        del root.find(".//word").attrib["text"]

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    with pytest.raises(SchemaMismatch, match=expected):
        validate_schema(xml_path)


def test_malformed_xml_raises_schema_mismatch(tmp_path):
    (tmp_path / "xml").mkdir()
    bad = tmp_path / "xml" / "broken.xml"
    bad.write_text("<form><line></form>", encoding="utf-8")
    with pytest.raises(SchemaMismatch, match="not well-formed"):
        validate_schema(bad)


def test_missing_directories_fail_with_a_useful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="no xml/ directory"):
        parse_dataset(tmp_path)

    (tmp_path / "xml").mkdir()
    with pytest.raises(FileNotFoundError, match=r"no \.xml files"):
        parse_dataset(tmp_path)


def test_word_image_path_follows_iam_nesting():
    from pathlib import Path

    path = word_image_path(Path("/root"), "a01-000u", "a01-000u-00-01")
    assert path.as_posix().endswith("words/a01/a01-000u/a01-000u-00-01.png")
