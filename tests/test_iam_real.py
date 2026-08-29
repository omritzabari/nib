"""Validation against genuine IAM data. Skipped until the data is present.

This is the safety net for the one real unknown in the data layer: the parser and
the fixture were both written against a *reconstruction* of IAM's XML schema,
because the FKI site was unreachable at the time. These tests do nothing until
real IAM XML appears under ``data/raw/iam``, and then run automatically and say
precisely where the reconstruction was wrong.

To activate them, put the extracted archives here::

    data/raw/iam/xml/*.xml          from xml.tgz
    data/raw/iam/words/...          from words.tgz     (optional for most tests)
    data/raw/iam/ascii/words.txt    from ascii.tgz     (optional)

Then just run pytest. Nothing else needs changing.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from nib.config import find_repo_root
from nib.data.iam_parser import group_by_writer, parse_dataset, validate_schema

IAM_ROOT = find_repo_root() / "data" / "raw" / "iam"
XML_FILES = sorted((IAM_ROOT / "xml").glob("*.xml")) if (IAM_ROOT / "xml").is_dir() else []

# Published figures for the IAM Handwriting Database, used as sanity bounds
# rather than exact assertions -- filtering choices legitimately change counts.
EXPECTED_WRITERS = 657
EXPECTED_FORMS = 1539

needs_iam = pytest.mark.skipif(
    not XML_FILES,
    reason=(
        f"no IAM XML found under {IAM_ROOT}. "
        "These checks activate on their own once xml.tgz is extracted there."
    ),
)


def test_report_whether_iam_is_present():
    """Always runs, so the suite says out loud whether the real check happened."""
    if XML_FILES:
        print(f"\nIAM present: {len(XML_FILES)} XML files under {IAM_ROOT}")
    else:
        archives = sorted(IAM_ROOT.glob("*.tgz")) if IAM_ROOT.is_dir() else []
        if archives:
            print(f"\nIAM archives found but not extracted: {[a.name for a in archives]}")
        else:
            print(f"\nIAM not present at {IAM_ROOT} -- real-data checks skipped")


@needs_iam
def test_real_schema_matches_the_reconstruction():
    """The headline question. If this fails, the message names the difference and
    the parser -- not the real data -- is what needs to change."""
    for xml_path in XML_FILES[:25]:
        validate_schema(xml_path)


@needs_iam
def test_real_forms_carry_distinct_writer_ids():
    writers = {ET.parse(p).getroot().get("writer-id") for p in XML_FILES}
    assert None not in writers, "some real forms have no writer-id"
    assert len(writers) > 1


@needs_iam
def test_real_dataset_parses_to_plausible_totals():
    records, report = parse_dataset(IAM_ROOT, require_image=_words_present())
    print("\n" + report.summary())

    assert report.forms == pytest.approx(EXPECTED_FORMS, rel=0.1), (
        f"parsed {report.forms} forms, expected around {EXPECTED_FORMS}"
    )
    assert len(report.writers) == pytest.approx(EXPECTED_WRITERS, rel=0.05), (
        f"found {len(report.writers)} writers, expected around {EXPECTED_WRITERS}"
    )
    assert records, "parsed zero usable words from real IAM"


@needs_iam
def test_real_drop_rate_is_not_alarming():
    """Some loss is expected -- bad segmentation, out-of-charset characters. Losing
    most of the dataset means a bug, not a filter."""
    _, report = parse_dataset(IAM_ROOT, require_image=_words_present())
    kept_share = report.kept / report.total_seen
    assert kept_share > 0.5, f"kept only {kept_share:.1%} of words. Breakdown:\n{report.summary()}"


@needs_iam
def test_real_writers_have_enough_words_for_few_shot():
    """Style conditioning needs several sample words per writer. A writer with two
    usable words cannot be styled from, and should be found now rather than as a
    crash during evaluation."""
    records, _ = parse_dataset(IAM_ROOT, require_image=_words_present())
    grouped = group_by_writer(records)
    thin = {w: len(r) for w, r in grouped.items() if len(r) < 15}
    assert len(thin) < 0.1 * len(grouped), (
        f"{len(thin)} of {len(grouped)} writers have fewer than 15 usable words: "
        f"{dict(list(thin.items())[:10])}"
    )


def _words_present() -> bool:
    directory: Path = IAM_ROOT / "words"
    return directory.is_dir() and any(directory.rglob("*.png"))
