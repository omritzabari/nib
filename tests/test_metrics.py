"""Tests for the three metrics.

None of these need a downloaded model: the recogniser and the embedder are behind
protocols, so the arithmetic and the failure behaviour can be checked exactly.
That is the point of the protocols -- a metric whose correctness can only be
assessed by running a 1.4 GB model is a metric nobody checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.engine.metrics import cer as cer_mod
from nib.engine.metrics import writer as writer_mod
from nib.engine.metrics.cer import (
    CerResult,
    corpus_cer,
    edit_distance,
)
from nib.engine.metrics.fid import (
    FidError,
    compute_fid,
    frechet_distance,
    gaussian_statistics,
)
from nib.engine.metrics.writer import RetrievalError, WriterRetrieval

# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "abd", 1),  # substitution
        ("abc", "ab", 1),  # deletion
        ("ab", "abc", 1),  # insertion
        ("", "abc", 3),
        ("kitten", "sitting", 3),  # the textbook case
    ],
)
def test_edit_distance(a, b, expected):
    assert edit_distance(a, b) == expected
    assert edit_distance(b, a) == expected, "edit distance must be symmetric"


def test_cer_is_normalised_by_the_target_not_the_prediction():
    """Normalising by the prediction would let a model score perfectly by
    emitting nothing at all."""
    assert cer_mod.cer("", "target") == 1.0
    assert cer_mod.cer("target", "target") == 0.0


def test_corpus_cer_weights_by_characters_not_by_sample():
    """Averaging per-sample rates gives a one-character word the same weight as a
    twelve-character one, so a few short words swing the figure -- and short words
    are where a recogniser errs most."""
    predictions = ["a", "handwriting"]  # 1 of 1 wrong, 0 of 11 wrong
    targets = ["b", "handwriting"]
    assert corpus_cer(predictions, targets) == pytest.approx(1 / 12)

    per_sample_mean = (1.0 + 0.0) / 2
    assert corpus_cer(predictions, targets) != pytest.approx(per_sample_mean)


def test_corpus_cer_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="predictions for"):
        corpus_cer(["a"], ["a", "b"])


def test_corpus_cer_rejects_an_empty_set():
    with pytest.raises(ValueError, match="empty set"):
        corpus_cer([], [])


class FakeRecogniser:
    """Reads perfectly, except it drops the last character of every word."""

    def __init__(self, mangle: bool = True):
        self.mangle = mangle
        self.texts: list[str] = []
        self.calls = 0

    def read(self, images):
        self.calls += 1
        out = []
        for image in images:
            text = self.texts[int(image[0, 0])]
            out.append(text[:-1] if self.mangle and text else text)
        return out


def images_for(texts):
    return [np.full((8, 8), i, dtype=np.uint8) for i in range(len(texts))]


def test_evaluate_reports_a_baseline_alongside_the_figure():
    """A CER figure without the recogniser's own error rate beside it is not
    interpretable: 9% could be excellent or dreadful."""
    texts = ["hello", "world", "handwriting"]
    recogniser = FakeRecogniser()
    recogniser.texts = texts

    result = cer_mod.evaluate(
        recogniser,
        generated_images=images_for(texts),
        targets=texts,
        real_images=images_for(texts),
        real_targets=texts,
    )
    assert result.generated > 0
    assert result.real == pytest.approx(result.generated)
    assert result.gap == pytest.approx(0.0)
    assert "gap" in result.summary()


def test_a_result_without_a_baseline_says_so():
    result = CerResult(generated=0.09, num_samples=10)
    assert result.gap is None
    assert "no baseline" in result.summary()


def test_evaluation_batches_without_losing_samples():
    texts = [f"word{i}" for i in range(70)]
    recogniser = FakeRecogniser(mangle=False)
    recogniser.texts = texts
    result = cer_mod.evaluate(recogniser, images_for(texts), texts, batch_size=16)
    assert result.generated == 0.0
    assert result.num_samples == 70
    assert recogniser.calls == 5


def test_a_recogniser_returning_the_wrong_count_raises():
    class Broken:
        def read(self, images):
            return ["only one"]

    with pytest.raises(RuntimeError, match="returned 1 readings"):
        cer_mod.evaluate(Broken(), images_for(["a", "b"]), ["a", "b"])


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------


def features(n=200, d=16, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(shift, 1.0, (n, d))


def test_fid_of_a_distribution_against_itself_is_near_zero():
    """The sanity check that says the implementation is right at all."""
    f = features(seed=1)
    assert compute_fid(f, f).value == pytest.approx(0.0, abs=1e-6)


def test_fid_grows_as_the_distributions_separate():
    real = features(seed=2)
    near = compute_fid(real, features(shift=0.5, seed=3)).value
    far = compute_fid(real, features(shift=3.0, seed=4)).value
    assert 0 < near < far


def test_fid_is_never_negative():
    """Numerical error in the matrix square root can push it slightly below zero,
    and a negative distance in a README is embarrassing."""
    assert compute_fid(features(seed=5), features(seed=6)).value >= 0.0


def test_fid_refuses_a_sample_count_too_small_to_mean_anything():
    """FID is strongly biased on small sets. A loud refusal beats a number that
    ends up in a README."""
    with pytest.raises(FidError, match="strongly biased"):
        compute_fid(features(n=10), features(n=10))


def test_mismatched_feature_dimensions_raise():
    with pytest.raises(FidError, match="dimensions differ"):
        compute_fid(features(d=16), features(d=32))


def test_the_sample_count_is_part_of_the_result():
    """FID falls as sample count rises, so two runs are only comparable at equal
    counts -- which means the count has to travel with the number."""
    result = compute_fid(features(n=200), features(n=150))
    assert result.num_real == 200
    assert result.num_generated == 150
    assert "150 generated vs 200 real" in result.summary()


def test_gaussian_statistics_shapes():
    mu, sigma = gaussian_statistics(features(n=100, d=8))
    assert mu.shape == (8,)
    assert sigma.shape == (8, 8)


def test_statistics_need_more_than_one_sample():
    with pytest.raises(FidError, match="at least two"):
        gaussian_statistics(np.zeros((1, 4)))


def test_frechet_distance_between_identical_gaussians_is_zero():
    mu = np.zeros(4)
    sigma = np.eye(4)
    value, residual = frechet_distance(mu, sigma, mu, sigma)
    assert value == pytest.approx(0.0, abs=1e-9)
    assert residual < 1e-6


def test_frechet_distance_matches_the_closed_form_for_diagonal_gaussians():
    """An independent check of the formula, not just of its self-consistency.
    For N(0, I) and N(m, I) the distance is exactly |m| squared."""
    d = 6
    mu_a, mu_b = np.zeros(d), np.full(d, 2.0)
    identity = np.eye(d)
    value, _ = frechet_distance(mu_a, identity, mu_b, identity)
    assert value == pytest.approx(float(mu_b.dot(mu_b)), rel=1e-6)


# ---------------------------------------------------------------------------
# writer retrieval
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Embeds an image as its first eight values. Images built by `styled` below
    therefore cluster by writer, deterministically."""

    def __call__(self, images):
        return np.stack([np.asarray(im, dtype=np.float64).reshape(-1)[:8] for im in images])


def styled(writer_index: int, count: int, dim: int = 8, seed: int = 0):
    """Images whose first row is characteristic of the writer.

    Each writer gets a distinct *direction*, not a distinct magnitude. Retrieval
    compares by cosine, so writers separated only by scale would be identical to
    it -- and an earlier version of this fixture put writer 0 at the origin, where
    cosine similarity has no meaning at all. That produced a 96% top-1 score and
    looked like a bug in the metric rather than in the test data.
    """
    rng = np.random.default_rng(seed + writer_index)
    base = np.zeros(dim)
    base[writer_index % dim] = 1.0
    base[(writer_index + 3) % dim] = 0.5
    return [base + rng.normal(0, 0.01, dim) for _ in range(count)]


def gallery_data(writers=5, per_writer=6):
    images, ids = [], []
    for w in range(writers):
        images.extend(styled(w, per_writer))
        ids.extend([f"{w:04d}"] * per_writer)
    return images, ids


def test_retrieval_finds_the_right_writer_when_style_is_separable():
    images, ids = gallery_data()
    retrieval = WriterRetrieval(FakeEmbedder()).fit(images, ids)
    result = retrieval.evaluate(images, ids)
    assert result.top1 == 1.0
    assert result.num_writers == 5


def test_retrieval_is_near_chance_when_style_carries_no_information():
    """The other direction, and the one that matters: if the generator ignored its
    style input, this metric must say so rather than flattering it."""
    rng = np.random.default_rng(3)
    images = [rng.normal(0, 1, 8) for _ in range(120)]
    ids = [f"{i % 6:04d}" for i in range(120)]

    retrieval = WriterRetrieval(FakeEmbedder()).fit(images, ids)
    result = retrieval.evaluate(images, ids)
    assert result.top1 < 0.5, (
        f"top-1 of {result.top1:.0%} on noise; the metric is not measuring style"
    )


def test_chance_is_reported_beside_the_score():
    """40% top-1 among 94 writers is a strong result. Without the 1% chance level
    beside it, nobody can tell."""
    images, ids = gallery_data(writers=4)
    result = WriterRetrieval(FakeEmbedder()).fit(images, ids).evaluate(images, ids)
    assert result.chance == pytest.approx(0.25)
    assert result.lift_over_chance == pytest.approx(0.75)
    assert "chance" in result.summary()


def test_topk_is_at_least_top1():
    images, ids = gallery_data(writers=6)
    result = WriterRetrieval(FakeEmbedder()).fit(images, ids).evaluate(images, ids, k=3)
    assert result.topk >= result.top1
    assert result.k == 3


def test_k_is_clamped_to_the_gallery_size():
    images, ids = gallery_data(writers=3)
    result = WriterRetrieval(FakeEmbedder()).fit(images, ids).evaluate(images, ids, k=99)
    assert result.k == 3
    assert result.topk == 1.0


def test_evaluating_before_fitting_raises():
    with pytest.raises(RetrievalError, match="fit\\(\\) must be called"):
        WriterRetrieval(FakeEmbedder()).evaluate([np.zeros((4, 8))], ["0000"])


def test_a_query_writer_absent_from_the_gallery_is_refused():
    """Unanswerable rather than wrong. Counting it as a miss would depress the
    score for a reason that has nothing to do with style."""
    images, ids = gallery_data(writers=3)
    retrieval = WriterRetrieval(FakeEmbedder()).fit(images, ids)
    with pytest.raises(RetrievalError, match="not in the gallery"):
        retrieval.evaluate(styled(9, 2), ["9999", "9999"])


def test_per_writer_scores_are_reported():
    """An average hides a model that nails five writers and fails ninety."""
    images, ids = gallery_data(writers=4)
    result = WriterRetrieval(FakeEmbedder()).fit(images, ids).evaluate(images, ids)
    assert set(result.per_writer) == {"0000", "0001", "0002", "0003"}


def test_an_embedder_returning_the_wrong_shape_raises():
    class Broken:
        def __call__(self, images):
            return np.zeros((len(images) + 1, 4))

    with pytest.raises(RetrievalError, match="expected"):
        WriterRetrieval(Broken()).fit([np.ones(8)] * 3, ["a", "b", "c"])


def test_embeddings_are_length_normalised():
    """Without it, a writer whose embeddings happen to be larger in magnitude wins
    every comparison regardless of style."""
    scaled = writer_mod._normalise(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(scaled, axis=1), 1.0)


# ---------------------------------------------------------------------------
# reference numbers
#
# The three baselines are properties of a pack, not of the project. These tests
# guard the one thing that would silently invalidate every comparison: two packs
# sharing a file, so a generated line ends up scored against a word-level floor.
# ---------------------------------------------------------------------------


def test_two_packs_do_not_share_a_reference_file(tmp_path):
    from nib.engine.metrics import references as ref

    assert ref.path_for(tmp_path, "cvl_lines_64.lmdb") != ref.path_for(
        tmp_path, "cvl_words_64.lmdb"
    )


def test_the_reference_file_is_named_after_its_pack(tmp_path):
    from nib.engine.metrics import references as ref

    assert ref.path_for(tmp_path, "cvl_lines_64.lmdb").name == "references_cvl_lines_64.json"


def test_references_round_trip(tmp_path):
    from nib.engine.metrics import references as ref

    values = {"fid_floor": 41.2, "cer_real": 0.09, "retrieval_real": 0.55, "pack": "p.lmdb"}
    ref.save(tmp_path / "outputs", "cvl_lines_64.lmdb", values)

    assert ref.load(tmp_path / "outputs", "cvl_lines_64.lmdb") == values


def test_unmeasured_references_are_none_rather_than_a_default(tmp_path):
    """A caller handed some other pack's numbers would report a comparison it
    never made. None forces the caller to say so."""
    from nib.engine.metrics import references as ref

    assert ref.load(tmp_path, "never_measured.lmdb") is None


def test_missing_names_the_baselines_a_run_did_not_produce(tmp_path):
    from nib.engine.metrics import references as ref

    assert ref.missing({"fid_floor": 1.0}) == ["cer_real", "retrieval_real"]
    assert ref.missing({"fid_floor": 1.0, "cer_real": 0.1, "retrieval_real": 0.5}) == []


def test_a_run_that_measured_less_does_not_delete_what_was_measured_before(tmp_path):
    """The Colab case. CER is read from the raw CVL images rather than from the
    pack, so a machine with the pack and not the 5 GB of sources measures two of
    the three -- and must not take the third down with it."""
    from nib.engine.metrics import references as ref

    ref.save(tmp_path, "p.lmdb", {"fid_floor": 33.0, "cer_real": 0.12, "retrieval_real": 0.6})
    _, carried = ref.update(tmp_path, "p.lmdb", {"fid_floor": 19.0, "retrieval_real": 0.83})

    after = ref.load(tmp_path, "p.lmdb")
    assert after["fid_floor"] == 19.0, "a re-measured number must win"
    assert after["cer_real"] == 0.12, "an unmeasured number must survive"
    assert carried == ["cer_real"]


def test_update_says_nothing_was_carried_when_the_file_is_new(tmp_path):
    from nib.engine.metrics import references as ref

    _, carried = ref.update(tmp_path, "fresh.lmdb", {"fid_floor": 19.0})
    assert carried == []
