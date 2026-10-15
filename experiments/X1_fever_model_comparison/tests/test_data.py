"""Tests for src/data.py against the real assembled dataset.

Checks come from README.md (Checks, Data Card) and practice/specs/X1.md
(Data assembly and Features-and-training tables). Every expected count is read
from data/fever/data_card.json, which scripts/get_data.py wrote.

Interface under test, from Interfaces.md:
  return_dataset_splits(dataset) -> (training, validation, test)   three pandas frames
  build_dataset_features(...)    -> DatasetContainer(training, validation, test, vectorizer)
      each split a Dataset(matrix: scipy CSR (N, 2V + 2) or (N, V), labels: int64 (N,), pages: str (N,))

Two call-site assumptions, both isolated in the fixtures `splits` and `build`:
  1. `dataset` is the claims.parquet frame loaded with pandas, as run.py will load it.
  2. build_dataset_features takes the three split frames plus the vectorizer settings
     (ngram_range, min_df) and a claim_only flag, which Interfaces.md item 2 says the
     caller supplies. If the final signature differs, change `build` only.

The dataset is gitignored; without data/fever/claims.parquet every test here is skipped.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import data

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data" / "fever"
CLAIMS_PATH = DATA_DIR / "claims.parquet"
CARD_PATH = DATA_DIR / "data_card.json"

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]  # integer codes 0, 1, 2 in this order
SPLIT_NAMES = ["training", "validation", "test"]

# Vectorizer settings for the feature build. These are design values to be recorded
# on the data card once chosen; the tests only need them to be the same for every build.
FEATURE_SETTINGS = {"ngram_range": (1, 2), "min_df": 2}

pytestmark = pytest.mark.skipif(
    not CLAIMS_PATH.exists() or not CARD_PATH.exists(),
    reason="data/fever/claims.parquet and data_card.json are produced by scripts/get_data.py and are not committed",
)


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def card() -> dict:
    return json.loads(CARD_PATH.read_text())


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """The assembled dataset, loaded the way run.py will load it."""
    return pd.read_parquet(CLAIMS_PATH)


@pytest.fixture(scope="module")
def splits(dataset) -> dict[str, pd.DataFrame]:
    training, validation, test = data.return_dataset_splits(dataset)
    return {"training": training, "validation": validation, "test": test}


@pytest.fixture(scope="module")
def build(splits):
    """Call build_dataset_features. The one place the feature-build signature is assumed."""

    def _build(claim_only: bool = False):
        return data.build_dataset_features(
            splits["training"], splits["validation"], splits["test"],
            claim_only=claim_only, **FEATURE_SETTINGS,
        )

    return _build


@pytest.fixture(scope="module")
def container(build):
    """The full input, (N, 2V + 2). Built once for the module; it is the expensive step."""
    return build(claim_only=False)


@pytest.fixture(scope="module")
def claim_only_container(build):
    """The claim block alone, (N, V), for the claim-only baseline."""
    return build(claim_only=True)


def vocabulary_size(container) -> int:
    return len(container.vectorizer.vocabulary_)


# ------------------------------------------------- return_dataset_splits


def test_three_frames_each_holding_one_split(splits):
    for name in SPLIT_NAMES:
        frame = splits[name]
        assert isinstance(frame, pd.DataFrame)
        assert set(frame["split"].unique()) == {name}, f"{name} frame holds rows from another split"


def test_claims_per_split_match_the_data_card(splits, card):
    expected = card["split"]["claims_per_split"]
    for name in SPLIT_NAMES:
        assert len(splits[name]) == expected[name]
    assert sum(len(f) for f in splits.values()) == card["claims_kept"]


def test_pages_per_split_match_the_data_card(splits, card):
    expected = card["split"]["pages_per_split"]
    for name in SPLIT_NAMES:
        assert splits[name]["primary_page"].nunique() == expected[name]


def test_labels_per_split_match_the_data_card(splits, card):
    expected = card["split"]["labels_per_split"]
    for name in SPLIT_NAMES:
        observed = splits[name]["label"].value_counts().to_dict()
        assert observed == expected[name]


def test_no_page_in_two_splits(splits):
    """Check: grouping preserved. The page is the independent unit; a leaked page is named."""
    pages = {name: set(splits[name]["primary_page"]) for name in SPLIT_NAMES}
    for a, b in [("training", "validation"), ("training", "test"), ("validation", "test")]:
        shared = pages[a] & pages[b]
        assert shared == set(), f"pages in both {a} and {b}: {sorted(shared)[:10]}"


def test_no_claim_in_two_splits(splits):
    ids = {name: set(splits[name]["id"]) for name in SPLIT_NAMES}
    for a, b in [("training", "validation"), ("training", "test"), ("validation", "test")]:
        assert ids[a] & ids[b] == set()
    for name in SPLIT_NAMES:
        assert splits[name]["id"].is_unique


def test_every_row_has_a_page_a_label_and_evidence(splits):
    for name in SPLIT_NAMES:
        frame = splits[name]
        assert frame["primary_page"].notna().all()
        assert frame["label"].isin(LABELS).all()
        assert frame["claim"].map(lambda s: isinstance(s, str) and s.strip() != "").all()
        assert frame["evidence_text"].map(lambda s: isinstance(s, str)).all()


# ------------------------------------------------ build_dataset_features


def test_container_has_three_datasets_and_a_fitted_vectorizer(container):
    for name in SPLIT_NAMES:
        assert hasattr(container, name)
    assert hasattr(container.vectorizer, "vocabulary_"), "vectorizer is not fitted"
    assert vocabulary_size(container) > 0


def test_matrix_shape_is_n_by_2v_plus_2_and_csr(container, splits):
    V = vocabulary_size(container)
    for name in SPLIT_NAMES:
        matrix = getattr(container, name).matrix
        assert sp.isspmatrix_csr(matrix) or (sp.issparse(matrix) and matrix.format == "csr")
        assert matrix.shape == (len(splits[name]), 2 * V + 2)


def test_labels_are_int64_codes_in_label_order(container, splits):
    """SUPPORTS -> 0, REFUTES -> 1, NOT ENOUGH INFO -> 2, row for row."""
    code = {label: i for i, label in enumerate(LABELS)}
    for name in SPLIT_NAMES:
        labels = getattr(container, name).labels
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.int64
        assert labels.shape == (len(splits[name]),)
        expected = splits[name]["label"].map(code).to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(labels, expected)


def test_pages_follow_the_rows(container, splits):
    for name in SPLIT_NAMES:
        pages = getattr(container, name).pages
        assert isinstance(pages, np.ndarray)
        assert pages.shape == (len(splits[name]),)
        np.testing.assert_array_equal(pages.astype(str), splits[name]["primary_page"].to_numpy().astype(str))


def test_vocabulary_fitted_on_the_training_set_only(container):
    """Check: no term that occurs only in the validation or test set is in the vocabulary.

    Every vocabulary column must be nonzero in at least one training text, claim or evidence.
    A term present in the vocabulary but absent from every training row could only have come
    from another split.
    """
    V = vocabulary_size(container)
    training = container.training.matrix
    claim_block = training[:, :V]
    evidence_block = training[:, V:2 * V]
    nonzero_per_term = np.asarray((claim_block != 0).sum(axis=0)).ravel() + np.asarray((evidence_block != 0).sum(axis=0)).ravel()
    leaked = np.flatnonzero(nonzero_per_term == 0)
    terms = np.array(container.vectorizer.get_feature_names_out())
    assert leaked.size == 0, f"{leaked.size} vocabulary terms never occur in the training set, e.g. {terms[leaked[:10]].tolist()}"


def test_vocabulary_terms_meet_min_df_on_training_texts(container):
    """Every term's document frequency over training claim and evidence texts is at least min_df."""
    V = vocabulary_size(container)
    training = container.training.matrix
    df = np.asarray((training[:, :V] != 0).sum(axis=0)).ravel() + np.asarray((training[:, V:2 * V] != 0).sum(axis=0)).ravel()
    assert df.min() >= FEATURE_SETTINGS["min_df"]


def test_tfidf_rows_have_unit_length_or_are_empty(container):
    """Each block is a TF-IDF vector normalized to unit length; an empty text gives a zero row."""
    V = vocabulary_size(container)
    rng = np.random.default_rng(0)
    for name in SPLIT_NAMES:
        matrix = getattr(container, name).matrix
        rows = rng.choice(matrix.shape[0], size=min(2000, matrix.shape[0]), replace=False)
        for lo, hi in [(0, V), (V, 2 * V)]:
            norms = sp.linalg.norm(matrix[rows, lo:hi], axis=1)
            assert np.all(np.isclose(norms, 1.0, atol=1e-6) | np.isclose(norms, 0.0, atol=1e-12))


def test_pair_features_are_standardized_with_training_statistics(container):
    """Columns 2V and 2V+1 (overlap count, cosine) have mean 0 and standard deviation 1 on training."""
    V = vocabulary_size(container)
    pair = container.training.matrix[:, 2 * V:].toarray()
    assert pair.shape[1] == 2
    np.testing.assert_allclose(pair.mean(axis=0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pair.std(axis=0), [1.0, 1.0], atol=1e-4)


def test_pair_features_on_other_splits_are_not_refit(container):
    """Validation and test pair columns use the training mean and std, so they are not exactly standardized."""
    V = vocabulary_size(container)
    for name in ["validation", "test"]:
        pair = getattr(container, name).matrix[:, 2 * V:].toarray()
        assert np.isfinite(pair).all()
        # Both columns can't be exactly standardized on a different sample; one differing is enough.
        assert not (np.allclose(pair.mean(axis=0), 0.0, atol=1e-9) and np.allclose(pair.std(axis=0), 1.0, atol=1e-9))


def test_pair_features_recover_overlap_and_cosine(container):
    """Undo the standardization on a few training rows and check the raw values against the blocks.

    overlap = number of vocabulary terms nonzero in both blocks; cosine = dot product of the two
    unit-length blocks. Standardization is (x - mean) / std with training statistics, so the
    ranking and the pairwise differences of the raw values are preserved up to one scale.
    """
    V = vocabulary_size(container)
    matrix = container.training.matrix
    rows = np.arange(0, matrix.shape[0], max(1, matrix.shape[0] // 200))[:200]
    claim = matrix[rows, :V]
    evidence = matrix[rows, V:2 * V]
    raw_overlap = np.asarray(claim.multiply(evidence).astype(bool).sum(axis=1)).ravel().astype(float)
    raw_cosine = np.asarray(claim.multiply(evidence).sum(axis=1)).ravel()
    pair = matrix[rows, 2 * V:].toarray()
    for raw, col in [(raw_overlap, pair[:, 0]), (raw_cosine, pair[:, 1])]:
        if raw.std() == 0:
            continue
        correlation = np.corrcoef(raw, col)[0, 1]
        assert correlation > 0.999, f"standardized pair feature does not match the raw value (corr {correlation:.4f})"


def test_claim_only_build_is_the_claim_block(container, claim_only_container, splits):
    """The claim-only baseline receives the claim block alone, (N, V), from the same vocabulary."""
    V = vocabulary_size(container)
    assert vocabulary_size(claim_only_container) == V
    for name in SPLIT_NAMES:
        full = getattr(container, name)
        claim_only = getattr(claim_only_container, name)
        assert claim_only.matrix.shape == (len(splits[name]), V)
        difference = (claim_only.matrix - full.matrix[:, :V])
        assert abs(difference).max() < 1e-9 if difference.nnz else True
        np.testing.assert_array_equal(claim_only.labels, full.labels)
        np.testing.assert_array_equal(claim_only.pages, full.pages)
