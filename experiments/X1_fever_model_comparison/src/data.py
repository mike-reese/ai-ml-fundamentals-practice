"""Features for experiment X1: the three splits as frames, and the TF-IDF input matrices.

Everything that is fitted (the vectorizer, the pair-feature mean and std) is fitted on the
training split only and then applied to all three splits. See Interfaces.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")  # integer codes 0, 1, 2 in this order
LABEL_CODE = {label: i for i, label in enumerate(LABELS)}
SPLIT_NAMES = ("training", "validation", "test")


@dataclass
class Dataset:
    """One split, ready for a model."""

    matrix: sp.csr_matrix  # (N, 2V + 2) for the full input, (N, V) for claim-only
    labels: np.ndarray     # int64 (N,), codes from LABEL_CODE
    pages: np.ndarray      # str (N,), primary page per row; the bootstrap resamples these


@dataclass
class DatasetContainer:
    """The three splits plus the fitted vectorizer, so V can be reported."""

    training: Dataset
    validation: Dataset
    test: Dataset
    vectorizer: TfidfVectorizer


def return_dataset_splits(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition the assembled dataset by its `split` column.

    Returns (training, validation, test). Row order within each split is preserved.
    """
    training = dataset[dataset["split"] == "training"].reset_index(drop=True)
    validation = dataset[dataset["split"] == "validation"].reset_index(drop=True)
    test = dataset[dataset["split"] == "test"].reset_index(drop=True)
    return training, validation, test


def _transform_split(frame: pd.DataFrame, vectorizer: TfidfVectorizer):
    """Transform one split with the fitted vectorizer.

    Returns (claim, evidence, pair): the two (N, V) CSR blocks, unit-length rows, and the
    raw (N, 2) pair features [overlap count, cosine]. Nothing is fitted here.
    """
    claim = vectorizer.transform(frame["claim"])
    evidence = vectorizer.transform(frame["evidence_text"])
    prod = claim.multiply(evidence)  # nonzero only where both blocks are nonzero
    overlap = np.asarray((prod != 0).sum(axis=1)).ravel().astype(np.float64)
    cosine = np.asarray(prod.sum(axis=1)).ravel()  # rows are unit length, so the dot product is the cosine
    return claim, evidence, np.column_stack([overlap, cosine])


def _to_dataset(frame: pd.DataFrame, matrix: sp.csr_matrix) -> Dataset:
    labels = frame["label"].map(LABEL_CODE).to_numpy(dtype=np.int64)
    pages = frame["primary_page"].to_numpy(dtype=str)
    return Dataset(matrix=matrix, labels=labels, pages=pages)


def build_dataset_features(
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    ngram_range: tuple[int, int],
    min_df: int,
    claim_only: bool = False,
) -> DatasetContainer:
    """Fit the TF-IDF vectorizer and the pair-feature scaler on the training split only,
    then transform all three splits.

    Full input (claim_only=False), shape (N, 2V + 2):
        [0, V)        claim block, TF-IDF, unit length
        [V, 2V)       evidence block, same vocabulary, unit length
        [2V, 2V + 2)  overlap count and cosine between the blocks, standardized with
                      the training mean and std
    Claim-only input (claim_only=True), shape (N, V): the claim block alone, same vocabulary.
    """
    # Fit the vectorizer on the training corpus: claims and evidence texts as one vocabulary,
    # so column j means the same term in both blocks. transform() never refits.
    corpus = training_dataset["claim"].tolist() + training_dataset["evidence_text"].tolist()
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df).fit(corpus)

    frames = {"training": training_dataset, "validation": validation_dataset, "test": test_dataset}

    if claim_only:
        datasets = {name: _to_dataset(f, vectorizer.transform(f["claim"]).tocsr()) for name, f in frames.items()}
        return DatasetContainer(**datasets, vectorizer=vectorizer)

    # Transform each split once, keeping the raw pair features.
    parts = {name: _transform_split(f, vectorizer) for name, f in frames.items()}

    # Fit the scaler on the training pair features only; apply the same mean and std everywhere.
    pair_train = parts["training"][2]
    mean = pair_train.mean(axis=0)
    std = pair_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # a constant column would divide by zero; leave it centered instead

    datasets = {}
    for name, (claim, evidence, pair) in parts.items():
        standardized = sp.csr_matrix((pair - mean) / std)
        matrix = sp.hstack([claim, evidence, standardized], format="csr")
        datasets[name] = _to_dataset(frames[name], matrix)
    return DatasetContainer(**datasets, vectorizer=vectorizer)
