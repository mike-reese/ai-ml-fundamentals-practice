"""Make the experiment's own code importable as `import data`, the way run.py will.

pytest loads this file before the tests in this directory. It puts
experiments/X1_fever_model_comparison/src on sys.path, so `import data` resolves
to src/data.py whether pytest is run from the repository root or from here.
"""
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))


# ------------------------------------------------------------ shared fixtures
# A tiny synthetic claims frame with the columns run.py, checks.py and data.py read, so the
# pipeline can be exercised end to end in a second without data/fever. The text carries a
# planted label signal: each label has its own set of words that appear more often.

import numpy as np
import pandas as pd
import pytest

_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
_CLAIM_WORDS = {"A": ["born", "released", "founded", "located"], "B": ["never", "only", "solely", "unable"]}
_EVIDENCE_WORDS = {"X": ["city", "album", "team", "year"], "Y": ["river", "song", "club", "decade"],
                   "Z": ["beloved", "famous", "popular", "wonderful"]}
_FILLER = ["the", "a", "in", "of", "and", "film", "person", "actor", "band", "country"]


def make_claims_frame(n_pages: int, rows_per_page: int, seed: int, split: str) -> pd.DataFrame:
    """The label is an interaction between a claim word group and an evidence word group:
    (A, X) or (B, Y) -> SUPPORTS; (A, Y) or (B, X) -> REFUTES; evidence Z -> NOT ENOUGH INFO.
    A linear model cannot separate SUPPORTS from REFUTES (it is XOR); an MLP can. 15% label noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pages):
        page = f"{split}_page_{p}"
        for i in range(rows_per_page):
            cg = rng.choice(["A", "B"])
            eg = rng.choice(["X", "Y", "Z"])
            if eg == "Z":
                label = "NOT ENOUGH INFO"
            else:
                label = "SUPPORTS" if (cg, eg) in {("A", "X"), ("B", "Y")} else "REFUTES"
            if rng.random() < 0.15:
                label = _LABELS[rng.integers(0, 3)]
            claim = " ".join(rng.permutation(list(rng.choice(_CLAIM_WORDS[cg], size=2)) + list(rng.choice(_FILLER, size=4))))
            evidence = " ".join(rng.permutation(list(rng.choice(_EVIDENCE_WORDS[eg], size=2)) + list(rng.choice(_FILLER, size=5))))
            rows.append({"id": len(rows) + 1_000_000 * (hash(split) % 7), "label": label, "claim": claim,
                         "primary_page": page, "split": split, "evidence_text": evidence})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_frame() -> pd.DataFrame:
    """Training 40 pages x 6 rows, validation 15 x 6, test 15 x 6, as one frame like claims.parquet."""
    return pd.concat([make_claims_frame(40, 6, 1, "training"), make_claims_frame(15, 6, 2, "validation"),
                      make_claims_frame(15, 6, 3, "test")], ignore_index=True)


@pytest.fixture(scope="session")
def tiny_container(tiny_frame):
    import data
    tr, va, te = data.return_dataset_splits(tiny_frame)
    return data.build_dataset_features(tr, va, te, ngram_range=(1, 1), min_df=1, claim_only=False)


@pytest.fixture(scope="session")
def tiny_claim_only_container(tiny_frame):
    import data
    tr, va, te = data.return_dataset_splits(tiny_frame)
    return data.build_dataset_features(tr, va, te, ngram_range=(1, 1), min_df=1, claim_only=True)


TINY_SETTINGS = {"learning_rate": 1e-2, "weight_decay": 0.0, "batch_size": 32, "max_epochs": 6, "patience": 3, "device": "cpu"}
