"""Unit tests for src/checks.py on the tiny synthetic container from conftest. No data files.

Each check function is run on the tiny container with CPU settings and a few epochs; the point
is the row it returns (keys, observed values, the holds verdict logic), not the numbers on real
data, which checks.py's main() writes to results/checks.json.

Interface under test, from Interfaces.md (Checks.py).
"""
import copy
import json

import numpy as np
import pytest

import checks
from conftest import TINY_SETTINGS

ROW_KEYS = {"quantity", "condition", "tolerance", "observed", "holds", "run_names"}


class Untouchable:
    """Stands in for the test Dataset: any attribute access is a failure."""

    def __getattr__(self, name):
        raise AssertionError(f"a check read the test split (.{name})")


@pytest.fixture(scope="module")
def container(tiny_container):
    c = copy.copy(tiny_container)  # a copy: the session fixture is shared with test_run.py, which needs its test split
    c.test = Untouchable()
    return c


@pytest.fixture(scope="module")
def claim_only(tiny_claim_only_container):
    c = copy.copy(tiny_claim_only_container)
    c.test = Untouchable()
    return c


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    return {**TINY_SETTINGS, "results_dir": tmp_path_factory.mktemp("results")}


def assert_row(row):
    assert ROW_KEYS <= set(row), ROW_KEYS - set(row)
    assert row["holds"] in (True, False, "reported")
    json.dumps(row)  # every observed value must serialize


# ------------------------------------------------------------------ helpers


def test_shuffle_labels_is_a_seeded_permutation():
    labels = np.array([0] * 50 + [1] * 30 + [2] * 20, dtype=np.int64)
    a = checks.shuffle_labels(labels, seed=0)
    b = checks.shuffle_labels(labels, seed=0)
    c = checks.shuffle_labels(labels, seed=1)
    assert a.dtype == np.int64 and a.shape == labels.shape
    np.testing.assert_array_equal(np.bincount(a), np.bincount(labels))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, labels)


def test_within():
    assert checks.within(0.70, 0.71, 0.02)
    assert checks.within(0.70, 0.72, 0.02)
    assert not checks.within(0.70, 0.7201, 0.02)
    assert checks.within(1 / 3, 0.38, 0.05)


def test_majority_predictions():
    training = np.array([0, 0, 0, 1, 2, 2], dtype=np.int64)
    p = checks.majority_predictions(training, 5)
    assert p.dtype == np.int64 and p.tolist() == [0, 0, 0, 0, 0]


def test_rows_to_markdown_has_one_line_per_row_with_the_observed_value():
    rows = [
        {"quantity": "Q1", "condition": "C1", "tolerance": "exact", "observed": {"n": 3}, "holds": True, "run_names": []},
        {"quantity": "Q2", "condition": "C2", "tolerance": "0.02", "observed": {"gap": 0.0123}, "holds": False, "run_names": ["r"]},
        {"quantity": "Q3", "condition": "C3", "tolerance": "reported", "observed": {"f1": 0.5}, "holds": "reported", "run_names": []},
    ]
    md = checks.rows_to_markdown(rows)
    lines = [l for l in md.splitlines() if l.startswith("|")]
    assert len(lines) == 2 + 3  # header, separator, rows
    assert "Q1" in lines[2] and "pass" in lines[2]
    assert "Q2" in lines[3] and "fail" in lines[3] and "0.0123" in lines[3]
    assert "Q3" in lines[4] and "reported" in lines[4]


# ---------------------------------------------------------------- the rows


def test_check_vocabulary(container):
    row = checks.check_vocabulary(container)
    assert_row(row)
    assert row["observed"]["vocabulary_size"] == len(container.vectorizer.vocabulary_)
    assert row["observed"]["terms_absent_from_training"] == 0
    assert row["holds"] is True


def test_check_vocabulary_fails_on_a_leaked_term(container):
    """Plant a column that is zero on every training row; the check must see it."""
    leaked = copy.copy(container)
    V = len(container.vectorizer.vocabulary_)
    m = container.training.matrix.tolil()
    m[:, 0] = 0  # kill term 0 in the claim block
    m[:, V] = 0  # and in the evidence block
    leaked.training = type(container.training)(matrix=m.tocsr(), labels=container.training.labels, pages=container.training.pages)
    row = checks.check_vocabulary(leaked)
    assert row["holds"] is False and row["observed"]["terms_absent_from_training"] == 1


def test_check_against_sklearn(container, settings):
    row = checks.check_against_sklearn(container, n_train=0, settings=settings, seed=0)
    assert_row(row)
    o = row["observed"]
    assert {"torch_macro_f1", "sklearn_macro_f1", "gap", "n_train"} <= set(o)
    assert o["n_train"] == container.training.matrix.shape[0]
    assert o["gap"] == pytest.approx(abs(o["torch_macro_f1"] - o["sklearn_macro_f1"]))
    assert row["holds"] == (o["gap"] <= 0.02)
    assert len(row["run_names"]) == 1 and row["run_names"][0].startswith("check_")


def test_check_against_sklearn_subsets_training(container, settings):
    row = checks.check_against_sklearn(container, n_train=60, settings=settings, seed=0)
    assert row["observed"]["n_train"] == 60


def test_check_shuffled_labels(container, settings):
    row = checks.check_shuffled_labels(container, n_train=0, settings=settings, seed=0)
    assert_row(row)
    o = row["observed"]
    assert {"macro_f1", "chance_uniform", "chance_prior_only", "distance_from_uniform", "distance_from_prior_only",
            "label_prior_entropy"} <= set(o)
    assert o["chance_uniform"] == pytest.approx(1 / 3)
    assert o["distance_from_uniform"] == pytest.approx(abs(o["macro_f1"] - 1 / 3))
    assert o["distance_from_prior_only"] == pytest.approx(abs(o["macro_f1"] - o["chance_prior_only"]))
    assert 0 < o["label_prior_entropy"] <= np.log(3) + 1e-9
    assert row["holds"] == (min(o["distance_from_uniform"], o["distance_from_prior_only"]) <= 0.05)


def test_check_seeded_rerun_holds_on_cpu(container, settings):
    row = checks.check_seeded_rerun(container, settings=settings, seed=3)
    assert_row(row)
    o = row["observed"]
    assert o["max_abs_difference"] <= 1e-6
    assert row["holds"] is True
    assert len(row["run_names"]) == 2
    assert o["epochs"] >= 1


def test_baseline_majority(container):
    row = checks.baseline_majority(container)
    assert_row(row)
    o = row["observed"]
    counts = np.bincount(container.training.labels, minlength=3)
    assert o["majority_code"] == int(counts.argmax())
    assert 0 <= o["macro_f1"] <= 1 and 0 <= o["accuracy"] <= 1
    assert row["holds"] == "reported"


def test_baseline_claim_only(claim_only, settings):
    row = checks.baseline_claim_only(claim_only, settings=settings, seed=0)
    assert_row(row)
    o = row["observed"]
    assert {"macro_f1", "log_loss", "macro_f1_supports_refutes"} <= set(o)
    assert row["holds"] == "reported"
    assert o["n_features"] == claim_only.training.matrix.shape[1]


# ------------------------------------------------------------------- main


def test_main_writes_checks_json_and_returns_exit_status(tiny_container, tiny_claim_only_container, tmp_path, monkeypatch):
    """main() on the tiny container: every row present, JSON written, status 0 or 1 by the holds."""
    monkeypatch.setattr(checks, "load_containers", lambda: (tiny_container, tiny_claim_only_container))
    status = checks.main(["--results", str(tmp_path), "--device", "cpu", "--max-epochs", "4", "--n-train", "0"])
    out = json.loads((tmp_path / "checks.json").read_text())
    quantities = [r["quantity"] for r in out["rows"]]
    assert len(quantities) == 6
    assert any("sklearn" in q.lower() or "scikit" in q.lower() for q in quantities)
    assert any("shuffled" in q.lower() for q in quantities)
    assert any("rerun" in q.lower() for q in quantities)
    assert any("claim" in q.lower() and "only" in q.lower() for q in quantities)
    assert any("majority" in q.lower() for q in quantities)
    assert "settings" in out
    failed = [r for r in out["rows"] if r["holds"] is False]
    assert status == (1 if failed else 0)
    assert (tmp_path / "checks.md").exists()
