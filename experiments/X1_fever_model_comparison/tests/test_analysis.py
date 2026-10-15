"""Unit tests for src/analysis.py on synthetic arrays. No data files, no training.

Interface under test, from Interfaces.md (Analysis.py):
  macro_f1(labels, predictions, weights=None) -> float
  page_weights(page_index, n_pages, rng) -> float (N,) row weights of one page draw
  paired_bootstrap_by_page(labels, pages, predictions_a, predictions_b, n_resamples, seed) -> dict
  paired_bootstrap_by_claim(labels, predictions_a, predictions_b, n_resamples, seed) -> dict
  sizing(labels, pages, predictions_a, predictions_b, page_counts, target_pages, n_resamples, n_repeats, seed) -> dict
  intra_page_correlation(correct, pages) -> dict
  select_configuration(summary, criterion) -> dict
  seed_differences(summary, chosen) -> frame
  calibration_bins(probabilities, labels, n_bins) -> frame
  per_class_report(labels, predictions) -> dict
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import analysis

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def make_pages(n_pages: int, rows_per_page: int) -> np.ndarray:
    return np.repeat([f"page_{i}" for i in range(n_pages)], rows_per_page).astype(str)


# ------------------------------------------------------------------ macro_f1


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 500).astype(np.int64)
    preds = rng.integers(0, 3, 500).astype(np.int64)
    assert analysis.macro_f1(labels, preds) == pytest.approx(f1_score(labels, preds, average="macro"))
    assert analysis.macro_f1(labels, labels) == pytest.approx(1.0)


def test_macro_f1_with_integer_weights_equals_repeated_rows():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, 200).astype(np.int64)
    preds = rng.integers(0, 3, 200).astype(np.int64)
    w = rng.integers(0, 4, 200)
    repeated = analysis.macro_f1(np.repeat(labels, w), np.repeat(preds, w))
    assert analysis.macro_f1(labels, preds, weights=w.astype(float)) == pytest.approx(repeated)


def test_macro_f1_absent_class_counts_as_zero_like_sklearn():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    preds = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    assert analysis.macro_f1(labels, preds) == pytest.approx(f1_score(labels, preds, average="macro", labels=[0, 1, 2]))
    assert analysis.macro_f1(labels, preds) == pytest.approx((0.8 + 0.8 + 0) / 3)


# ---------------------------------------------------------- page_weights


def test_page_weights_are_constant_within_a_page_and_count_n_pages_draws():
    page_index = np.repeat(np.arange(5), 4)  # 5 pages, 4 rows each
    w = analysis.page_weights(page_index, 5, np.random.default_rng(0))
    assert w.shape == (20,)
    for p in range(5):
        assert len(set(w[page_index == p])) == 1, "a page was split across the draw"
    per_page = np.array([w[page_index == p][0] for p in range(5)])
    assert per_page.sum() == 5, "one draw takes n_pages pages with replacement"
    assert (per_page >= 0).all() and np.issubdtype(per_page.dtype, np.floating)


# -------------------------------------------------- paired_bootstrap_by_page


def test_paired_bootstrap_identical_systems_give_zero_interval():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 400).astype(np.int64)
    preds = rng.integers(0, 3, 400).astype(np.int64)
    pages = make_pages(40, 10)
    r = analysis.paired_bootstrap_by_page(labels, pages, preds, preds, n_resamples=200, seed=0)
    assert r["difference"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0 and r["se"] == 0
    assert r["n_pages"] == 40 and r["n_resamples"] == 200 and r["draws"].shape == (200,)


def test_paired_bootstrap_detects_a_clear_advantage_and_is_seeded():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 3, 600).astype(np.int64)
    pages = make_pages(60, 10)
    a = labels.copy()                                   # always right
    b = labels.copy()
    wrong = rng.random(600) < 0.3                       # b wrong on 30% of rows
    b[wrong] = (labels[wrong] + 1) % 3
    r1 = analysis.paired_bootstrap_by_page(labels, pages, a, b, n_resamples=300, seed=5)
    r2 = analysis.paired_bootstrap_by_page(labels, pages, a, b, n_resamples=300, seed=5)
    assert r1["difference"] > 0 and r1["ci_low"] > 0
    assert r1["ci_low"] <= r1["difference"] <= r1["ci_high"]
    assert r1["se"] > 0
    np.testing.assert_array_equal(r1["draws"], r2["draws"])
    r3 = analysis.paired_bootstrap_by_page(labels, pages, a, b, n_resamples=300, seed=6)
    assert not np.array_equal(r1["draws"], r3["draws"])


def test_paired_bootstrap_by_page_is_wider_than_by_claim_when_pages_are_homogeneous():
    """If every row of a page is right or wrong together, the page unit must give a wider interval."""
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 3, 800).astype(np.int64)
    pages = make_pages(40, 20)
    a = labels.copy()
    b = labels.copy()
    bad_pages = rng.random(40) < 0.4
    bad_rows = np.repeat(bad_pages, 20)
    b[bad_rows] = (labels[bad_rows] + 1) % 3
    by_page = analysis.paired_bootstrap_by_page(labels, pages, a, b, n_resamples=400, seed=0)
    by_claim = analysis.paired_bootstrap_by_claim(labels, a, b, n_resamples=400, seed=0)
    assert by_page["se"] > by_claim["se"] * 1.5
    assert by_claim["draws"].shape == (400,) and by_claim["difference"] == pytest.approx(by_page["difference"])


def test_paired_bootstrap_by_page_seeds_averages_the_seeds_on_shared_pages():
    rng = np.random.default_rng(9)
    labels = rng.integers(0, 3, 600).astype(np.int64)
    pages = make_pages(60, 10)
    a0, a1 = labels.copy(), labels.copy()
    b0, b1 = labels.copy(), labels.copy()
    for arr, rate in [(b0, 0.3), (b1, 0.1), (a1, 0.05)]:
        wrong = rng.random(600) < rate
        arr[wrong] = (labels[wrong] + 1) % 3
    r = analysis.paired_bootstrap_by_page_seeds(labels, pages, [a0, a1], [b0, b1], n_resamples=300, seed=1)
    single = [analysis.paired_bootstrap_by_page(labels, pages, a, b, n_resamples=300, seed=1)["difference"] for a, b in [(a0, b0), (a1, b1)]]
    assert r["n_seeds"] == 2 and r["per_seed"] == pytest.approx(single)
    assert r["difference"] == pytest.approx(np.mean(single))
    assert r["ci_low"] <= r["difference"] <= r["ci_high"] and r["draws"].shape == (300,)
    one = analysis.paired_bootstrap_by_page_seeds(labels, pages, [a0], [b0], n_resamples=300, seed=1)
    ref = analysis.paired_bootstrap_by_page(labels, pages, a0, b0, n_resamples=300, seed=1)
    np.testing.assert_allclose(one["draws"], ref["draws"])  # one seed reduces to the single-pair bootstrap


# ------------------------------------------------------------------- sizing


def test_sizing_table_and_extrapolation():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 3, 2000).astype(np.int64)
    pages = make_pages(200, 10)
    a = labels.copy()
    b = labels.copy()
    wrong = rng.random(2000) < 0.25
    b[wrong] = (labels[wrong] + 1) % 3
    r = analysis.sizing(labels, pages, a, b, page_counts=[25, 50, 100, 200], target_pages=615,
                        n_resamples=200, n_repeats=2, seed=0)
    t = r["table"]
    assert list(t["pages"]) == [25, 50, 100, 200]
    assert (t["se"] > 0).all() and (t["half_width"] > t["se"]).all()
    assert t["se"].iloc[0] > t["se"].iloc[-1], "standard error must fall as pages grow"
    assert r["fit"] > 0
    assert 0 < r["expected_se_at_target"] < t["se"].iloc[-1]
    assert r["expected_half_width_at_target"] == pytest.approx(1.96 * r["expected_se_at_target"])


# ---------------------------------------------------- intra_page_correlation


def test_intra_page_correlation_is_one_when_pages_are_homogeneous_and_near_zero_when_random():
    pages = make_pages(50, 10)
    homogeneous = np.repeat(np.arange(50) % 2, 10).astype(bool)
    r = analysis.intra_page_correlation(homogeneous, pages)
    assert r["icc"] == pytest.approx(1.0, abs=0.02)
    assert r["n_pages"] == 50 and r["mean_page_size"] == pytest.approx(10.0)
    assert r["design_effect"] == pytest.approx(1 + 9 * r["icc"])
    random = np.random.default_rng(0).random(500) < 0.5
    r2 = analysis.intra_page_correlation(random, pages)
    assert abs(r2["icc"]) < 0.1


# ------------------------------------------------------ select_configuration


def make_summary():
    rows = []
    for system in ["logreg", "mlp"]:
        for wd in [0.0, 1e-4, 1e-3]:
            for seed in [0, 1, 2]:
                f1 = {0.0: 0.70, 1e-4: 0.74, 1e-3: 0.72}[wd] + (0.03 if system == "mlp" else 0) + 0.001 * seed
                rows.append({"run_name": f"{system}_{wd}_{seed}", "system": system, "weight_decay": wd,
                             "hidden_size": 256 if system == "mlp" else None, "seed": seed,
                             "val_macro_f1": f1, "val_log_loss": 1 - f1})
    return pd.DataFrame(rows)


def test_select_configuration_picks_the_best_mean_over_seeds_per_system():
    chosen = analysis.select_configuration(make_summary(), criterion="val_macro_f1")
    assert set(chosen) == {"logreg", "mlp"}
    for system in chosen:
        assert chosen[system]["weight_decay"] == 1e-4
        assert chosen[system]["n_seeds"] == 3
    assert chosen["mlp"]["hidden_size"] == 256 and chosen["logreg"]["hidden_size"] is None
    assert chosen["mlp"]["mean"] == pytest.approx(0.74 + 0.03 + 0.001)
    assert chosen["mlp"]["std"] == pytest.approx(np.std([0.771, 0.772, 0.773], ddof=1))


def test_select_configuration_by_log_loss_prefers_lower():
    s = make_summary()
    s.loc[(s.system == "logreg") & (s.weight_decay == 1e-3), "val_log_loss"] = 0.01  # best loss, not best F1
    chosen = analysis.select_configuration(s, criterion="val_log_loss")
    assert chosen["logreg"]["weight_decay"] == 1e-3
    assert chosen["mlp"]["weight_decay"] == 1e-4


def test_seed_differences_pairs_the_same_seed():
    s = make_summary()
    chosen = analysis.select_configuration(s)
    d = analysis.seed_differences(s, chosen)
    assert list(d["seed"]) == [0, 1, 2]
    np.testing.assert_allclose(d["difference"], d["mlp_macro_f1"] - d["logreg_macro_f1"])
    np.testing.assert_allclose(d["difference"], 0.03)
    assert d.attrs["mean_difference"] == pytest.approx(0.03)
    assert d.attrs["sd_difference"] == pytest.approx(0.0, abs=1e-12)


# -------------------------------------------------------- calibration_bins


def test_calibration_bins_cover_every_row():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(1000, 3))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, 3, 1000).astype(np.int64)
    b = analysis.calibration_bins(probs, labels, n_bins=10)
    assert len(b) == 10 and b["count"].sum() == 1000
    assert {"bin", "lower", "upper", "count", "mean_confidence", "accuracy"} <= set(b.columns)
    nonempty = b[b["count"] > 0]
    assert ((nonempty["accuracy"] >= 0) & (nonempty["accuracy"] <= 1)).all()
    assert ((nonempty["mean_confidence"] >= nonempty["lower"]) & (nonempty["mean_confidence"] <= nonempty["upper"])).all()
    # the argmax probability is at least 1/3, so the first three bins of ten are empty
    assert (b["count"].iloc[:3] == 0).all()


# -------------------------------------------------------- per_class_report


def test_per_class_report_confusion_and_accuracy():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    preds = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    r = analysis.per_class_report(labels, preds)
    assert r["confusion"] == [[2, 0, 0], [0, 2, 0], [1, 1, 0]]
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert set(LABELS) <= set(r["report"])
    assert r["report"]["SUPPORTS"]["recall"] == pytest.approx(1.0)
    assert r["report"]["NOT ENOUGH INFO"]["support"] == 2
