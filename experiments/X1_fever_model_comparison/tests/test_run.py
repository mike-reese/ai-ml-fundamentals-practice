"""Tests for src/run.py on the tiny synthetic container from conftest. No data files.

validation_mode is run end to end on the tiny container with a two-point grid and two seeds, on
the CPU, into a temporary directory. The point is the files it writes and their shapes, not the
numbers.

Interface under test, from Interfaces.md (Run.py).
"""
import json

import numpy as np
import pandas as pd
import pytest

import data
import run
from conftest import TINY_SETTINGS


# -------------------------------------------------------------- pure helpers


def test_run_name_is_stable_and_distinct():
    assert run.run_name("logreg", 0.0, None, 1) == "logreg_wd0_s1"
    assert run.run_name("mlp", 1e-4, 256, 0) == "mlp_wd1e-04_h256_s0"
    names = {run.run_name(s, wd, h, seed) for s, h in [("logreg", None), ("mlp", 256), ("mlp", 512)]
             for wd in [0.0, 1e-5, 1e-4, 1e-3] for seed in [0, 1, 2]}
    assert len(names) == 3 * 4 * 3


def test_grid_runs_enumerates_every_configuration_in_declared_order():
    grid = {"weight_decay": [0.0, 1e-4], "hidden_sizes": [16, 32]}
    runs = run.grid_runs(grid, seeds=[0, 1])
    assert len(runs) == (1 + 2) * 2 * 2  # logreg + two MLP widths, two decays, two seeds
    assert runs[0] == {"system": "logreg", "weight_decay": 0.0, "hidden_size": None, "seed": 0}
    assert all(r["hidden_size"] is None for r in runs if r["system"] == "logreg")
    assert all(r["hidden_size"] in (16, 32) for r in runs if r["system"] == "mlp")
    assert len({run.run_name(**r) for r in runs}) == len(runs)


def test_summarize_reads_the_best_epoch_row():
    history = pd.DataFrame({"epoch": [1, 2, 3], "train_loss": [1.0, 0.8, 0.7],
                            "val_loss": [0.95, 0.85, 0.9], "val_macro_f1": [0.5, 0.6, 0.58], "seconds": [10.0, 10.0, 10.0]})
    result = type("R", (), {})()
    result.history, result.best_epoch, result.best_val_loss, result.best_val_macro_f1 = history, 2, 0.85, 0.6
    result.config = {"run_name": "mlp_wd0_h8_s0", "system": "mlp", "weight_decay": 0.0, "hidden_size": 8, "seed": 0}
    row = run.summarize(result)
    assert row["run_name"] == "mlp_wd0_h8_s0" and row["system"] == "mlp" and row["seed"] == 0
    assert row["best_epoch"] == 2 and row["epochs_run"] == 3
    assert row["val_log_loss"] == 0.85 and row["val_macro_f1"] == 0.6
    assert row["train_loss_at_best"] == 0.8
    assert row["generalization_gap"] == pytest.approx(0.8 - 0.85)
    assert row["seconds"] == pytest.approx(30.0)


def test_save_predictions_roundtrip(tmp_path, tiny_container):
    ds = tiny_container.validation
    N = ds.matrix.shape[0]
    probs = np.full((N, 3), 1 / 3)
    probs[0] = [0.8, 0.1, 0.1]
    path = tmp_path / "p.parquet"
    run.save_predictions(path, probs, ds)
    f = pd.read_parquet(path)
    assert list(f.columns) == ["row", "page", "label", "p_supports", "p_refutes", "p_nei"]
    assert f["row"].tolist() == list(range(N))
    np.testing.assert_array_equal(f["label"].to_numpy(), ds.labels)
    np.testing.assert_array_equal(f["page"].to_numpy().astype(str), ds.pages.astype(str))
    np.testing.assert_allclose(f[["p_supports", "p_refutes", "p_nei"]].to_numpy(), probs)


# ------------------------------------------------------------- validation_mode


@pytest.fixture(scope="module")
def pilot(tmp_path_factory, tiny_container):
    out = tmp_path_factory.mktemp("pilot")
    report = run.validation_mode(
        container=tiny_container,
        results_dir=out / "results", figures_dir=out / "figures",
        grid={"weight_decay": [0.0, 1e-3], "hidden_sizes": [8]}, seeds=[0, 1],
        training=TINY_SETTINGS, n_resamples=100,
        sizing_page_counts=[5, 10, 15], target_pages=615, skip_existing=False,
    )
    return out, report


def test_validation_summary_has_one_row_per_run(pilot):
    out, _ = pilot
    s = pd.read_csv(out / "results" / "validation_summary.csv")
    assert len(s) == 2 * 2 * 2  # two systems, two decays, two seeds
    assert {"run_name", "system", "weight_decay", "hidden_size", "seed", "best_epoch", "epochs_run",
            "val_log_loss", "val_macro_f1", "train_loss_at_best", "generalization_gap", "seconds"} <= set(s.columns)
    assert s["run_name"].is_unique
    assert ((s["val_macro_f1"] >= 0) & (s["val_macro_f1"] <= 1)).all()


def test_predictions_and_epoch_records_exist_for_every_run(pilot, tiny_container):
    out, _ = pilot
    s = pd.read_csv(out / "results" / "validation_summary.csv")
    N = tiny_container.validation.matrix.shape[0]
    for name in s["run_name"]:
        p = pd.read_parquet(out / "results" / "predictions" / f"{name}.parquet")
        assert len(p) == N
        np.testing.assert_allclose(p[["p_supports", "p_refutes", "p_nei"]].sum(axis=1), 1.0, atol=1e-6)
        record = (out / "results" / f"{name}.jsonl").read_text().splitlines()
        assert len(record) == int(s.loc[s.run_name == name, "epochs_run"].iloc[0])


def test_report_holds_every_analysis(pilot):
    out, report = pilot
    on_disk = json.loads((out / "results" / "validation_report.json").read_text())
    assert on_disk.keys() == report.keys()
    for key in ["constants", "chosen", "bootstrap_by_page", "bootstrap_by_claim", "seed_differences",
                "intra_page_correlation", "sizing", "calibration", "per_class", "n_train", "n_validation"]:
        assert key in report, key
    assert set(report["chosen"]) == {"logreg", "mlp"}
    assert report["chosen"]["mlp"]["hidden_size"] == 8
    # one paired bootstrap per seed, MLP minus logreg, with a draws count of n_resamples
    assert set(report["bootstrap_by_page"]) == {"0", "1"}
    for seed, b in report["bootstrap_by_page"].items():
        assert b["ci_low"] <= b["difference"] <= b["ci_high"]
        assert b["n_resamples"] == 100 and "draws" not in b
    assert len(report["seed_differences"]["per_seed"]) == 2
    assert {"mean_difference", "sd_difference"} <= set(report["seed_differences"])
    assert set(report["intra_page_correlation"]) == {"logreg", "mlp"}
    assert list(p["pages"] for p in report["sizing"]["table"]) == [5, 10, 15]
    assert report["sizing"]["target_pages"] == 615 and report["sizing"]["expected_half_width_at_target"] > 0
    for system in ["logreg", "mlp"]:
        assert len(report["calibration"][system]) == 10
        assert "confusion" in report["per_class"][system]
    assert report["constants"]["grid"]["weight_decay"] == [0.0, 1e-3]


def test_figures_are_written(pilot):
    out, _ = pilot
    assert (out / "figures" / "loss_over_epochs.png").stat().st_size > 1000
    assert (out / "figures" / "sizing.png").stat().st_size > 1000


def test_skip_existing_does_not_retrain(pilot, tiny_container, monkeypatch):
    out, _ = pilot
    import train
    calls = []
    monkeypatch.setattr(train, "train", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError("retrained")))
    run.validation_mode(
        container=tiny_container, results_dir=out / "results", figures_dir=out / "figures",
        grid={"weight_decay": [0.0, 1e-3], "hidden_sizes": [8]}, seeds=[0, 1], training=TINY_SETTINGS,
        n_resamples=50, sizing_page_counts=[5, 10], target_pages=615, skip_existing=True,
    )
    assert calls == []


def test_limit_train_subsets_the_training_split(tmp_path, tiny_container):
    report = run.validation_mode(
        container=tiny_container, results_dir=tmp_path / "r", figures_dir=tmp_path / "f",
        grid={"weight_decay": [0.0], "hidden_sizes": [8]}, seeds=[0], training=TINY_SETTINGS,
        n_resamples=50, sizing_page_counts=[5, 10], target_pages=615, skip_existing=False, limit_train=100,
    )
    assert report["n_train"] == 100
    assert report["n_validation"] == tiny_container.validation.matrix.shape[0]


# ------------------------------------------------------------------ test mode


def make_claim(tmp_path, **overrides):
    claim = {
        "frozen": "2026-10-15",
        "question": "q", "hypothesis": "h",
        "decision_rule": "supported when the 95% page-unit interval of the mean difference excludes 0 in the MLP's favor",
        "features": {"ngram_range": [1, 2], "min_df": 2},
        "training": {"optimizer": "AdamW", "learning_rate": 1e-2, "batch_size": 32, "max_epochs": 3, "patience": 3,
                     "early_stopping": "validation log loss"},
        "systems": {"logreg": {"weight_decay": 0.0, "hidden_size": None}, "mlp": {"weight_decay": 0.0, "hidden_size": 8}},
        "seeds": [0, 1],
        "bootstrap": {"unit": "page", "n_resamples": 50, "level": 0.95},
        "checks": [],
    }
    claim.update(overrides)
    path = tmp_path / "claim.json"
    path.write_text(json.dumps(claim))
    return path


def test_verdict():
    assert run.verdict(0.01, 0.05) == "supported"
    assert run.verdict(-0.01, 0.05) == "not established"
    assert run.verdict(-0.05, -0.01) == "not established"
    assert run.verdict(0.0, 0.05) == "not established"


def test_load_claim_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "FEATURE_SETTINGS", {"ngram_range": (1, 2), "min_df": 2})
    claim = run.load_claim(make_claim(tmp_path))
    assert claim["seeds"] == [0, 1]
    with pytest.raises(ValueError):
        run.load_claim(make_claim(tmp_path, features={"ngram_range": [1, 1], "min_df": 2}))
    with pytest.raises(ValueError):
        run.load_claim(make_claim(tmp_path, seeds=[]))
    bad = make_claim(tmp_path); d = json.loads(bad.read_text()); del d["decision_rule"]; bad.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        run.load_claim(bad)


@pytest.fixture(scope="module")
def test_run(tmp_path_factory, tiny_container, tiny_claim_only_container):
    out = tmp_path_factory.mktemp("testmode")
    claim = json.loads(make_claim(out).read_text())
    claim["training"]["device"] = "cpu"
    report = run.test_mode(tiny_container, tiny_claim_only_container, claim, results_dir=out / "results", figures_dir=out / "figures")
    return out, report


def test_test_mode_reads_the_test_split_once_per_model(tiny_container, tiny_claim_only_container, tmp_path, monkeypatch):
    import train
    calls = []
    original = train.predict_proba

    def counting(model, dataset, *a, **k):
        if dataset is tiny_container.test or dataset is tiny_claim_only_container.test:
            calls.append(1)
        return original(model, dataset, *a, **k)

    monkeypatch.setattr(train, "predict_proba", counting)
    claim = json.loads(make_claim(tmp_path).read_text()); claim["training"]["device"] = "cpu"
    run.test_mode(tiny_container, tiny_claim_only_container, claim, results_dir=tmp_path / "r", figures_dir=tmp_path / "f")
    assert len(calls) == 2 * 2 + 1  # two systems x two seeds, plus the claim-only baseline


def test_test_mode_writes_summary_predictions_and_report(test_run, tiny_container):
    out, report = test_run
    s = pd.read_csv(out / "results" / "test_summary.csv")
    assert len(s) == 5
    assert {"run_name", "system", "seed", "best_epoch", "epochs_run", "val_log_loss", "test_macro_f1", "test_log_loss", "test_accuracy"} <= set(s.columns)
    N = tiny_container.test.matrix.shape[0]
    for name in s["run_name"]:
        p = pd.read_parquet(out / "results" / "predictions" / f"{name}.parquet")
        assert len(p) == N
        np.testing.assert_array_equal(p["label"].to_numpy(), tiny_container.test.labels)
    on_disk = json.loads((out / "results" / "test_report.json").read_text())
    assert on_disk.keys() == report.keys()


def test_test_mode_report_holds_the_primary_result_and_the_verdict(test_run):
    _, r = test_run
    for key in ["claim", "primary", "per_seed", "seed_spread", "secondary_by_claim", "table", "confusion", "calibration",
                "generalization_gap", "verdict", "n_test", "n_test_pages"]:
        assert key in r, key
    p = r["primary"]
    assert p["ci_low"] <= p["difference"] <= p["ci_high"] and p["n_resamples"] == 50 and p["n_seeds"] == 2 and "draws" not in p
    assert r["verdict"] == run.verdict(p["ci_low"], p["ci_high"])
    assert len(r["per_seed"]) == 2 and {"mean_difference", "sd_difference"} <= set(r["seed_spread"])
    assert set(r["table"]) == {"mlp", "logreg", "majority", "claim_only"}
    for row in r["table"].values():
        assert {"macro_f1", "accuracy", "log_loss", "per_class"} <= set(row)
    assert r["table"]["majority"]["macro_f1"] < 0.5
    assert set(r["confusion"]) >= {"mlp", "logreg"} and len(r["confusion"]["mlp"]) == 3
    assert r["claim"]["seeds"] == [0, 1]


def test_test_mode_figure_is_written(test_run):
    out, _ = test_run
    assert (out / "figures" / "test_loss_over_epochs.png").stat().st_size > 1000


def test_test_mode_refuses_a_second_read(test_run, tiny_container, tiny_claim_only_container):
    out, _ = test_run
    claim = json.loads(make_claim(out / "again").read_text()) if (out / "again").mkdir() is None else None
    claim["training"]["device"] = "cpu"
    with pytest.raises(RuntimeError, match="once"):
        run.test_mode(tiny_container, tiny_claim_only_container, claim, results_dir=out / "results", figures_dir=out / "figures")


def test_main_test_mode_requires_a_committed_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CLAIM_PATH", tmp_path / "claim.json")
    with pytest.raises(RuntimeError, match="claim.json"):
        run.main(["test", "--results", str(tmp_path / "r"), "--figures", str(tmp_path / "f")])
    make_claim(tmp_path)
    monkeypatch.setattr(run, "claim_is_committed", lambda path: False)
    with pytest.raises(RuntimeError, match="commit"):
        run.main(["test", "--results", str(tmp_path / "r"), "--figures", str(tmp_path / "f")])
