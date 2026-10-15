"""Unit tests for src/train.py on tiny synthetic Datasets.

Nothing here reads data/fever or runs a real training job. Every Dataset is a few hundred rows
of random sparse features with a planted class signal, built in memory, on the CPU. A call to
train() on it takes well under a second. The point is the seams between functions:

  subset        -> slices matrix, labels and pages together, seeded
  batches       -> the CSR-to-dense-tensor boundary: every row once, right dtype, right values
  predict_proba -> softmax of the model's logits, rows in Dataset order whatever the batch size
  evaluate      -> the scikit-learn definitions of log loss and macro-F1
  train         -> early stopping, best-weight restore, the record file, the seed

Interface under test, from Interfaces.md (Train.py):
  train(system, training, validation, run_name, seed, learning_rate, weight_decay, hidden_size,
        batch_size, max_epochs, patience, device, results_dir) -> TrainResult
  batches(dataset, batch_size, rng, device) -> yields (x float32 sparse CSR (b, F), y int64 (b,))
  predict_proba(model, dataset, batch_size, device) -> (N, 3) float64
  evaluate(probabilities, labels) -> {"log_loss": float, "macro_f1": float}
  subset(dataset, n, seed) -> Dataset

The scikit-learn comparison (torch logistic regression within 0.02 macro-F1) is the completion
criterion for train.py. It needs real data and a real run, so it lives in checks.py, not here.
"""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

import data
import models
import train

F = 40            # features
N_TRAIN = 300
N_VAL = 120
DEVICE = "cpu"


# ------------------------------------------------------------------ fixtures


def make_dataset(n: int, seed: int) -> data.Dataset:
    """Random sparse rows with a planted signal: class c puts extra mass on features [10c, 10c+10)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, size=n).astype(np.int64)
    dense = rng.random((n, F)) * (rng.random((n, F)) < 0.15)  # ~15% nonzero noise
    for i, c in enumerate(labels):
        dense[i, 10 * c:10 * c + 10] += rng.random(10) * (rng.random(10) < 0.6)
    pages = np.array([f"page_{i // 4}" for i in range(n)], dtype=str)  # four rows per page
    return data.Dataset(matrix=sp.csr_matrix(dense), labels=labels, pages=pages)


@pytest.fixture(scope="module")
def training() -> data.Dataset:
    return make_dataset(N_TRAIN, seed=1)


@pytest.fixture(scope="module")
def validation() -> data.Dataset:
    return make_dataset(N_VAL, seed=2)


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="module")
def run(training, validation, results_dir):
    """Call train with this file's defaults. The one place the signature is assumed."""

    def _run(system="logreg", run_name="t", seed=0, learning_rate=1e-2, weight_decay=0.0,
             hidden_size=None, batch_size=32, max_epochs=6, patience=3):
        return train.train(
            system, training, validation, run_name=run_name, seed=seed,
            learning_rate=learning_rate, weight_decay=weight_decay, hidden_size=hidden_size,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience,
            device=DEVICE, results_dir=results_dir,
        )

    return _run


@pytest.fixture(scope="module")
def logreg_result(run):
    return run("logreg", run_name="logreg_shared")


@pytest.fixture(scope="module")
def mlp_result(run):
    return run("mlp", run_name="mlp_shared", hidden_size=8)


def read_record(results_dir: Path, run_name: str) -> list[dict]:
    lines = (results_dir / f"{run_name}.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# -------------------------------------------------------------------- subset


def test_subset_slices_matrix_labels_and_pages_together(validation):
    small = train.subset(validation, 50, seed=1)
    assert isinstance(small, data.Dataset)
    assert small.matrix.shape == (50, F)
    assert sp.issparse(small.matrix) and small.matrix.format == "csr"
    assert small.labels.shape == (50,) and small.labels.dtype == np.int64
    assert small.pages.shape == (50,)
    # each subset row must be a whole row of the original: find it by content and check its label and page travelled with it
    full = validation.matrix.toarray()
    for row, label, page in zip(small.matrix.toarray(), small.labels, small.pages):
        matches = np.flatnonzero((full == row).all(axis=1))
        assert matches.size == 1, "subset row is not a row of the original"
        assert validation.labels[matches[0]] == label
        assert validation.pages[matches[0]] == page


def test_subset_is_seeded_and_without_replacement(validation):
    a = train.subset(validation, 60, seed=7)
    b = train.subset(validation, 60, seed=7)
    c = train.subset(validation, 60, seed=8)
    np.testing.assert_array_equal(a.matrix.toarray(), b.matrix.toarray())
    assert not np.array_equal(a.matrix.toarray(), c.matrix.toarray())
    assert len({tuple(r) for r in a.matrix.toarray()}) == 60, "a row was drawn twice"


# ------------------------------------------------------------------- batches


def test_batches_cover_every_row_once_in_order_without_rng(training):
    seen = []
    for x, y in train.batches(training, batch_size=64, rng=None, device=DEVICE):
        assert isinstance(x, torch.Tensor) and x.dtype == torch.float32
        assert x.layout == torch.sparse_csr, "batches must send sparse rows, never densified"
        assert isinstance(y, torch.Tensor) and y.dtype == torch.int64
        assert x.shape[1] == F and x.shape[0] == y.shape[0]
        seen.append((x.to_dense().numpy(), y.numpy()))
    sizes = [x.shape[0] for x, _ in seen]
    assert sizes == [64, 64, 64, 64, 44], "every row once; the last batch holds the remainder"
    xs = np.vstack([x for x, _ in seen])
    ys = np.concatenate([y for _, y in seen])
    np.testing.assert_allclose(xs, training.matrix.toarray(), atol=1e-6)  # densified values match the CSR rows
    np.testing.assert_array_equal(ys, training.labels)


def test_batches_shuffle_with_the_generator_and_keep_rows_with_labels(training):
    rows = []
    for x, y in train.batches(training, batch_size=50, rng=np.random.default_rng(3), device=DEVICE):
        rows.append((x.to_dense().numpy(), y.numpy()))
    xs = np.vstack([x for x, _ in rows]); ys = np.concatenate([y for _, y in rows])
    full = training.matrix.toarray()
    assert not np.allclose(xs, full), "rows were not shuffled"
    # the same rows, as a set, each still paired with its own label
    order = [int(np.flatnonzero(np.isclose(full, r, atol=1e-6).all(axis=1))[0]) for r in xs]  # x is float32
    assert sorted(order) == list(range(N_TRAIN))
    np.testing.assert_array_equal(ys, training.labels[order])
    # the same seed gives the same order; a different seed does not
    again = np.vstack([x.to_dense().numpy() for x, _ in train.batches(training, 50, rng=np.random.default_rng(3), device=DEVICE)])
    other = np.vstack([x.to_dense().numpy() for x, _ in train.batches(training, 50, rng=np.random.default_rng(4), device=DEVICE)])
    np.testing.assert_array_equal(again, xs)
    assert not np.array_equal(other, xs)


# ------------------------------------------------------------------ evaluate


def test_evaluate_matches_scikit_learn_definitions():
    from sklearn.metrics import f1_score, log_loss

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=300).astype(np.int64)
    logits = rng.normal(size=(300, 3))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    m = train.evaluate(probs, labels)
    assert set(m) >= {"log_loss", "macro_f1"}
    assert m["log_loss"] == pytest.approx(log_loss(labels, probs, labels=[0, 1, 2]))
    assert m["macro_f1"] == pytest.approx(f1_score(labels, probs.argmax(axis=1), average="macro"))


def test_evaluate_perfect_and_uniform_predictions():
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    perfect = np.eye(3)[labels] * 0.97 + 0.01  # rows 0.98/0.01/0.01, sum 1
    assert train.evaluate(perfect, labels)["macro_f1"] == pytest.approx(1.0)
    uniform = np.full((6, 3), 1 / 3)
    assert train.evaluate(uniform, labels)["log_loss"] == pytest.approx(np.log(3))


def test_evaluate_macro_f1_when_a_class_is_never_predicted():
    """Macro-F1 averages over all three classes even if one is absent from the predictions."""
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    probs = np.array([[0.9, 0.1, 0], [0.9, 0.1, 0], [0.1, 0.9, 0], [0.1, 0.9, 0], [0.9, 0.1, 0], [0.1, 0.9, 0]])
    m = train.evaluate(probs, labels)
    # class 0 and class 1: precision 2/3, recall 1, F1 0.8; class 2: never predicted, F1 0
    assert m["macro_f1"] == pytest.approx((0.8 + 0.8 + 0) / 3)


# ------------------------------------------------------------- predict_proba


def test_predict_proba_is_the_softmax_of_the_logits_in_row_order(validation):
    torch.manual_seed(0)
    model = models.MultinomialLogisticRegression(F)  # untrained is fine: this tests the seam, not accuracy
    probs = train.predict_proba(model, validation, batch_size=32, device=DEVICE)
    assert isinstance(probs, np.ndarray) and probs.dtype == np.float64
    assert probs.shape == (N_VAL, 3)
    assert np.isfinite(probs).all() and (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(validation.matrix.toarray()).float()), dim=1).numpy()
    np.testing.assert_allclose(probs, expected, atol=1e-6)


def test_predict_proba_does_not_depend_on_batch_size(validation):
    torch.manual_seed(0)
    model = models.OneHiddenLayerMLP(F, hidden_size=8)
    a = train.predict_proba(model, validation, batch_size=7, device=DEVICE)
    b = train.predict_proba(model, validation, batch_size=1000, device=DEVICE)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_predict_proba_leaves_the_model_in_eval_mode_and_builds_no_graph(validation):
    model = models.OneHiddenLayerMLP(F, hidden_size=8)
    model.train()
    train.predict_proba(model, validation, device=DEVICE)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


# --------------------------------------------------------------------- train


def test_train_cannot_receive_the_test_split():
    """The one-test-read rule is enforced by the signature: there is no parameter for it."""
    params = inspect.signature(train.train).parameters
    assert "test" not in params and "test_dataset" not in params
    assert {"training", "validation"} <= set(params)


def test_train_result_fields_and_history_shape(logreg_result):
    r = logreg_result
    for field in ["model", "history", "best_epoch", "best_val_loss", "best_val_macro_f1", "config"]:
        assert hasattr(r, field), field
    assert isinstance(r.model, models.MultinomialLogisticRegression)
    h = r.history
    assert isinstance(h, pd.DataFrame)
    assert {"epoch", "train_loss", "val_loss", "val_macro_f1", "seconds"} <= set(h.columns)
    assert 1 <= len(h) <= 6
    assert h["epoch"].tolist() == list(range(1, len(h) + 1))
    assert np.isfinite(h[["train_loss", "val_loss", "val_macro_f1"]].to_numpy()).all()
    assert (h["seconds"] > 0).all()


def test_best_epoch_is_the_minimum_validation_loss(logreg_result):
    h = logreg_result.history
    i = int(h["val_loss"].idxmin())
    assert logreg_result.best_epoch == int(h.loc[i, "epoch"])
    assert logreg_result.best_val_loss == pytest.approx(h.loc[i, "val_loss"])
    assert logreg_result.best_val_macro_f1 == pytest.approx(h.loc[i, "val_macro_f1"])


def test_training_learns_the_planted_signal(logreg_result):
    """On synthetic data with a clear signal, validation loss falls below the uniform predictor."""
    h = logreg_result.history
    assert h["val_loss"].min() < h["val_loss"].iloc[0]
    assert logreg_result.best_val_loss < np.log(3)


def test_validation_metrics_in_history_come_from_evaluate_and_predict_proba(logreg_result, validation):
    """The last epoch's row must equal scoring the last-epoch weights with the public functions.

    train() must restore the best weights before returning, so compare against the best epoch.
    """
    probs = train.predict_proba(logreg_result.model, validation, device=DEVICE)
    m = train.evaluate(probs, validation.labels)
    assert m["log_loss"] == pytest.approx(logreg_result.best_val_loss, rel=1e-5)
    assert m["macro_f1"] == pytest.approx(logreg_result.best_val_macro_f1, abs=1e-9)


def test_returned_model_carries_the_best_epoch_weights_not_the_last(run, validation):
    """Force overfitting past the best epoch with a large learning rate and no early stop, then check
    that the returned model scores like the best epoch, not like the final one."""
    r = run("logreg", run_name="restore", learning_rate=0.5, max_epochs=10, patience=100)
    h = r.history
    if r.best_epoch == len(h):
        pytest.skip("the last epoch was the best; nothing to distinguish on this draw")
    m = train.evaluate(train.predict_proba(r.model, validation, device=DEVICE), validation.labels)
    assert m["log_loss"] == pytest.approx(r.best_val_loss, rel=1e-5)
    assert m["log_loss"] != pytest.approx(h["val_loss"].iloc[-1], rel=1e-5)


def test_config_records_every_setting(logreg_result):
    c = logreg_result.config
    for key in ["system", "run_name", "seed", "learning_rate", "weight_decay", "hidden_size",
                "batch_size", "max_epochs", "patience", "n_features", "n_parameters"]:
        assert key in c, key
    assert c["n_features"] == F
    assert c["n_parameters"] == 3 * F + 3
    assert c["system"] == "logreg" and c["hidden_size"] is None
    assert c["n_parameters"] == sum(p.numel() for p in logreg_result.model.parameters())


def test_early_stopping_stops_patience_epochs_after_the_best(run):
    r = run("logreg", run_name="early", learning_rate=0.5, max_epochs=40, patience=2)
    n = len(r.history)
    assert n <= 40
    assert n <= r.best_epoch + 2, "kept training more than `patience` epochs past the best"
    if n < 40:
        assert n == r.best_epoch + 2, "stopped early but not exactly `patience` epochs after the best"


def test_max_epochs_is_an_upper_bound(run):
    r = run("logreg", run_name="capped", max_epochs=2, patience=10)
    assert len(r.history) == 2


def test_hidden_size_must_match_the_system(run):
    with pytest.raises(ValueError):
        run("logreg", run_name="bad1", hidden_size=8)
    with pytest.raises(ValueError):
        run("mlp", run_name="bad2", hidden_size=None)
    with pytest.raises(ValueError):
        run("tree", run_name="bad3")


def test_mlp_trains_and_reports_its_parameter_count(mlp_result):
    H = 8
    assert isinstance(mlp_result.model, models.OneHiddenLayerMLP)
    assert mlp_result.config["system"] == "mlp" and mlp_result.config["hidden_size"] == H
    assert mlp_result.config["n_parameters"] == F * H + H + H * 3 + 3
    assert mlp_result.config["n_parameters"] == sum(p.numel() for p in mlp_result.model.parameters())
    assert mlp_result.best_val_loss < np.log(3)


def test_weight_decay_shrinks_the_weights(run):
    a = run("logreg", run_name="wd0", weight_decay=0.0, max_epochs=4, patience=10)
    b = run("logreg", run_name="wd1", weight_decay=1.0, max_epochs=4, patience=10)
    norm = lambda m: float(sum(p.detach().pow(2).sum() for p in m.parameters()).sqrt())  # noqa: E731
    assert norm(b.model) < norm(a.model)


# -------------------------------------------------------------------- record


def test_record_has_one_line_per_epoch_with_the_config_on_every_row(logreg_result, results_dir):
    rows = read_record(results_dir, "logreg_shared")
    h = logreg_result.history
    assert len(rows) == len(h)
    required = {"run_name", "system", "seed", "learning_rate", "weight_decay", "hidden_size", "batch_size", "execution",
                "epoch", "train_loss", "val_loss", "val_macro_f1", "seconds", "best_so_far"}
    for row in rows:
        assert required <= set(row), required - set(row)
        assert row["run_name"] == "logreg_shared" and row["system"] == "logreg"
        assert isinstance(row["best_so_far"], bool)
    assert [r["epoch"] for r in rows] == h["epoch"].tolist()
    np.testing.assert_allclose([r["val_loss"] for r in rows], h["val_loss"].to_numpy())
    np.testing.assert_allclose([r["train_loss"] for r in rows], h["train_loss"].to_numpy())
    np.testing.assert_allclose([r["val_macro_f1"] for r in rows], h["val_macro_f1"].to_numpy())


def test_record_best_so_far_flags_new_minima(logreg_result, results_dir):
    rows = read_record(results_dir, "logreg_shared")
    best = np.inf
    for row in rows:
        assert row["best_so_far"] == (row["val_loss"] < best)
        best = min(best, row["val_loss"])


def test_record_is_appended_per_epoch_not_written_at_the_end(training, validation, results_dir):
    """The dashboard reads the file while training runs, so each epoch's line must land as it finishes."""
    seen = []
    original = train.evaluate

    def spy(probs, labels):
        path = results_dir / "live.jsonl"
        seen.append(len(path.read_text().splitlines()) if path.exists() else 0)
        return original(probs, labels)

    train.evaluate = spy
    try:
        train.train("logreg", training, validation, run_name="live", seed=0, learning_rate=1e-2, weight_decay=0.0,
                    hidden_size=None, batch_size=32, max_epochs=3, patience=10, device=DEVICE, results_dir=results_dir)
    finally:
        train.evaluate = original
    # when epoch k is being scored, exactly k-1 lines are already on disk
    assert seen == [0, 1, 2]


def test_execution_is_written_on_every_row(training, validation, results_dir):
    train.train("logreg", training, validation, run_name="exec", seed=0, learning_rate=1e-2, weight_decay=0.0,
                hidden_size=None, batch_size=32, max_epochs=2, patience=5, device=DEVICE, results_dir=results_dir,
                execution="pilot_test")
    assert all(r["execution"] == "pilot_test" for r in read_record(results_dir, "exec"))


def test_record_is_overwritten_by_a_rerun_of_the_same_name(run, results_dir):
    run("logreg", run_name="again", max_epochs=3, patience=5)
    run("logreg", run_name="again", max_epochs=2, patience=5)
    assert len(read_record(results_dir, "again")) == 2


# --------------------------------------------------------------- reproducibility


def test_same_seed_reproduces_the_history_and_a_different_seed_does_not(run):
    a = run("logreg", run_name="seed_a", seed=11, max_epochs=3, patience=5)
    b = run("logreg", run_name="seed_b", seed=11, max_epochs=3, patience=5)
    c = run("logreg", run_name="seed_c", seed=12, max_epochs=3, patience=5)
    for col in ["train_loss", "val_loss", "val_macro_f1"]:
        np.testing.assert_allclose(a.history[col].to_numpy(), b.history[col].to_numpy(), rtol=1e-6, atol=1e-8)
    assert not np.allclose(a.history["train_loss"].to_numpy(), c.history["train_loss"].to_numpy(), rtol=1e-6)
    # the seed covers the initial weights too, not only the batch order
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        np.testing.assert_allclose(pa.detach().numpy(), pb.detach().numpy(), atol=1e-7)
