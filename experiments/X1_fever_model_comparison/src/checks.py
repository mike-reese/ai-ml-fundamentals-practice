"""The "Features and training" check table and the two baselines, on real data. See Interfaces.md.

    uv run python src/checks.py                     # every row, 30,000-row training subset for the two cheap rows
    uv run python src/checks.py --n-train 0         # the full training split everywhere
    uv run python src/checks.py --only sklearn rerun

Writes results/checks.json and results/checks.md; prints the markdown table. Exit status 1 if a
row with a tolerance fails. Nothing here reads the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import analysis
import data
import train

EXPERIMENT = Path(__file__).resolve().parent.parent
REPO = EXPERIMENT.parent.parent
CLAIMS_PATH = REPO / "data" / "fever" / "claims.parquet"
FEATURE_SETTINGS = {"ngram_range": (1, 2), "min_df": 2}  # must match run.py


# ------------------------------------------------------------------ helpers


def shuffle_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(np.asarray(labels, dtype=np.int64))


def within(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance + 1e-12


def majority_predictions(training_labels: np.ndarray, n: int) -> np.ndarray:
    code = int(np.bincount(np.asarray(training_labels, dtype=np.int64), minlength=3).argmax())
    return np.full(n, code, dtype=np.int64)


def rows_to_markdown(rows: list[dict]) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = ["| Check | Condition | Tolerance | Observed | Holds |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        observed = "; ".join(f"{k} {fmt(v)}" for k, v in r["observed"].items())
        holds = "reported" if r["holds"] == "reported" else ("pass" if r["holds"] else "fail")
        lines.append(f"| {r['quantity']} | {r['condition']} | {r['tolerance']} | {observed} | {holds} |")
    return "\n".join(lines)


def _row(quantity, condition, tolerance, observed, holds, run_names=()):
    return {"quantity": quantity, "condition": condition, "tolerance": tolerance,
            "observed": observed, "holds": holds, "run_names": list(run_names)}


def _train_logreg(training: data.Dataset, validation: data.Dataset, run_name: str, settings: dict, seed: int):
    return train.train("logreg", training, validation, run_name=run_name, seed=seed, hidden_size=None, **settings)


def _validation_f1(model, validation: data.Dataset, device) -> tuple[float, float, np.ndarray]:
    probs = train.predict_proba(model, validation, device=device)
    m = train.evaluate(probs, validation.labels)
    return m["macro_f1"], m["log_loss"], probs


# ----------------------------------------------------------------- the rows


def check_vocabulary(container) -> dict:
    """No vocabulary term is absent from every training text (claim or evidence block)."""
    V = len(container.vectorizer.vocabulary_)
    m = container.training.matrix
    df = np.asarray((m[:, :V] != 0).sum(axis=0)).ravel() + np.asarray((m[:, V:2 * V] != 0).sum(axis=0)).ravel()
    absent = int((df == 0).sum())
    return _row("Vectorizer vocabulary fitted on the training set only",
                "no term that occurs only in the validation or test set is in it", "exact",
                {"vocabulary_size": V, "terms_absent_from_training": absent}, absent == 0)


def check_against_sklearn(container, n_train: int, settings: dict, seed: int) -> dict:
    """Torch logistic regression vs sklearn's LogisticRegression on the same rows, validation macro-F1."""
    training = train.subset(container.training, n_train, seed) if n_train else container.training
    validation = container.validation
    r = _train_logreg(training, validation, "check_sklearn_torch", settings, seed)
    torch_f1, _, _ = _validation_f1(r.model, validation, settings.get("device"))
    ref = LogisticRegression(max_iter=1000).fit(training.matrix, training.labels)
    ref_f1 = train.evaluate(ref.predict_proba(validation.matrix), validation.labels)["macro_f1"]
    gap = abs(torch_f1 - ref_f1)
    return _row("Torch logistic regression against sklearn LogisticRegression, same features, validation macro-F1",
                "within 0.02", "0.02",
                {"torch_macro_f1": float(torch_f1), "sklearn_macro_f1": float(ref_f1), "gap": float(gap),
                 "n_train": int(training.matrix.shape[0]), "sklearn_C": 1.0, "torch_best_epoch": r.best_epoch},
                gap <= 0.02, [r.config["run_name"]])


def check_shuffled_labels(container, n_train: int, settings: dict, seed: int) -> dict:
    """Negative control: training labels permuted; validation macro-F1 must sit at chance."""
    training = train.subset(container.training, n_train, seed) if n_train else container.training
    shuffled = data.Dataset(matrix=training.matrix, labels=shuffle_labels(training.labels, seed), pages=training.pages)
    r = _train_logreg(shuffled, container.validation, "check_shuffled_labels", settings, seed)
    f1, loss, _ = _validation_f1(r.model, container.validation, settings.get("device"))
    labels = container.validation.labels
    # Two chance references: uniform guessing (1/3) and learning only the label prior (the majority-class
    # macro-F1; with 53% SUPPORTS a model fitted to noise predicts SUPPORTS everywhere). Either is "no signal".
    prior_f1 = analysis.macro_f1(labels, majority_predictions(training.labels, len(labels)))
    p = np.bincount(training.labels, minlength=3) / len(training.labels)
    prior_entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    distance_uniform = abs(f1 - 1 / 3)
    distance_prior = abs(f1 - prior_f1)
    return _row("Shuffled training labels, negative control",
                "validation macro-F1 at chance: within 0.05 of 1/3 (uniform) or of the majority-class macro-F1 (prior only)",
                "0.05",
                {"macro_f1": float(f1), "chance_uniform": 1 / 3, "chance_prior_only": float(prior_f1),
                 "distance_from_uniform": float(distance_uniform), "distance_from_prior_only": float(distance_prior),
                 "log_loss": float(loss), "label_prior_entropy": prior_entropy, "n_train": int(training.matrix.shape[0])},
                min(distance_uniform, distance_prior) <= 0.05, [r.config["run_name"]])


def check_seeded_rerun(container, settings: dict, seed: int) -> dict:
    """The same seed twice reproduces the per-epoch validation metrics."""
    a = _train_logreg(container.training, container.validation, "check_rerun_a", settings, seed)
    b = _train_logreg(container.training, container.validation, "check_rerun_b", settings, seed)
    same_length = len(a.history) == len(b.history)
    diff = 0.0
    if same_length:
        for col in ["val_loss", "val_macro_f1", "train_loss"]:
            diff = max(diff, float(np.abs(a.history[col].to_numpy() - b.history[col].to_numpy()).max()))
    else:
        diff = float("inf")
    return _row("Seeded rerun", "the same seed reproduces the saved validation metrics", "1e-6",
                {"max_abs_difference": diff, "epochs": int(len(a.history)), "same_epoch_count": bool(same_length),
                 "val_macro_f1": float(a.best_val_macro_f1)},
                same_length and diff <= 1e-6, [a.config["run_name"], b.config["run_name"]])


def baseline_majority(container) -> dict:
    labels = container.validation.labels
    preds = majority_predictions(container.training.labels, len(labels))
    return _row("Majority-class baseline", "training majority class predicted for every validation claim", "reported",
                {"majority_code": int(preds[0]), "majority_label": analysis.LABELS[int(preds[0])],
                 "macro_f1": analysis.macro_f1(labels, preds), "accuracy": float((preds == labels).mean())},
                "reported")


def baseline_claim_only(claim_only_container, settings: dict, seed: int) -> dict:
    """Torch logistic regression on the claim block alone: the annotation artifact."""
    c = claim_only_container
    r = _train_logreg(c.training, c.validation, "check_claim_only", settings, seed)
    f1, loss, probs = _validation_f1(r.model, c.validation, settings.get("device"))
    labels = c.validation.labels
    sr = labels < 2  # SUPPORTS and REFUTES rows only
    f1_sr = float(f1_score(labels[sr], probs[sr].argmax(axis=1), average="macro", labels=[0, 1], zero_division=0))
    return _row("Claim-only baseline", "torch logistic regression on the claim block alone, validation", "reported",
                {"macro_f1": float(f1), "log_loss": float(loss), "macro_f1_supports_refutes": f1_sr,
                 "n_features": int(c.training.matrix.shape[1]), "best_epoch": r.best_epoch},
                "reported", [r.config["run_name"]])


# -------------------------------------------------------------------- main


def load_containers():
    frame = pd.read_parquet(CLAIMS_PATH)
    tr, va, te = data.return_dataset_splits(frame)
    full = data.build_dataset_features(tr, va, te, claim_only=False, **FEATURE_SETTINGS)
    claim_only = data.build_dataset_features(tr, va, te, claim_only=True, **FEATURE_SETTINGS)
    return full, claim_only


ROWS = ["vocabulary", "sklearn", "shuffled", "rerun", "majority", "claim_only"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=EXPERIMENT / "results")
    ap.add_argument("--n-train", type=int, default=30_000, help="training rows for the sklearn and shuffled rows; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", nargs="*", choices=ROWS, default=None)
    args = ap.parse_args(argv)

    settings = {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                "max_epochs": args.max_epochs, "patience": args.patience, "device": args.device or train.default_device(),
                "results_dir": args.results, "execution": f"checks_{time.strftime('%Y%m%d_%H%M%S')}"}
    want = set(args.only or ROWS)
    full, claim_only = load_containers()

    rows = []
    if "vocabulary" in want:
        rows.append(check_vocabulary(full))
    if "sklearn" in want:
        rows.append(check_against_sklearn(full, args.n_train, settings, args.seed))
    if "shuffled" in want:
        rows.append(check_shuffled_labels(full, args.n_train, settings, args.seed))
    if "rerun" in want:
        rows.append(check_seeded_rerun(full, settings, args.seed))
    if "majority" in want:
        rows.append(baseline_majority(full))
    if "claim_only" in want:
        rows.append(baseline_claim_only(claim_only, settings, args.seed))

    args.results.mkdir(parents=True, exist_ok=True)
    out = {"rows": rows, "settings": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in settings.items()},
                                      "n_train": args.n_train, "seed": args.seed, "features": FEATURE_SETTINGS}}
    (args.results / "checks.json").write_text(json.dumps(out, indent=2))
    md = rows_to_markdown(rows)
    (args.results / "checks.md").write_text(md + "\n")
    print(md)
    failed = [r["quantity"] for r in rows if r["holds"] is False]
    if failed:
        print("FAILED: " + "; ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
