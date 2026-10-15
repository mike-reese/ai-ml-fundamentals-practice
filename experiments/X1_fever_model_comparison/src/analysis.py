"""Statistics for experiment X1, as pure functions of arrays. See Interfaces.md (Analysis.py).

The paired bootstrap resamples pages: one draw takes n_pages pages with replacement, which
becomes a weight per row (how many times its page was drawn), and each system's macro-F1 for
that draw is computed from a weighted confusion matrix. Written in numpy here, not imported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
N_CLASSES = 3


# ----------------------------------------------------------------- metrics


def confusion(labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """(3, 3) counts, rows true, columns predicted, optionally weighted per row."""
    cell = labels.astype(np.int64) * N_CLASSES + predictions.astype(np.int64)
    return np.bincount(cell, weights=weights, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES).astype(float)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)  # absent class -> 0, as sklearn
    return float(f1.mean())


def macro_f1(labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None) -> float:
    return macro_f1_from_confusion(confusion(labels, predictions, weights))


# --------------------------------------------------------------- bootstrap


def page_index_of(pages: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(np.asarray(pages).astype(str)))
    return codes.astype(np.int64), len(uniques)


def page_weights(page_index: np.ndarray, n_pages: int, rng: np.random.Generator) -> np.ndarray:
    """Row weights of one page draw: n_pages pages with replacement, every row of a page together."""
    counts = np.bincount(rng.integers(0, n_pages, size=n_pages), minlength=n_pages)
    return counts[page_index].astype(float)


def _percentile_summary(difference: float, draws: np.ndarray) -> dict:
    return {
        "difference": float(difference),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "se": float(draws.std(ddof=1)) if len(draws) > 1 else 0.0,
        "n_resamples": int(len(draws)),
        "draws": draws,
    }


def paired_bootstrap_by_page(labels, pages, predictions_a, predictions_b, n_resamples: int = 2000, seed: int = 0) -> dict:
    """Percentile bootstrap of macro_f1(a) - macro_f1(b), resampling pages. Both systems are scored on
    the same drawn pages in every draw, so the interval is of the difference."""
    labels = np.asarray(labels, dtype=np.int64)
    a = np.asarray(predictions_a, dtype=np.int64)
    b = np.asarray(predictions_b, dtype=np.int64)
    page_index, n_pages = page_index_of(pages)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        w = page_weights(page_index, n_pages, rng)
        draws[i] = macro_f1(labels, a, w) - macro_f1(labels, b, w)
    out = _percentile_summary(macro_f1(labels, a) - macro_f1(labels, b), draws)
    out["n_pages"] = int(n_pages)
    return out


def paired_bootstrap_by_page_seeds(labels, pages, predictions_a: list, predictions_b: list,
                                   n_resamples: int = 2000, seed: int = 0) -> dict:
    """The primary interval: mean over seeds of macro_f1(a_k) - macro_f1(b_k), resampling pages. One draw
    resamples pages once and scores every seed's pair on those same pages before averaging, so the
    interval is of the mean difference and the seeds are paired with each other as well."""
    labels = np.asarray(labels, dtype=np.int64)
    A = [np.asarray(a, dtype=np.int64) for a in predictions_a]
    B = [np.asarray(b, dtype=np.int64) for b in predictions_b]
    assert len(A) == len(B) and len(A) >= 1
    page_index, n_pages = page_index_of(pages)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        w = page_weights(page_index, n_pages, rng)
        draws[i] = np.mean([macro_f1(labels, a, w) - macro_f1(labels, b, w) for a, b in zip(A, B)])
    per_seed = [macro_f1(labels, a) - macro_f1(labels, b) for a, b in zip(A, B)]
    out = _percentile_summary(float(np.mean(per_seed)), draws)
    out["n_pages"] = int(n_pages)
    out["n_seeds"] = len(A)
    out["per_seed"] = [float(d) for d in per_seed]
    return out


def paired_bootstrap_by_claim(labels, predictions_a, predictions_b, n_resamples: int = 2000, seed: int = 0) -> dict:
    """The same with rows as the unit: one draw takes N rows with replacement. Secondary line only."""
    labels = np.asarray(labels, dtype=np.int64)
    a = np.asarray(predictions_a, dtype=np.int64)
    b = np.asarray(predictions_b, dtype=np.int64)
    n = len(labels)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        draws[i] = macro_f1(labels, a, w) - macro_f1(labels, b, w)
    out = _percentile_summary(macro_f1(labels, a) - macro_f1(labels, b), draws)
    out["n_rows"] = int(n)
    return out


# ------------------------------------------------------------------ sizing


def sizing(labels, pages, predictions_a, predictions_b, page_counts: list[int], target_pages: int,
           n_resamples: int = 1000, n_repeats: int = 3, seed: int = 0) -> dict:
    """Bootstrap standard error of the difference as a function of pages held out, from the
    validation split, and its extrapolation se = c / sqrt(pages) to the test split's page count."""
    labels = np.asarray(labels, dtype=np.int64)
    a = np.asarray(predictions_a, dtype=np.int64)
    b = np.asarray(predictions_b, dtype=np.int64)
    page_index, n_pages = page_index_of(pages)
    rng = np.random.default_rng(seed)
    rows = []
    for m in page_counts:
        m = int(min(m, n_pages))
        ses, halves = [], []
        for _ in range(n_repeats):
            chosen = rng.choice(n_pages, size=m, replace=False)
            keep = np.isin(page_index, chosen)
            r = paired_bootstrap_by_page(labels[keep], np.asarray(pages)[keep], a[keep], b[keep],
                                         n_resamples=n_resamples, seed=int(rng.integers(0, 2**31)))
            ses.append(r["se"])
            halves.append((r["ci_high"] - r["ci_low"]) / 2)
        rows.append({"pages": m, "se": float(np.mean(ses)), "half_width": float(np.mean(halves))})
    table = pd.DataFrame(rows)
    x = 1 / np.sqrt(table["pages"].to_numpy(float))
    c = float((table["se"].to_numpy() * x).sum() / (x * x).sum())  # least squares through the origin
    expected_se = c / np.sqrt(target_pages)
    return {
        "table": table,
        "fit": c,
        "target_pages": int(target_pages),
        "expected_se_at_target": float(expected_se),
        "expected_half_width_at_target": float(1.96 * expected_se),
    }


# ---------------------------------------------------- intra-page correlation


def intra_page_correlation(correct, pages) -> dict:
    """One-way ANOVA intraclass correlation of correctness by page, and the design effect."""
    y = np.asarray(correct, dtype=float)
    page_index, k = page_index_of(pages)
    n = len(y)
    sizes = np.bincount(page_index, minlength=k).astype(float)
    sums = np.bincount(page_index, weights=y, minlength=k)
    means = sums / sizes
    grand = y.mean()
    ms_between = float((sizes * (means - grand) ** 2).sum() / (k - 1)) if k > 1 else 0.0
    ms_within = float(((y - means[page_index]) ** 2).sum() / (n - k)) if n > k else 0.0
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1) if k > 1 else 1.0
    denom = ms_between + (n0 - 1) * ms_within
    icc = float((ms_between - ms_within) / denom) if denom > 0 else 0.0
    mean_size = float(sizes.mean())
    return {"icc": icc, "mean_page_size": mean_size, "design_effect": float(1 + (mean_size - 1) * icc), "n_pages": int(k)}


# ---------------------------------------------------------------- selection


def select_configuration(summary: pd.DataFrame, criterion: str = "val_macro_f1") -> dict:
    """Per system, the (weight_decay, hidden_size) with the best mean over seeds. Ties go to the
    other criterion. Never chosen per seed."""
    other = "val_log_loss" if criterion == "val_macro_f1" else "val_macro_f1"
    higher_is_better = criterion == "val_macro_f1"
    s = summary.copy()
    s["hidden_size"] = s["hidden_size"].astype("float")
    g = (s.groupby(["system", "weight_decay", "hidden_size"], dropna=False)
          .agg(mean=(criterion, "mean"), std=(criterion, "std"), n_seeds=(criterion, "size"), other=(other, "mean"))
          .reset_index())
    chosen = {}
    for system, grp in g.groupby("system"):
        grp = grp.sort_values(["mean", "other"], ascending=[not higher_is_better, higher_is_better])
        best = grp.iloc[0]
        h = best["hidden_size"]
        chosen[system] = {
            "weight_decay": float(best["weight_decay"]),
            "hidden_size": None if pd.isna(h) else int(h),
            "mean": float(best["mean"]),
            "std": float(best["std"]) if not pd.isna(best["std"]) else 0.0,
            "n_seeds": int(best["n_seeds"]),
            "criterion": criterion,
        }
    return chosen


def _rows_of(summary: pd.DataFrame, system: str, choice: dict) -> pd.DataFrame:
    s = summary[summary["system"] == system]
    s = s[np.isclose(s["weight_decay"].astype(float), choice["weight_decay"])]
    if choice["hidden_size"] is None:
        s = s[s["hidden_size"].isna()]
    else:
        s = s[s["hidden_size"].astype(float) == float(choice["hidden_size"])]
    return s


def seed_differences(summary: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    """Per seed, MLP macro-F1 minus logistic regression macro-F1 at the chosen configurations."""
    mlp = _rows_of(summary, "mlp", chosen["mlp"]).set_index("seed")["val_macro_f1"]
    logreg = _rows_of(summary, "logreg", chosen["logreg"]).set_index("seed")["val_macro_f1"]
    seeds = sorted(set(mlp.index) & set(logreg.index))
    out = pd.DataFrame({"seed": seeds, "mlp_macro_f1": mlp.loc[seeds].to_numpy(), "logreg_macro_f1": logreg.loc[seeds].to_numpy()})
    out["difference"] = out["mlp_macro_f1"] - out["logreg_macro_f1"]
    out.attrs["mean_difference"] = float(out["difference"].mean())
    out.attrs["sd_difference"] = float(out["difference"].std(ddof=1)) if len(out) > 1 else 0.0
    return out


# ------------------------------------------------------------- calibration


def calibration_bins(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table for the argmax probability. Empty bins are kept with count 0."""
    p = np.asarray(probabilities, dtype=float)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == np.asarray(labels)).astype(float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    correct_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(count > 0, conf_sum / count, np.nan)
        acc = np.where(count > 0, correct_sum / count, np.nan)
    edges = np.linspace(0, 1, n_bins + 1)
    return pd.DataFrame({"bin": np.arange(n_bins), "lower": edges[:-1], "upper": edges[1:], "count": count,
                         "mean_confidence": mean_conf, "accuracy": acc})


def per_class_report(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    report = classification_report(labels, predictions, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    return {"report": report, "confusion": cm.tolist(), "accuracy": float((labels == predictions).mean())}
