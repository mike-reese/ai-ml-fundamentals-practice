"""Experiment X1: the pilot on the validation split. See Interfaces.md (Run.py).

    uv run python src/run.py validation                       # the pilot: grid x seeds, analyses, figures
    uv run python src/run.py validation --limit-train 20000   # try things out on a training subset
    uv run python src/run.py validation --skip-existing       # resume a grid

    uv run python src/run.py test                             # the one test-set read, under the frozen claim.json

Test mode refuses to run unless claim.json is committed, and refuses to run twice.

The grid, seeds and training settings below are PROPOSED until the design review fixes them.
They are copied into results/validation_report.json so the report is self-describing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import analysis
import data
import train

EXPERIMENT = Path(__file__).resolve().parent.parent
REPO = EXPERIMENT.parent.parent
CLAIMS_PATH = REPO / "data" / "fever" / "claims.parquet"
DATA_CARD_PATH = REPO / "data" / "fever" / "data_card.json"
CLAIM_PATH = EXPERIMENT / "claim.json"

# ------------------------------------------------------- design values (PROPOSED)
FEATURE_SETTINGS = {"ngram_range": (1, 2), "min_df": 2}
GRID = {"weight_decay": [0.0, 0.01, 0.1, 1.0], "hidden_sizes": [256]}  # 1e-5..1e-3 were inert under AdamW: pilot 2026-10-15
SEEDS = [0, 1, 2]
TRAINING = {"learning_rate": 1e-2, "batch_size": 256, "max_epochs": 30, "patience": 3}  # 1e-3 under-stepped: pilot 2026-10-15, val loss still falling at epoch 16
N_RESAMPLES = 2000
SIZING_PAGE_COUNTS = [100, 200, 300, 400, 500, 616]
PROB_COLUMNS = ["p_supports", "p_refutes", "p_nei"]


def log(msg: str) -> None:
    print(f"[run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ helpers


def run_name(system: str, weight_decay: float, hidden_size: int | None, seed: int) -> str:
    wd = "0" if weight_decay == 0 else f"{weight_decay:.0e}"
    h = "" if hidden_size is None else f"_h{hidden_size}"
    return f"{system}_wd{wd}{h}_s{seed}"


def grid_runs(grid: dict, seeds: list[int]) -> list[dict]:
    systems = [("logreg", None)] + [("mlp", h) for h in grid["hidden_sizes"]]
    return [{"system": s, "weight_decay": wd, "hidden_size": h, "seed": seed}
            for s, h in systems for wd in grid["weight_decay"] for seed in seeds]


def summarize(result) -> dict:
    h = result.history
    best = h[h["epoch"] == result.best_epoch].iloc[0]
    c = result.config
    return {
        "run_name": c["run_name"], "system": c["system"], "weight_decay": c["weight_decay"],
        "hidden_size": c["hidden_size"], "seed": c["seed"],
        "best_epoch": int(result.best_epoch), "epochs_run": int(len(h)),
        "val_log_loss": float(result.best_val_loss), "val_macro_f1": float(result.best_val_macro_f1),
        "train_loss_at_best": float(best["train_loss"]),
        "generalization_gap": float(best["train_loss"] - result.best_val_loss),
        "seconds": float(h["seconds"].sum()),
    }


def save_predictions(path: Path, probabilities: np.ndarray, dataset: data.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"row": np.arange(len(dataset.labels)), "page": dataset.pages.astype(str), "label": dataset.labels})
    for i, col in enumerate(PROB_COLUMNS):
        frame[col] = probabilities[:, i]
    frame.to_parquet(path, index=False)


def load_predictions(path: Path) -> np.ndarray:
    return pd.read_parquet(path)[PROB_COLUMNS].to_numpy()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return jsonable(x.to_dict(orient="records"))
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, Path):
        return str(x)
    return x


# ------------------------------------------------------------------ figures


def figure_loss_over_epochs(results_dir: Path, figures_dir: Path, chosen: dict, seeds: list[int]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"logreg": "#2a78d6", "mlp": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("train_loss", "training loss"), ("val_loss", "validation loss"), ("val_macro_f1", "validation macro-F1")]
    for system, choice in chosen.items():
        for seed in seeds:
            path = results_dir / f"{run_name(system, choice['weight_decay'], choice['hidden_size'], seed)}.jsonl"
            if not path.exists():
                continue
            rows = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
            for ax, (col, _) in zip(axes, panels):
                ax.plot(rows["epoch"], rows[col], color=colors.get(system, "gray"), linewidth=2, alpha=0.85,
                        label=f"{system}" if seed == seeds[0] else None)
                best = rows.loc[rows["val_loss"].idxmin()]
                ax.plot([best["epoch"]], [best[col]], "o", color=colors.get(system, "gray"), markersize=5,
                        markeredgecolor="white", markeredgewidth=1.5)
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("epoch")
        ax.grid(axis="y", color="#e4e3df", linewidth=1)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, title="one line per seed; dot = best validation epoch", fontsize=9, title_fontsize=8)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "loss_over_epochs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_sizing(figures_dir: Path, sizing: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = sizing["table"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(t["pages"], t["se"], "o-", color="#2a78d6", linewidth=2, label="bootstrap SE of the difference (validation)")
    xs = np.linspace(t["pages"].min(), max(t["pages"].max(), sizing["target_pages"]), 100)
    ax.plot(xs, sizing["fit"] / np.sqrt(xs), "-", color="#8a8984", linewidth=1, label=f"fit {sizing['fit']:.3f} / sqrt(pages)")
    ax.axvline(sizing["target_pages"], color="#eb6834", linewidth=1)
    ax.annotate(f"{sizing['target_pages']} test pages\nhalf-width ≈ {sizing['expected_half_width_at_target']:.4f}",
                (sizing["target_pages"], sizing["expected_se_at_target"]), textcoords="offset points", xytext=(-8, 12),
                ha="right", fontsize=9)
    ax.set_xlabel("pages held out")
    ax.set_ylabel("standard error")
    ax.set_title("Sizing: standard error of the macro-F1 difference by pages", fontsize=11, loc="left")
    ax.grid(axis="y", color="#e4e3df", linewidth=1)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "sizing.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------- validation mode


def validation_mode(container, results_dir: Path, figures_dir: Path, grid: dict, seeds: list[int], training: dict,
                    n_resamples: int, sizing_page_counts: list[int], target_pages: int, skip_existing: bool,
                    limit_train: int = 0, execution: str | None = None) -> dict:
    """The pilot. Trains the grid, writes the summary and predictions, runs every analysis, draws the figures."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = results_dir / "predictions"
    summary_path = results_dir / "validation_summary.csv"

    execution = execution or f"pilot_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_limit{limit_train}" if limit_train else "")
    training_ds = train.subset(container.training, limit_train, seed=0) if limit_train else container.training
    validation_ds = container.validation
    device = training.get("device") or train.default_device()
    settings = {k: v for k, v in training.items() if k not in ("device", "weight_decay")}  # the grid owns weight decay

    existing = {}
    if skip_existing and summary_path.exists():
        existing = {r["run_name"]: r for r in pd.read_csv(summary_path).to_dict(orient="records")}

    rows = []
    runs = grid_runs(grid, seeds)
    for i, cfg in enumerate(runs):
        name = run_name(**cfg)
        pred_path = pred_dir / f"{name}.parquet"
        if skip_existing and name in existing and pred_path.exists():
            log(f"[{i + 1}/{len(runs)}] {name}: skipped, exists")
            rows.append(existing[name])
            continue
        log(f"[{i + 1}/{len(runs)}] {name}: training")
        result = train.train(cfg["system"], training_ds, validation_ds, run_name=name, seed=cfg["seed"],
                             weight_decay=cfg["weight_decay"], hidden_size=cfg["hidden_size"],
                             device=device, results_dir=results_dir, execution=execution, **settings)
        probs = train.predict_proba(result.model, validation_ds, device=device)
        save_predictions(pred_path, probs, validation_ds)
        row = summarize(result)
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)  # partial progress survives a crash
        log(f"    best epoch {row['best_epoch']} of {row['epochs_run']}, val loss {row['val_log_loss']:.4f}, "
            f"val macro-F1 {row['val_macro_f1']:.4f}, {row['seconds']:.0f}s")
        del result
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)

    # ---- analyses on the chosen configurations
    chosen = analysis.select_configuration(summary, criterion="val_macro_f1")
    log(f"chosen: {chosen}")
    labels = validation_ds.labels
    pages = validation_ds.pages

    def preds_of(system: str, seed: int) -> np.ndarray:
        c = chosen[system]
        return load_predictions(pred_dir / f"{run_name(system, c['weight_decay'], c['hidden_size'], seed)}.parquet")

    by_page, by_claim = {}, {}
    for seed in seeds:
        a = preds_of("mlp", seed).argmax(axis=1)
        b = preds_of("logreg", seed).argmax(axis=1)
        r = analysis.paired_bootstrap_by_page(labels, pages, a, b, n_resamples=n_resamples, seed=seed)
        by_page[str(seed)] = {k: v for k, v in r.items() if k != "draws"}
        r = analysis.paired_bootstrap_by_claim(labels, a, b, n_resamples=n_resamples, seed=seed)
        by_claim[str(seed)] = {k: v for k, v in r.items() if k != "draws"}
        log(f"seed {seed}: difference {by_page[str(seed)]['difference']:+.4f}, page CI "
            f"[{by_page[str(seed)]['ci_low']:+.4f}, {by_page[str(seed)]['ci_high']:+.4f}]")

    sd = analysis.seed_differences(summary, chosen)
    seed_diff = {"per_seed": sd.to_dict(orient="records"), "mean_difference": sd.attrs["mean_difference"],
                 "sd_difference": sd.attrs["sd_difference"]}

    first = seeds[0]
    icc, calibration, per_class = {}, {}, {}
    for system in chosen:
        p = preds_of(system, first)
        icc[system] = analysis.intra_page_correlation(p.argmax(axis=1) == labels, pages)
        calibration[system] = analysis.calibration_bins(p, labels)
        per_class[system] = analysis.per_class_report(labels, p.argmax(axis=1))

    sizing = analysis.sizing(labels, pages, preds_of("mlp", first).argmax(axis=1), preds_of("logreg", first).argmax(axis=1),
                             page_counts=sizing_page_counts, target_pages=target_pages,
                             n_resamples=max(100, n_resamples // 2), n_repeats=3, seed=first)
    log(f"sizing: expected half-width at {target_pages} pages {sizing['expected_half_width_at_target']:.4f}")

    figure_loss_over_epochs(results_dir, figures_dir, chosen, seeds)
    figure_sizing(figures_dir, sizing)

    report = jsonable({
        "mode": "validation",
        "execution": execution,
        "constants": {"grid": grid, "seeds": seeds, "training": training, "features": FEATURE_SETTINGS,
                      "n_resamples": n_resamples, "sizing_page_counts": sizing_page_counts, "target_pages": target_pages,
                      "limit_train": limit_train, "device": device},
        "n_train": int(training_ds.matrix.shape[0]),
        "n_validation": int(validation_ds.matrix.shape[0]),
        "n_features": int(training_ds.matrix.shape[1]),
        "vocabulary_size": int(len(container.vectorizer.vocabulary_)),
        "chosen": chosen,
        "bootstrap_by_page": by_page,
        "bootstrap_by_claim": by_claim,
        "seed_differences": seed_diff,
        "intra_page_correlation": icc,
        "sizing": sizing,
        "calibration": calibration,
        "per_class": per_class,
    })
    (results_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    log(f"report written: {results_dir / 'validation_report.json'}")
    return report


# ---------------------------------------------------------------- test mode


def verdict(ci_low: float, ci_high: float) -> str:
    """The declared decision rule: supported when the interval excludes 0 in the MLP's favor."""
    return "supported" if ci_low > 0 else "not established"


CLAIM_KEYS = ["frozen", "question", "hypothesis", "decision_rule", "features", "training", "systems", "seeds", "bootstrap", "checks"]


def load_claim(path: Path) -> dict:
    claim = json.loads(Path(path).read_text())
    missing = [k for k in CLAIM_KEYS if k not in claim]
    if missing:
        raise ValueError(f"claim.json is missing {missing}")
    f = claim["features"]
    if tuple(f["ngram_range"]) != tuple(FEATURE_SETTINGS["ngram_range"]) or f["min_df"] != FEATURE_SETTINGS["min_df"]:
        raise ValueError(f"claim.json features {f} differ from run.py FEATURE_SETTINGS {FEATURE_SETTINGS}")
    if not claim["seeds"]:
        raise ValueError("claim.json declares no seeds")
    for system in ["logreg", "mlp"]:
        if system not in claim["systems"]:
            raise ValueError(f"claim.json has no configuration for {system}")
    return claim


def claim_is_committed(path: Path) -> bool:
    """Tracked by git with no uncommitted changes: the freeze commit is the authority."""
    cwd = Path(path).parent
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=cwd, capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(path)], cwd=cwd, capture_output=True, text=True).stdout.strip()
    return tracked and not dirty


def figure_test_loss(results_dir: Path, figures_dir: Path, names: list[str]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"logreg": "#2a78d6", "mlp": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("train_loss", "training loss"), ("val_loss", "validation loss"), ("val_macro_f1", "validation macro-F1")]
    seen = set()
    for name in names:
        path = results_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
        system = rows["system"].iloc[0]
        for ax, (col, _) in zip(axes, panels):
            ax.plot(rows["epoch"], rows[col], color=colors.get(system, "gray"), linewidth=2, alpha=0.85, label=None if system in seen else system)
        seen.add(system)
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title, fontsize=11, loc="left"); ax.set_xlabel("epoch"); ax.grid(axis="y", color="#e4e3df", linewidth=1)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "test_loss_over_epochs.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def test_mode(container, claim_only_container, claim: dict, results_dir: Path, figures_dir: Path) -> dict:
    """The one test-set read. Trains the declared configurations, scores the test split once per model,
    runs the declared bootstrap, writes the verdict. Refuses to run if results_dir already exists."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    if results_dir.exists():
        raise RuntimeError(f"{results_dir} exists: the test split is read once. Rerunning is a Deviation to record.")
    results_dir.mkdir(parents=True)
    pred_dir = results_dir / "predictions"
    ckpt_dir = results_dir / "checkpoints"

    tr = claim["training"]
    device = tr.get("device") or train.default_device()
    settings = {"learning_rate": tr["learning_rate"], "batch_size": tr["batch_size"], "max_epochs": tr["max_epochs"],
                "patience": tr["patience"], "device": device, "results_dir": results_dir}
    seeds = list(claim["seeds"])
    n_resamples = int(claim["bootstrap"]["n_resamples"])
    execution = f"test_{time.strftime('%Y%m%d_%H%M%S')}"
    test = container.test
    labels, pages = test.labels, test.pages

    rows, probs, names = [], {}, []

    def fit_and_score(system, cfg, seed, training_ds, validation_ds, test_ds, name):
        log(f"{name}: training")
        r = train.train(system, training_ds, validation_ds, run_name=name, seed=seed, weight_decay=cfg["weight_decay"],
                        hidden_size=cfg["hidden_size"], execution=execution, **settings)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(r.model.state_dict(), ckpt_dir / f"{name}.pt")      # the weights behind this model's test number
        p = train.predict_proba(r.model, test_ds, device=device)      # the test-set read for this model
        save_predictions(pred_dir / f"{name}.parquet", p, test_ds)
        m = train.evaluate(p, test_ds.labels)
        h = r.history; best = h[h["epoch"] == r.best_epoch].iloc[0]
        rows.append({"run_name": name, "system": system, "seed": seed, "weight_decay": cfg["weight_decay"],
                     "hidden_size": cfg["hidden_size"], "best_epoch": r.best_epoch, "epochs_run": len(h),
                     "val_log_loss": r.best_val_loss, "val_macro_f1": r.best_val_macro_f1,
                     "test_macro_f1": m["macro_f1"], "test_log_loss": m["log_loss"], "test_accuracy": float((p.argmax(1) == test_ds.labels).mean()),
                     "train_loss_at_best": float(best["train_loss"]), "generalization_gap": float(best["train_loss"] - r.best_val_loss),
                     "seconds": float(h["seconds"].sum())})
        pd.DataFrame(rows).to_csv(results_dir / "test_summary.csv", index=False)
        log(f"    test macro-F1 {m['macro_f1']:.4f}, log loss {m['log_loss']:.4f}, best epoch {r.best_epoch} of {len(h)}")
        names.append(name)
        return p

    for seed in seeds:
        for system in ["logreg", "mlp"]:
            cfg = claim["systems"][system]
            name = run_name(system, cfg["weight_decay"], cfg["hidden_size"], seed) + "_test"
            probs[(system, seed)] = fit_and_score(system, cfg, seed, container.training, container.validation, test, name)
    cfg = claim["systems"]["logreg"]
    claim_only_probs = fit_and_score("logreg", cfg, seeds[0], claim_only_container.training, claim_only_container.validation,
                                     claim_only_container.test, f"claim_only_wd{cfg['weight_decay']}_s{seeds[0]}_test")

    # ---- the primary result
    A = [probs[("mlp", s)].argmax(1) for s in seeds]
    B = [probs[("logreg", s)].argmax(1) for s in seeds]
    primary = analysis.paired_bootstrap_by_page_seeds(labels, pages, A, B, n_resamples=n_resamples, seed=seeds[0])
    per_seed = []
    for s_, a, b in zip(seeds, A, B):
        r = analysis.paired_bootstrap_by_page(labels, pages, a, b, n_resamples=n_resamples, seed=s_)
        per_seed.append({"seed": s_, **{k: v for k, v in r.items() if k != "draws"}})
    diffs = np.array(primary["per_seed"])
    seed_spread = {"mean_difference": float(diffs.mean()), "sd_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0}
    secondary = {}
    for s_, a, b in zip(seeds, A, B):
        r = analysis.paired_bootstrap_by_claim(labels, a, b, n_resamples=n_resamples, seed=s_)
        secondary[str(s_)] = {k: v for k, v in r.items() if k != "draws"}

    # ---- the four-row table (systems at the first seed; every seed is in test_summary.csv)
    first = seeds[0]
    majority = np.full(len(labels), int(np.bincount(container.training.labels, minlength=3).argmax()))
    maj_probs = np.full((len(labels), 3), 1e-6); maj_probs[np.arange(len(labels)), majority] = 1 - 2e-6
    table, confusion, calibration = {}, {}, {}
    for key, p in [("mlp", probs[("mlp", first)]), ("logreg", probs[("logreg", first)]), ("majority", maj_probs), ("claim_only", claim_only_probs)]:
        pred = p.argmax(1)
        m = train.evaluate(p, labels)
        pc = analysis.per_class_report(labels, pred)
        table[key] = {"macro_f1": m["macro_f1"], "log_loss": m["log_loss"], "accuracy": pc["accuracy"], "per_class": pc["report"]}
        confusion[key] = pc["confusion"]
        calibration[key] = analysis.calibration_bins(p, labels)
    gap = {r["run_name"]: r["generalization_gap"] for r in rows}

    figure_test_loss(results_dir, figures_dir, names)
    report = jsonable({
        "mode": "test", "execution": execution, "claim": claim,
        "n_test": int(len(labels)), "n_test_pages": int(len(set(pages.tolist()))),
        "primary": {k: v for k, v in primary.items() if k != "draws"},
        "per_seed": per_seed, "seed_spread": seed_spread, "secondary_by_claim": secondary,
        "table": table, "confusion": confusion, "calibration": calibration, "generalization_gap": gap,
        "verdict": verdict(primary["ci_low"], primary["ci_high"]),
    })
    (results_dir / "test_report.json").write_text(json.dumps(report, indent=2))
    log(f"PRIMARY: mean difference {primary['difference']:+.4f}, 95% page-unit CI [{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}] -> {report['verdict']}")
    log(f"report written: {results_dir / 'test_report.json'}")
    return report


# -------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["validation", "test"])
    ap.add_argument("--limit-train", type=int, default=0, help="train on a subset of this many rows (0 = all)")
    ap.add_argument("--skip-existing", action="store_true", help="reuse runs whose predictions are already saved")
    ap.add_argument("--results", type=Path, default=EXPERIMENT / "results")
    ap.add_argument("--figures", type=Path, default=EXPERIMENT / "figures")
    args = ap.parse_args(argv)

    if args.mode == "test":
        if not CLAIM_PATH.exists():
            raise RuntimeError(f"{CLAIM_PATH} does not exist: freeze claim.json before the test-set read")
        if not claim_is_committed(CLAIM_PATH):
            raise RuntimeError("claim.json has uncommitted changes or is untracked: commit the freeze first")
        claim = load_claim(CLAIM_PATH)
        log("loading claims and building features")
        frame = pd.read_parquet(CLAIMS_PATH)
        tr, va, te = data.return_dataset_splits(frame)
        container = data.build_dataset_features(tr, va, te, claim_only=False, **FEATURE_SETTINGS)
        claim_only = data.build_dataset_features(tr, va, te, claim_only=True, **FEATURE_SETTINGS)
        test_mode(container, claim_only, claim, results_dir=args.results / "test", figures_dir=args.figures)
        return 0

    log("loading claims and building features")
    frame = pd.read_parquet(CLAIMS_PATH)
    tr, va, te = data.return_dataset_splits(frame)
    container = data.build_dataset_features(tr, va, te, claim_only=False, **FEATURE_SETTINGS)
    log(f"features: V = {len(container.vectorizer.vocabulary_):,}, F = {container.training.matrix.shape[1]:,}")
    target_pages = json.loads(DATA_CARD_PATH.read_text())["split"]["pages_per_split"]["test"]

    validation_mode(container, args.results, args.figures, GRID, SEEDS, TRAINING, N_RESAMPLES, SIZING_PAGE_COUNTS,
                    target_pages, args.skip_existing, limit_train=args.limit_train)
    return 0


if __name__ == "__main__":
    sys.exit(main())
