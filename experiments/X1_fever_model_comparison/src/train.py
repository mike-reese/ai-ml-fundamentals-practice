"""Training loop for experiment X1. See Interfaces.md (Train.py).

train() fits one configuration of one system on the training split, scores the validation split
after every epoch, stops early on validation log loss, restores the best epoch's weights, and
appends one JSON line per epoch to <results_dir>/<run_name>.jsonl as it goes.

The test split cannot be passed to anything here except predict_proba, which the evaluation code
calls once. Softmax is applied in predict_proba and nowhere else.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, log_loss
from torch import nn

from data import Dataset
from models import MultinomialLogisticRegression, OneHiddenLayerMLP

DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results"
N_CLASSES = 3


@dataclass
class TrainResult:
    model: nn.Module          # on the device, best-epoch weights restored, in eval mode
    history: pd.DataFrame     # one row per epoch: epoch, train_loss, val_loss, val_macro_f1, seconds
    best_epoch: int           # 1-based
    best_val_loss: float
    best_val_macro_f1: float
    config: dict = field(default_factory=dict)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------- data


def subset(dataset: Dataset, n: int, seed: int) -> Dataset:
    """n rows drawn without replacement; matrix, labels and pages sliced by the same index."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(dataset.matrix.shape[0], size=n, replace=False))
    return Dataset(matrix=dataset.matrix[idx], labels=dataset.labels[idx], pages=dataset.pages[idx])


def batches(dataset: Dataset, batch_size: int, rng: np.random.Generator | None = None,
            device: str | torch.device = "cpu") -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (x, y) per batch: x a float32 sparse CSR tensor (b, F) on the device, y int64 (b,).

    With rng, the row order is one permutation drawn from it; without, Dataset order.
    Nothing is densified: the models multiply sparse rows by their dense weights.
    """
    n = dataset.matrix.shape[0]
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        x = to_sparse_tensor(dataset.matrix[idx], device)
        y = torch.from_numpy(dataset.labels[idx]).to(device)
        yield x, y


def to_sparse_tensor(rows, device) -> torch.Tensor:
    """A scipy CSR block as a torch sparse CSR tensor on the device. Only the nonzeros travel:
    a 256-row TF-IDF batch is about 15,000 values, not 256 x 586,876 zeros."""
    rows = rows.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(rows.indptr).long(), torch.from_numpy(rows.indices).long(),
        torch.from_numpy(rows.data).float(), size=rows.shape, check_invariants=False,  # scipy built them
    ).to(device)


# -------------------------------------------------------------- evaluation


@torch.no_grad()
def predict_proba(model: nn.Module, dataset: Dataset, batch_size: int = 1024,
                  device: str | torch.device | None = None) -> np.ndarray:
    """Softmax of the model's logits, (N, 3) float64, rows in Dataset order."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for x, _ in batches(dataset, batch_size, rng=None, device=device):
        out.append(torch.softmax(model(x), dim=1).cpu().numpy())
    probs = np.vstack(out).astype(np.float64)
    return probs / probs.sum(axis=1, keepdims=True)  # float32 softmax rows sum to 1 only to ~1e-7; make it exact


def evaluate(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """log_loss and macro_f1 from scikit-learn, so every score in the experiment uses one definition."""
    classes = list(range(N_CLASSES))
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "macro_f1": float(f1_score(labels, probabilities.argmax(axis=1), average="macro", labels=classes)),
    }


# ---------------------------------------------------------------- training


def build_model(system: str, n_features: int, hidden_size: int | None) -> nn.Module:
    if system == "logreg":
        if hidden_size is not None:
            raise ValueError("hidden_size must be None for system='logreg'")
        return MultinomialLogisticRegression(n_features, N_CLASSES)
    if system == "mlp":
        if hidden_size is None:
            raise ValueError("hidden_size is required for system='mlp'")
        return OneHiddenLayerMLP(n_features, hidden_size, N_CLASSES)
    raise ValueError(f"unknown system {system!r}; expected 'logreg' or 'mlp'")


def train(
    system: str,
    training: Dataset,
    validation: Dataset,
    run_name: str,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    hidden_size: int | None = None,
    batch_size: int = 256,
    max_epochs: int = 30,
    patience: int = 3,
    device: str | torch.device | None = None,
    results_dir: Path = DEFAULT_RESULTS,
    execution: str | None = None,
) -> TrainResult:
    """Fit one configuration. See the module docstring and Interfaces.md."""
    device = device or default_device()
    n_features = training.matrix.shape[1]

    # Seed before the model is built so the initial weights are part of the seed, and keep the
    # batch order on its own generator so it does not depend on how many torch draws the model made.
    torch.manual_seed(seed)
    shuffle_rng = np.random.default_rng(seed)
    model = build_model(system, n_features, hidden_size).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # decoupled decay, as the spec names
    loss_fn = nn.CrossEntropyLoss()

    config = {
        "run_name": run_name, "system": system, "seed": seed, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "hidden_size": hidden_size, "batch_size": batch_size,
        "max_epochs": max_epochs, "patience": patience, "n_features": n_features, "n_parameters": n_parameters,
        "execution": execution,  # which invocation of run.py or checks.py made this run; groups runs on the dashboard
    }
    record_keys = ["run_name", "system", "seed", "learning_rate", "weight_decay", "hidden_size", "batch_size", "execution"]

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    record_path = results_dir / f"{run_name}.jsonl"
    record_path.write_text("")  # a rerun of the same name starts over

    history = []
    best_val_loss, best_epoch, best_state, best_f1 = np.inf, 0, None, 0.0
    since_best = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for x, y in batches(training, batch_size, rng=shuffle_rng, device=device):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        train_loss = total / count

        metrics = evaluate(predict_proba(model, validation, device=device), validation.labels)
        val_loss, val_f1 = metrics["log_loss"], metrics["macro_f1"]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch, best_f1 = val_loss, epoch, val_f1
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_macro_f1": val_f1, "seconds": time.time() - t0}
        history.append(row)
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**{k: config[k] for k in record_keys}, **row, "best_so_far": is_best}) + "\n")

        if since_best >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model, history=pd.DataFrame(history), best_epoch=best_epoch,
        best_val_loss=float(best_val_loss), best_val_macro_f1=float(best_f1), config=config,
    )
