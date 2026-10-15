# ai-ml-fundamentals-practice

Exercises and experiments I wrote while studying ML fundamentals.

The tasks were set by an AI tutor (Claude). I wrote the rep code without AI code generation; the tutor reviewed each commit after I pushed it. For experiment X1 the data-assembly script and most of the pipeline code were written agentically by the tutor at my request, with the model code and the design decisions mine; its README records who wrote what. Reps are exercises. Only the items under Experiments are findings.

## Reps

| Rep | Mechanism | Level | Status | Checked against |
| --- | --- | --- | --- | --- |
| `F04D-L1` | Paired bootstrap by group | 1 | complete | `scipy.stats.bootstrap` |
| `F04A-L1` | Standardization fitted on the training set; group-split check | 1 | complete | `sklearn.preprocessing.StandardScaler` |
| `F04B-L1` | Confusion matrix, per-class precision, recall, F1, macro-F1 and log loss | 1 | complete | `sklearn.metrics.precision_recall_fscore_support`, `f1_score`, `log_loss` |
| `F04C-L1` | One optimizer step with L2 regularization; model selection on the validation set | 1 | complete | `torch.optim.SGD(weight_decay=...)`, `numpy.argmin` |
| `F01-1-L1` | Matrix multiplication from loops, with batch broadcasting | 1 | complete | `torch.matmul` |
| `F01-2-L1` | Stable softmax along a chosen axis | 1 | complete | `torch.softmax` |
| `F02-1-L1` | Linear-model MSE gradients from their formulas; one SGD step on a list of parameters; one mini-batch epoch | 1 | complete | `torch.nn.functional.mse_loss` with `backward()`, `torch.optim.SGD` |
| `F03-1-L1` | Logistic regression: stable sigmoid, binary cross-entropy from the logits, its gradient; 500 full-batch SGD steps | 1 | complete | `torch.nn.BCEWithLogitsLoss`, `torch.nn.Linear`, `torch.optim.SGD` |
| `F03-2-L1` | Softmax cross-entropy from the logits and its gradient; gradients of a linear classifier and of a candidate scorer with one shared weight; full-batch SGD training of both | 1 | complete | `torch.nn.CrossEntropyLoss`, `torch.nn.Linear`, `torch.optim.SGD` |
| `F03-3-L1` | One-hidden-layer MLP with ReLU: forward pass and backward pass by hand (the four parameter gradients, the ReLU mask); 1,000 full-batch SGD steps beside a single-layer baseline | 1 | complete | `torch.nn.Sequential(nn.Linear, nn.ReLU, nn.Linear)`, `torch.nn.BCEWithLogitsLoss`, `torch.optim.SGD` with autograd |

## Experiments

| Experiment | Finding | Verdict | Units | Status |
| --- | --- | --- | --- | --- |
| [`X1`](experiments/X1_fever_model_comparison/README.md) | On FEVER claim verification with the page given, a one-hidden-layer MLP did not outperform multinomial logistic regression on the same TF-IDF features: macro-F1 difference -0.028, 95% page-unit interval [-0.033, -0.022], lower on every seed. Reading the evidence added 0.013 macro-F1 over the claim text alone; every inspected failure turned on a token relation the bag-of-words input does not encode. | not established | 615 held-out Wikipedia pages (24,648 claims), 3 training seeds, 2,000 page resamples | complete |

## Layout

```
src/evalkit/       the toolkit: one mechanism per function
tests/             one test file per toolkit module; these hold the pass checks
reps/<id>_<name>/  README.md (the rep card), rep.py, results/
experiments/<id>_<name>/  README.md (the write-up), claim.json, run.py, results/, figures/
scripts/           dataset download at a pinned revision
data/              ignored by git
```

## Setup

```bash
uv sync
uv run pytest
```
