# ai-ml-fundamentals-practice

Exercises and experiments I wrote while studying ML fundamentals.

The tasks were set by an AI tutor (Claude). I wrote all the code here without AI code generation. The tutor reviewed each commit after I pushed it. Reps are exercises. Only the items under Experiments are findings.

## Reps

| Rep | Mechanism | Level | Status | Checked against |
| --- | --- | --- | --- | --- |
| `F04D-L1` | Paired bootstrap by group | 1 | complete | `scipy.stats.bootstrap` |
| `F04A-L1` | Standardization fitted on the training set; group-split check | 1 | complete | `sklearn.preprocessing.StandardScaler` |
| `F04B-L1` | Confusion matrix, per-class precision, recall, F1, macro-F1 and log loss | 1 | complete | `sklearn.metrics.precision_recall_fscore_support`, `f1_score`, `log_loss` |

## Experiments

| Experiment | Finding | Verdict | Units |
| --- | --- | --- | --- |

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
