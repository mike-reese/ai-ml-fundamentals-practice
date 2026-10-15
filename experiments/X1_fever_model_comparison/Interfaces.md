# Interfaces - Planned Beforehand

## Data.py
1. What does it compute? Two things, so two functions. One loads the assembled dataset and returns the three splits as frames. One builds features: fits the vectorizer and the pair-feature scaler on the training split only, transforms all three, and returns the input matrix and the integer label vector for each split, plus the fitted vectorizer so its vocabulary size can be reported.
2. What must the caller supply? The path to claims.parquet; the vectorizer settings, n-gram range and minimum document frequency; whether to build the full input or the claim block alone, for the claim-only baseline.
3. What comes back? Per split: a sparse matrix of shape (N, 2V + 2) or (N, V), a label vector of length N with values 0, 1, 2 in the order SUPPORTS, REFUTES, NOT ENOUGH INFO, and the page id per row, which the bootstrap needs later. Plus the fitted vectorizer.
4. Types? scipy sparse CSR for the matrices, numpy for labels and page ids. Conversion to torch happens in the training code at the batch boundary, since a 115,000 by 279,000 dense tensor would not fit in memory.

### Interface
Return Splits
Name:
  return_dataset_splits
arguments:
  dataset: A pandas dataframe, the assembled dataset loaded from data/fever/claims.parquet with pandas.read_parquet. run.py loads the frame and passes it; the function does not read the file.
returns:
  training_dataset: A pandas dataframe containing the partition of the dataset where split == "training"
  validation_dataset: A pandas dataframe containing the partition of the dataset where split == "validation"
  test_dataset: A pandas dataframe containing the partition of the dataset where split == "test"

Build Features
Name:
  build_dataset_features
arguments:
  training_dataset: A pandas dataframe, the training split from return_dataset_splits. The vectorizer and the pair-feature scaler are fitted on this frame only.
  validation_dataset: A pandas dataframe, the validation split. Transformed with the fitted vectorizer and scaler.
  test_dataset: A pandas dataframe, the test split. Transformed with the fitted vectorizer and scaler.
  ngram_range: A tuple (min_n, max_n) passed to TfidfVectorizer. Design value (1, 2).
  min_df: An integer passed to TfidfVectorizer, the minimum number of training texts a term must appear in. Design value 2.
  claim_only: A bool, default False. False builds the full input, claim block, evidence block and the two pair features, shape (N, 2V + 2). True builds the claim block alone, shape (N, V), for the claim-only baseline, from the same vocabulary.
returns:
  DatasetContainer: A dataclass containing a Dataset (matrix, labels, pages) for each of training, validation and test, and the fitted vectorizer so V can be reported.

## Models.py
1. What does it compute? The two compared systems as torch modules. Each maps one batch of feature rows to one row of three logits per claim. Multinomial logistic regression is one linear layer from F inputs to 3 logits. The MLP is a linear layer from F inputs to H hidden units, ReLU, then a linear layer from H to 3 logits. Neither applies softmax: the loss (cross-entropy) and the probability outputs (calibration, log loss) take the logits, and softmax belongs with them in the training and evaluation code, not in the model.
2. What must the caller supply? The input width F, which is 2V + 2 for the full input or V for the claim-only baseline, read from the matrix built by data.py. For the MLP, the hidden size H. Weight decay, learning rate and the seed are training settings, not model settings; the caller seeds torch before constructing a model so its initial weights are reproducible.
3. What comes back? A (batch, 3) float32 tensor of logits, column order SUPPORTS, REFUTES, NOT ENOUGH INFO, matching the label codes from data.py. The parameters are plain nn.Linear weights and biases, so the logistic regression's weight matrix is (F, 3) with 3F + 3 parameters, and the MLP's two matrices are (F, H) and (H, 3).
4. Types? Input is a torch float32 tensor of shape (batch, F), sparse CSR or dense; the output is the same either way. The training loop sends each batch as a sparse CSR tensor, because a TF-IDF row is about 60 nonzeros out of 586,876 and densifying a batch (1.2 GB) cost 550 ms of memory traffic per batch against a few ms of arithmetic (pilot 2026-10-15). The models use torch.sparse.mm for sparse input, which differentiates with respect to the dense weights. Output is a torch float32 tensor (batch, 3). Both modules work on CPU or GPU; the caller moves the model and each batch with .to(device).

### Interface
Multinomial Logistic Regression
Name:
  MultinomialLogisticRegression (nn.Module)
constructor arguments:
  n_features: An integer F, the width of the input rows. 2V + 2 for the full input, V for the claim-only baseline.
  n_classes: An integer, default 3. The number of logits. Kept as an argument so the module is not tied to FEVER's three labels, but X1 always uses 3.
forward arguments:
  x: A torch float32 tensor of shape (batch, n_features).
returns:
  logits: A torch float32 tensor of shape (batch, n_classes). No softmax applied.

One-Hidden-Layer MLP
Name:
  OneHiddenLayerMLP (nn.Module)
constructor arguments:
  n_features: An integer F, as above.
  hidden_size: An integer H, the width of the hidden layer. A design value chosen on the validation set and recorded in the README.
  n_classes: An integer, default 3.
forward arguments:
  x: A torch float32 tensor of shape (batch, n_features).
returns:
  logits: A torch float32 tensor of shape (batch, n_classes). No softmax applied.

Shared behavior
  Both modules use torch's default nn.Linear initialization. Both hold no dropout, no normalization layer and no activation on the output. The MLP contains the logistic regression as a special case (an identity hidden layer), which is what makes the comparison isolate the hidden layer.
  Complete when: for each module, constructing it with n_features equal to the matrix width from data.py and calling it on one dense batch returns a (batch, 3) tensor with no NaN, and the parameter count matches 3F + 3 for the logistic regression and FH + H + 3H + 3 for the MLP.

## Train.py
1. What does it compute? One trained model from one configuration. The loop: each epoch, shuffle the training row indices with a seeded generator, take them in batches, slice those rows from the CSR matrix, densify the slice, move it to the device, forward, cross-entropy on the logits, backward, optimizer step. After each epoch, score the validation split (log loss and macro-F1), append one row to the record, and stop early when validation log loss has not improved for `patience` epochs. The weights from the best validation epoch are restored before returning, so the returned model is the one the record says is best, not the last one. The same function serves both systems and every configuration in the validation grid; which system is a string argument, and the model is constructed inside the function after the seed is set so its initial weights are part of the seed.
2. What must the caller supply? The system name, the training and validation Datasets from data.py, and the configuration: seed, learning rate, weight decay, hidden size (MLP only), batch size, maximum epochs, patience. A run name, under which the record is saved. Optionally the device; the default is cuda when available. The test Dataset is not an argument: train.py has no way to receive it, which is how the one-test-read rule is enforced in code rather than by discipline.
3. What comes back? A TrainResult: the model with its best-epoch weights, the per-epoch history as a frame, the best epoch number and its validation log loss and macro-F1, and the configuration as a dict. The history is also on disk, one JSON line per epoch, at results/<run_name>.jsonl, so a crash mid-grid loses nothing finished and the loss-over-epochs plot reads from the files, not from memory. Training loss per epoch is the mean of the batch losses in that epoch; validation loss is the log loss over the whole split under no_grad, which is what the early stopping and the plot use.
4. Types? Datasets as data.py returns them: CSR matrix, int64 labels, str pages. Inside the loop, each batch is a torch float32 sparse CSR tensor (batch, F) and an int64 tensor (batch,) on the device, built from the scipy row slice with torch.sparse_csr_tensor; only the nonzeros are copied to the GPU. The record rows are plain Python scalars so they serialize as JSON. Metrics come from scikit-learn: f1_score(average="macro") and log_loss on the (N, 3) softmax probabilities, so the torch model and the scikit-learn check model are scored by the same code.

Where the pieces that are not here live: softmax over the logits is applied in predict_proba, below, and nowhere else. The scikit-learn LogisticRegression check and the shuffled-label check are tests in test_x1.py; they call train() and compare. The loss-over-epochs figure is drawn by the report code from the results/ files.

### Interface
Train
Name:
  train
arguments:
  system: A string, "logreg" or "mlp". Selects MultinomialLogisticRegression or OneHiddenLayerMLP from models.py.
  training: A Dataset from data.py, the training split. The only split whose labels reach the optimizer.
  validation: A Dataset from data.py, the validation split. Scored after each epoch; its labels drive early stopping and nothing else.
  run_name: A string. The record is written to results/<run_name>.jsonl, overwriting a previous run of the same name.
  seed: An integer. Seeds torch and the shuffle generator before the model is constructed, so initial weights and batch order are both reproducible.
  learning_rate: A float, passed to the optimizer (AdamW).
  weight_decay: A float, passed to AdamW as decoupled weight decay (the L2 regularization strength). The grid variable chosen on the validation set.
  hidden_size: An integer or None. Required for "mlp", must be None for "logreg".
  batch_size: An integer, default 256. The number of rows densified at a time; the main lever on GPU memory.
  max_epochs: An integer, default 30. Upper bound; early stopping usually ends sooner.
  patience: An integer, default 3. Epochs without a validation log loss improvement before stopping.
  device: A string or torch.device, default "cuda" if available else "cpu".
  execution: A string or None. The invocation that made this run (run.py and checks.py set one per invocation, e.g. pilot_20261015_1430_limit20000); written on every record row so the dashboard can group runs by execution.
  results_dir: A directory path, default results/ in the experiment directory. Where <run_name>.jsonl is written; the tests pass a temporary directory so real runs are never overwritten.
returns:
  TrainResult: A dataclass with fields model (the nn.Module on the device, best-epoch weights restored, in eval mode), history (a pandas frame, one row per epoch: epoch, train_loss, val_loss, val_macro_f1, seconds), best_epoch (int, 1-based), best_val_loss (float), best_val_macro_f1 (float), config (dict of every argument above except the Datasets, plus n_features and the parameter count).

Batches
Name:
  batches
arguments:
  dataset: A Dataset.
  batch_size: An integer.
  rng: A numpy Generator or None. With a generator, row order is shuffled by it (one permutation per call); with None, rows come in Dataset order, which predict_proba relies on.
  device: As above.
yields:
  (x, y): x a torch float32 sparse CSR tensor (b, F) on the device holding only the batch's nonzeros; y an int64 tensor (b,) of their labels. Every row appears exactly once per pass; the last batch is smaller when N is not a multiple of batch_size. Nothing is densified anywhere in training.

Predict Probabilities
Name:
  predict_proba
arguments:
  model: An nn.Module from models.py, already on the device.
  dataset: A Dataset from data.py. Any split; this is the one function that will later see the test split, called once by the evaluation code.
  batch_size: An integer, default 1024. Larger than training because no gradients are held.
  device: As above.
returns:
  probabilities: A numpy float64 array (N, 3), softmax of the logits, rows in the Dataset's row order, each row summing to 1. The only place softmax is applied.

Evaluate
Name:
  evaluate
arguments:
  probabilities: The (N, 3) array from predict_proba.
  labels: The int64 (N,) label codes from the Dataset.
returns:
  metrics: A dict with log_loss (float) and macro_f1 (float, prediction is the argmax of each row). Used after every epoch on the validation split, and by the test-set read and the bootstrap, so one definition of each metric serves everywhere.

Subset
Name:
  subset
arguments:
  dataset: A Dataset.
  n: An integer, the number of rows to keep.
  seed: An integer for the row draw.
returns:
  A Dataset of n rows drawn without replacement, matrix, labels and pages sliced together. For the pilot runs and the scikit-learn check, which train on a subset of the training split.

Record row (one JSON object per line in results/<run_name>.jsonl)
  run_name, system, seed, learning_rate, weight_decay, hidden_size, batch_size, execution, epoch, train_loss, val_loss, val_macro_f1, seconds, best_so_far (bool). The configuration is repeated on every row so a file is self-describing when read alone.

Tests (tests/test_train.py) exercise every function on tiny synthetic Datasets on the CPU and never read data/fever or run a real training job; the scikit-learn comparison below is a one-time check run by checks.py on real data and recorded in the README.

Complete when: train("logreg", ...) and train("mlp", ...) on a subset of the training split both return a TrainResult whose history has at least one row and whose val_loss decreases over the first epochs; and predict_proba of the torch logistic regression on the validation split gives a macro-F1 within 0.02 of sklearn.linear_model.LogisticRegression fitted on the same subset's matrix and scored by the same evaluate().

## Dashboard.py
1. What does it compute? Nothing about the models. It serves a local page that shows training runs while they are in progress. The page polls the results/ directory every few seconds, reads every <run_name>.jsonl that train.py is appending to, and redraws two charts: loss per epoch (training and validation) and validation macro-F1 per epoch, one series per run, with the best-so-far epoch marked. It reads the same files the report's loss-over-epochs figure will read, so the live view and the final figure cannot disagree.
2. What must the caller supply? Nothing required. Optionally the results directory (default results/ beside src/) and the port (default 8765). Run it in a second terminal: `uv run python src/dashboard.py`, then open http://localhost:8765.
3. What comes back? Two HTTP routes. `/` is the page. `/runs` is JSON: a list of runs, each {run_name, system, rows: [record rows in file order]}, built fresh on every request by reading the files, so there is no state to go stale. A partially written last line (train.py mid-write) is skipped, not an error.
4. Types? Plain Python http.server, no dependencies. The page is one static HTML file with inline SVG charts and no external scripts, so it works offline. Colors follow the entity: each run name keeps its color for the life of the page regardless of which runs are currently visible.

### Interface
Serve
Name:
  main (command line)
arguments:
  --results: A directory path, default results/ in the experiment directory. Every *.jsonl in it is a run.
  --port: An integer, default 8765.
routes:
  GET /       The dashboard page (src/dashboard.html).
  GET /runs   JSON list of runs as described in item 3.

Complete when: with two fake .jsonl files appended to by a script, the page shows two series per chart growing as lines arrive, the legend names both runs, hovering a point shows its row, and the table view lists every row.

## Analysis.py
1. What does it compute? The statistics of the experiment, as pure functions of arrays: macro-F1 from a confusion matrix, the paired percentile bootstrap of the macro-F1 difference resampling pages, the claim-unit bootstrap as the optional secondary line, the sizing calculation from a pilot, the intra-page correlation of correctness, the weight-decay selection from the run summary, calibration bins, and the per-class report. The bootstrap is written in numpy here, not imported from evalkit, as the spec requires.
2. What must the caller supply? Arrays aligned row for row: labels (int64 codes), pages (str), and the probabilities or predicted codes of one or two systems. For the selection, the run summary frame that run.py builds. No function here trains anything or reads a file.
3. What comes back? Plain dicts of floats and small pandas frames, ready to be written to JSON and pasted into the README. The bootstrap also returns its draws so the figure and the sizing step can reuse them.
4. Types? numpy in, dicts and frames out. The bootstrap resamples pages by drawing page counts with replacement and turning them into row weights, so one draw is a weighted confusion matrix, not a concatenation of rows; 2,000 draws on 25,000 rows take seconds.

### Interface
Macro-F1
Name:
  macro_f1
arguments:
  labels: int64 (N,) codes 0..2.
  predictions: int64 (N,) codes.
  weights: float (N,) or None. Row weights; the bootstrap passes page counts.
returns:
  float. Equals sklearn.metrics.f1_score(average="macro") when weights is None; a class with no true and no predicted rows contributes 0, as sklearn does.

Paired bootstrap by page
Name:
  paired_bootstrap_by_page
arguments:
  labels, pages: aligned arrays as above.
  predictions_a, predictions_b: int64 (N,) predicted codes of the two systems on the same rows. The difference is a minus b, so pass the MLP as a.
  n_resamples: int, default 2000.
  seed: int.
returns:
  dict: difference (float, on the full rows), ci_low, ci_high (2.5th and 97.5th percentiles of the draws), se (standard deviation of the draws), n_pages, n_resamples, draws (float (n_resamples,) array). One draw takes n_pages pages with replacement and scores both systems on the same drawn pages.

Paired bootstrap by page, several seeds
Name:
  paired_bootstrap_by_page_seeds
arguments:
  labels, pages: as above.
  predictions_a: list of int64 (N,) arrays, one per seed (the MLP); predictions_b: the same for logistic regression, same seed order.
  n_resamples, seed: as above.
returns:
  dict as paired_bootstrap_by_page, where difference is the mean over seeds of (macro_f1(a_k) - macro_f1(b_k)) and each draw resamples pages once and scores every seed's pair on those same pages before averaging. One interval for the primary number; per_seed holds each seed's full-sample difference.

Paired bootstrap by claim
Name:
  paired_bootstrap_by_claim
arguments and returns:
  As above without pages; one draw takes N rows with replacement. The optional secondary line.

Sizing
Name:
  sizing
arguments:
  labels, pages, predictions_a, predictions_b: as above, from the validation split.
  page_counts: list of ints, the numbers of pages held out to try (for example 100, 200, 400, 616).
  target_pages: int, the test split's page count, 615, from the data card.
  n_resamples: int, default 1000 per point.
  n_repeats: int, default 3 subsamples per page count, averaged.
  seed: int.
returns:
  dict: table (frame: pages, se, half_width), fit (c such that se ≈ c / sqrt(pages), least squares on the table), expected_se_at_target, expected_half_width_at_target (1.96 × se). The number the design review needs.

Intra-page correlation
Name:
  intra_page_correlation
arguments:
  correct: bool or 0/1 (N,) whether a system's prediction was right.
  pages: str (N,).
returns:
  dict: icc (one-way ANOVA intraclass correlation of correctness by page), mean_page_size, design_effect (1 + (mean_page_size − 1) × icc), n_pages. How much the page unit mattered.

Select weight decay
Name:
  select_configuration
arguments:
  summary: the run summary frame from run.py (one row per run: system, weight_decay, hidden_size, seed, val_macro_f1, val_log_loss, ...).
  criterion: "val_macro_f1" (default, the primary metric; higher is better) or "val_log_loss" (lower is better).
returns:
  dict system -> {weight_decay, hidden_size, mean, std, n_seeds}: for each system, the (weight_decay, hidden_size) whose mean over seeds is best; ties broken by the other criterion. The grid is averaged over seeds, never chosen per seed.

Seed variation
Name:
  seed_differences
arguments:
  summary, chosen: the frame and the dict above.
returns:
  frame: seed, mlp_macro_f1, logreg_macro_f1, difference; plus the mean and standard deviation of the difference as attributes mean_difference, sd_difference.

Calibration bins
Name:
  calibration_bins
arguments:
  probabilities: (N, 3); labels: (N,); n_bins: int, default 10.
returns:
  frame: bin, lower, upper, count, mean_confidence, accuracy, for the argmax probability. Bins with no rows are kept with count 0.

Per-class report
Name:
  per_class_report
arguments:
  labels, predictions.
returns:
  dict: report (sklearn classification_report as a dict, label names), confusion (3×3 list, rows true, columns predicted, label order SUPPORTS, REFUTES, NOT ENOUGH INFO), accuracy.

## Checks.py
1. What does it compute? The "Features and training" check table and the two baselines, on real data, each with its observed value, before the test set is read. Rows: vocabulary fitted on the training set only; torch logistic regression against scikit-learn's on the same features, validation macro-F1 within 0.02; shuffled training labels scoring at chance on validation, within 0.05 of 1/3; a seeded rerun reproducing the saved validation metrics to 1e-6. Baselines: the majority class of the training set, and the claim-only torch logistic regression, both scored on validation. Each row is run by its own function so one can be rerun alone.
2. What must the caller supply? Nothing required. Options: the training-subset size for the scikit-learn comparison and the shuffled-label run (default 30,000 rows; the full split with --n-train 0), the training settings (learning rate, weight decay, batch size, patience), the seed, and which rows to run. The claim-only baseline and the seeded rerun use the full training split because their numbers go in the README.
3. What comes back? results/checks.json: one entry per row with quantity, condition, tolerance, observed (a dict of the numbers), holds (bool or "reported"), and the settings used. The same table printed as markdown for the README. Exit status 1 if any row with a tolerance fails.
4. Types? Uses data.py, train.py and analysis.py only. Every training run goes through train.train with run names prefixed check_, so their epoch records land in results/ and appear on the dashboard like any other run. Never touches the test Dataset: the container's test field is not read by any function here.

### Interface
Name:
  check_vocabulary(container) -> row
  check_against_sklearn(container, n_train, settings, seed) -> row
  check_shuffled_labels(container, n_train, settings, seed) -> row
  check_seeded_rerun(container, settings, seed) -> row
  baseline_majority(container) -> row
  baseline_claim_only(claim_only_container, settings, seed) -> row
  main(argv) -> int
where
  settings: dict with learning_rate, weight_decay, batch_size, max_epochs, patience, device.
  row: dict with quantity (str), condition (str), tolerance (str), observed (dict of JSON-serializable numbers), holds (True | False | "reported"), and run_names (list of the train.py runs it made).
helpers (unit-tested on synthetic arrays):
  shuffle_labels(labels, seed) -> int64 (N,): a permutation of the labels; class counts unchanged.
  within(a, b, tolerance) -> bool.
  majority_predictions(training_labels, n) -> int64 (n,): the training majority code repeated.
  rows_to_markdown(rows) -> str: the README table.

## Run.py
1. What does it compute? The experiment, in two modes. Validation mode is the pilot: every configuration in the declared grid across the declared seeds, trained on the training split and scored on the validation split; the configuration per system chosen from the summary; the paired bootstrap of the difference by page on validation for each seed; the seed spread; the intra-page correlation; the sizing calculation; calibration and per-class tables; the loss-over-epochs figure. Test mode is the one test-set read, written after claim.json was frozen: it reads claim.json, trains the two chosen configurations with the declared seeds (and the claim-only baseline once), scores the test split once, runs the declared bootstrap and writes the verdict under the declared decision rule. Analysis functions come from analysis.py; training from train.py.
2. What must the caller supply? The mode. The grid, seeds, hidden sizes and training settings are declared constants at the top of run.py, marked PROPOSED until the design review fixes them, and are copied into the report so it is self-describing. Options: --limit-train N to pilot on a subset while trying things out (recorded in the report; the real pilot uses 0, the full split), --skip-existing to resume a grid without retraining runs whose predictions are already saved.
3. What comes back? Under results/: validation_summary.csv, one row per run (run_name, system, weight_decay, hidden_size, seed, best_epoch, epochs_run, val_log_loss, val_macro_f1, train_loss_at_best, generalization_gap, seconds); predictions/<run_name>.parquet with row, page, label and the three probabilities, in validation-split order; validation_report.json with the chosen configurations, the per-seed bootstrap intervals, the seed spread, the intra-page correlation, the sizing table and its expected half-width at 615 pages, the calibration bins and per-class reports of the chosen runs, and the constants used. Under figures/: loss_over_epochs.png (training and validation loss, and validation macro-F1, per system and seed for the chosen configurations, read from the epoch records) and sizing.png.
4. Types? Frames and JSON. Row order of every predictions file is the validation frame's order, so claim ids can be joined back by position from claims.parquet.

### Interface
Name:
  main(argv) -> int                            `uv run python src/run.py validation [--limit-train N] [--skip-existing]`
  run_name(system, weight_decay, hidden_size, seed) -> str     e.g. "mlp_wd1e-04_h256_s0", "logreg_wd0_s1"
  grid_runs(grid, seeds) -> list of dicts       every (system, weight_decay, hidden_size, seed) in the declared order
  summarize(result, config) -> dict             one validation_summary row from a TrainResult
  save_predictions(path, probabilities, dataset) -> None
  validation_mode(args) -> report dict          the pilot; writes everything listed in item 3

Test mode
Name:
  load_claim(path) -> dict                      claim.json, validated: required keys present, feature settings equal to run.py's
  test_mode(container, claim_only_container, claim, results_dir, figures_dir) -> report dict
  verdict(ci_low, ci_high) -> str               "supported" when ci_low > 0 (the interval excludes 0 in the MLP's favor), else "not established"
guards (main, mode "test"):
  claim.json must exist, be tracked by git and have no uncommitted changes (the freeze commit is the authority).
  results/test/ must not exist: the test split is read once. A second run is refused; deleting the directory to rerun is a Deviation to record.
what it trains:
  for each declared seed: the chosen logreg and the chosen mlp configuration on the full training split, early stopping on validation log loss as declared; the claim-only logreg baseline once (seed 0, chosen logreg decay) on the claim-only features.
what it reads from the test split:
  one predict_proba per trained model, 2 x seeds + 1 calls, inside this one invocation; nothing else in the experiment reads it.
what it writes (results/test/):
  predictions/<run_name>.parquet          row, page, label, p_supports, p_refutes, p_nei, test-split order
  test_summary.csv                        one row per trained run: run_name, system, seed, best_epoch, epochs_run, val_log_loss, test_macro_f1, test_log_loss, test_accuracy
  test_report.json                        the primary result: mean difference over seeds with its page-unit interval (one draw resamples pages once and scores every seed's pair on the same pages, then averages the per-seed differences), the per-seed differences and their spread, the claim-unit interval as the secondary line, the four-row table (mlp, logreg, majority, claim-only: macro-F1, accuracy, log loss, per-class precision/recall/F1/support), confusion matrices, calibration bins, the generalization gap per run, the verdict, and the claim.json contents it ran under
figures/:
  test_loss_over_epochs.png               the epoch curves of the six test-mode runs

constants (PROPOSED):
  WEIGHT_DECAY_GRID = [0.0, 0.01, 0.1, 1.0]
  HIDDEN_SIZES = [256]
  SEEDS = [0, 1, 2]
  TRAINING = {learning_rate: 1e-2, batch_size: 256, max_epochs: 30, patience: 3}
  N_RESAMPLES = 2000
  SIZING_PAGE_COUNTS = [100, 200, 300, 400, 500, 616]
