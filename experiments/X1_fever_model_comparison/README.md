# X1 -- FEVER Model comparison -- Two-Model comparison with Cross-Entropy

Status: design. The title will be rewritten to state the finding once the test set has been read. The data-assembly parameters (evidence k = 5; split shares 0.70 / 0.15 / 0.15 of pages; split seed 20261013) were confirmed by the owner on 2026-10-15 and are the values `scripts/get_data.py` used.

## Question
Using the FEVER dataset for claim verification, does a one-hidden layer MLP beat a multinomial logistic regression on the same TF-IDF features on held-out wikipedia pages?

Both classifiers are trained on the same features and scored on the same test claims. The primary result is the MLP's macro-F1 minus logistic regression's, with one 95 percent bootstrap confidence interval that resamples pages. The MLP contains the linear model as a special case, so the comparison isolates one thing: whether a hidden layer that can learn claim-evidence interactions beats a flat boundary on features that are already very high-dimensional. The secondary analyses carry the rest of the insight: whether better predictions come with better probabilities, which label each model misses, and how much of each score comes from the claim text alone.

## Setup
### Data Card
#### Source Data
FEVER v1.0 (Fact Extraction and VERification), Thorne et al. 2018, NAACL. Two parts, both from fever.ai:

| Part | Files | Content |
| --- | --- | --- |
| Claims | `train.jsonl`, `paper_dev.jsonl`, `paper_test.jsonl` | One JSON record per claim: `id`, `label`, `claim`, `evidence` (one evidence set per annotator; each line is `[annotation_id, evidence_id, page, sentence_id]`) |
| Wikipedia text | `wiki-pages.zip` | June 2017 dump, 5,416,537 pages, 109 JSON-lines files; one record per page with `id`, `text`, and numbered `lines` |

The Hugging Face conversion `fever/fever` (parquet branch, commit `5f577157`) was not used. It maps evidence sentence 0 to -1, the same value it uses for NOT ENOUGH INFO's null, which loses 95,529 of 311,431 training evidence lines. Its Wikipedia conversion also stops at 3,741,000 of 5,416,537 pages, so 4,894 cited pages were missing. Gold recall at 5 was 0.8255 with the conversion and 0.8854 with the original files.

#### Revision
Each file is pinned by SHA-256 in `scripts/get_data.py` and recorded in `data/fever/data_card.json`. A later download that does not match the pin raises.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `train.jsonl` | 33,024,303 | `eba7e8f87076753f8494718b9a857827af7bf73e76c9e4b75420207d26e588b6` |
| `paper_dev.jsonl` | 2,168,767 | `41158707810008747946bf23471e82df53e77a513524b9e3ec1c2e674ef5ef8c` |
| `paper_test.jsonl` | 2,181,168 | `fb7b0280a0adc2302bbb29bfb7af37274fa585de3171bcf908f180642d11d88e` |
| `wiki-pages.zip` | 1,713,485,474 | `4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2` |

Claim files first downloaded 2026-10-15; the Wikipedia archive 2026-10-13.

#### License
CC BY-SA 3.0 for the annotations (claims, labels, evidence). The Wikipedia text is under Wikipedia's terms. The data lives in `data/fever/`, which git ignores; nothing from the dataset is committed.

#### Data Configuration
**Labels.** SUPPORTS, REFUTES, NOT ENOUGH INFO. Assigned by FEVER's annotators under the rule in the paper, section 3. NOT ENOUGH INFO claims carry no evidence page.

**Claims loaded.** 165,447: train 145,449; paper_dev 9,999; paper_test 9,999. By label: SUPPORTS 86,701; REFUTES 36,441; NOT ENOUGH INFO 42,305. The training file is 55 / 20 / 25 percent by label; paper_dev and paper_test are exactly balanced at 3,333 per label. The three files are pooled and re-split, so the balance of our splits follows the pooled mix.

**Group: the primary page.** For a SUPPORTS or REFUTES claim, the one page present in every annotator's evidence set (Observed). When the intersection holds more than one page, the first-listed page of the first annotation is used. Every one of the 123,142 SUPPORTS and REFUTES claims has a primary page; 4,104 distinct pages, 1 to 214 claims each, median 25. Page ids in the claim files are Unicode NFD and the Wikipedia ids are NFC; 170 of 14,533 cited ids change under NFC, and all ids are normalized to NFC before joining.

**NOT ENOUGH INFO group (Inferred).** The longest primary-page title that appears in the claim text, case-insensitive, on word boundaries. Validated on the SUPPORTS and REFUTES claims, whose true page is known: match rate 0.9781 (120,449 of 123,142), accuracy among matches 0.9560. Applied to the 42,305 NOT ENOUGH INFO claims: 41,407 matched; 898 unmatched and excluded. The page the annotators actually used for these claims is Unavailable.

**Claims kept.** 164,549 (165,447 loaded minus 898 excluded). All 14,533 cited Wikipedia pages were found in the archive after NFC normalization; 0 missing.

**Evidence (confirmed, k = 5).** For every claim, the five sentences of its primary page with the highest TF-IDF cosine to the claim, best first, joined by a space. TF-IDF for this step: word 1-2-grams, `min_df=2`, sublinear term frequency, fitted on the training set's claims and page sentences only. Gold recall at 5 on the 123,142 claims with gold sentences on the primary page: 0.8854 mean; 0.8364 of claims have all their gold sentences in the five. The same rule is applied to all three labels. About 16 percent of SUPPORTS and REFUTES claims cite a second page; it is not read.

**Splits (confirmed: shares 0.70 / 0.15 / 0.15 of pages, seed 20261013).** Pages sorted, shuffled with the seed, cut at the shares; each claim follows its page.

| Split | Pages | Claims | SUPPORTS | REFUTES | NOT ENOUGH INFO |
| --- | --- | --- | --- | --- | --- |
| training | 2,873 | 114,883 | 60,687 | 25,305 | 28,891 |
| validation | 616 | 25,018 | 13,133 | 5,601 | 6,284 |
| test | 615 | 24,648 | 12,881 | 5,535 | 6,232 |

Pages in two splits: 0. Claims in two splits: 0. For contrast, a random claim split with the same shares and seed tears 4,088 of 4,104 pages (99.6 percent) across splits.

**Features (to be built; same for both systems).** One TF-IDF vocabulary fitted on the training set's claim and evidence texts together, with the `ngram_range`, `min_df` and resulting vocabulary size V recorded here once chosen. Each claim gives two blocks, claim and evidence, placed side by side, plus two pair features: the overlap count and the cosine between the blocks, standardized with the training set's mean and standard deviation. Input shape `(N, 2V + 2)`. The claim-only baseline receives the claim block alone.

#### Annotation Process
FEVER's annotators wrote each claim from a sentence of a Wikipedia page, often as a mutation of another claim from the same sentence, then a second group labeled each claim against Wikipedia and recorded the evidence sentences. Several annotators may give evidence sets for one claim. The claims therefore carry two known properties this design must handle: claims on one page are near copies of each other, which is why the page is the unit; and the claim text alone predicts the label well above chance, the annotation artifact that the claim-only baseline measures.

#### Exposure
The test set was cut by page and counted only. No label of the test set has been read by the assembly script, the walkthrough notebook, or author QA. The labels of `paper_test` are public, but this experiment treats the test set as unexposed from the moment it was cut. All piloting, the sizing calculation, weight-decay selection, hidden-size selection and early stopping use the validation set. The test set is read once, after `claim.json` is frozen.

### Systems
Multinomial Logistic Regression -- PyTorch. One weight matrix `(F, 3)` and bias, softmax over three logits, cross-entropy on the logits. F03 Part C.

One-Hidden-Layer MLP -- PyTorch. The same with one hidden layer and ReLU between; hidden size chosen on the validation set. F03 Part D.

Both on the TF-IDF processed claim and evidence from our dataset, with the two pair features.

A third model, `sklearn.linear_model.LogisticRegression` on the same features, is not a compared system; it is the check that the torch logistic regression is implemented correctly.

### Baselines
Majority Class -- the most common training-set label (SUPPORTS, 52.8 percent of training claims) predicted for every claim. The floor.

Claim-Only logistic regression -- the torch logistic regression on the claim block alone, no evidence. Measures the annotation artifact. The distance from it to each system is what reading the evidence added. Tutor pilot: 0.71 macro-F1 on SUPPORTS/REFUTES with no evidence.

### Independent Unit
The independent unit is the page, not the individual claim. Claims on one page were written by one annotator from the same sentences, as mutations of each other, and draw their evidence from the same sentences, so their errors are not separate events. The page governs the split, so no claim's near copy sits in training, and the confidence interval, so the count of evidence is not overstated. Test set: 615 pages carrying 24,648 claims. The intra-page correlation of correctness is measured on the validation set and reported so the reader can see how much this mattered.

### Seeds
Split seed: 20261013 (confirmed), used to shuffle pages once. Training seeds: at least three, declared in `claim.json` before the freeze; the same seeds for both systems. Seed spread is reported as its own number, separate from case variation, because the MLP's loss is not convex and its result depends on initialization.

### Hardware and Compute Budget
One machine: Intel Core i9-9900K (16 threads), 62 GB RAM, NVIDIA GeForce GTX 1080 Ti (11 GB). Python 3.12.14, torch 2.14.0+cu126, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, pyarrow 26.0.0; versions locked in `uv.lock`.

Budget: 2 systems x the weight-decay grid x the training seeds, plus the MLP hidden-size candidates, plus one claim-only run, plus one scikit-learn run for the check, plus one shuffled-label run. The grid, the seeds and the hidden sizes are declared in `claim.json`; the count of runs and the measured wall time per run are recorded here after the pilot. Data assembly ran once: the full Wikipedia scan is the longest step.

## Method
1. Task. Given a claim and text from one Wikipedia page, output SUPPORTS, REFUTES or NOT ENOUGH INFO. Why this form: FEVER's full task also finds the page among 5.4 million; we hand it over because retrieval starts at F05, so what is tested is the verification step alone. This is a stated limit.
2. Independent unit: the page. Claims on one page were written by one annotator from the same sentences, as mutations of each other, and will draw their evidence from the same sentences. Their errors are therefore not separate events. The page is the smallest unit whose errors we can treat as independent. It governs two things: the split, so no claim's near copy sits in training; and the confidence interval, so the count of evidence is not overstated. Why not the claim: a claim-level interval would be narrower than the data justifies. The pilot measures the intra-page correlation of correctness so the reader can see how much this mattered.
3. Splits. Pages shuffled with a declared seed and cut 70/15/15 into training, validation and test. Why pooled and re-split rather than FEVER's own dev and test: FEVER's splits share a third of their pages with its training set (451 of 1,460 dev pages, 499 of 1,499 test pages). Why shares of pages rather than claims: the page is the unit. Why 70/15/15: the training set fits about 420,000 parameters for the linear model and far more for the MLP, so it needs volume; validation only has to separate settings; the test set only has to give the confidence interval the planned width, which the pilot confirms.
4. Evidence construction: nearest-neighbor sentence selection. For each claim, rank the sentences of its page by cosine similarity between TF-IDF vectors and keep the five highest. Why a rule rather than the gold sentences: NOT ENOUGH INFO claims have no gold sentence, and giving gold to two labels and a rule to the third would let the model tell labels apart by input shape. Why five: the FEVER baseline's number; gold recall at five is 0.8854, so the cited sentence is present for about 89 percent of SUPPORTS and REFUTES claims and absent for the rest, which is a ceiling set by the data. Why only the primary page: about 16 percent of claims cite a second page; it is not read, and that is a stated limit. What this step is: nearest-neighbor search, not a k-nearest-neighbors classifier, because the sentences carry no labels.
5. Features: TF-IDF in two blocks. Each text becomes a vector with one position per vocabulary term, the term's count weighted by how rare the term is across training texts, normalized to unit length. Fitted on the training set only, so no test word enters the vocabulary or the rarity table. Why two blocks: the claim and the evidence are vectorized separately and placed side by side, so "not" in the claim and "not" in the evidence are different positions; one merged bag would lose which side a word came from. Why pair features: a linear model adds weighted positions and cannot represent a relationship between the two blocks, so the overlap count and the cosine between the blocks are appended as explicit relationship scalars.
6. Systems. Multinomial logistic regression: a linear function of the input to three logits, softmax, cross-entropy. The one-hidden-layer MLP: the same with one hidden layer and ReLU between. Why these two: the MLP contains the linear model as a special case, so the comparison isolates one thing, whether a hidden layer that can learn claim–evidence interactions beats a flat boundary on features that are already very high-dimensional. Why not deeper: that is a different question, and on bag-of-words features depth has little to compose.
7. Hypothesis. The MLP outperforms by a small margin, under 0.03 macro-F1, with a page-unit confidence interval that may include zero, because the interaction it can learn is real but the features already separate most of the data linearly. A written hypothesis is what makes the result a test rather than an observation. Pilot, 2026-10-15, validation split: under the declared stopping rule the MLP scored below logistic regression by 0.026 mean over three seeds, so the pilot predicts the hypothesis is not supported; the hypothesis stands as written and the test set decides.
8. Training. Cross-entropy on logits, a torch.optim optimizer named with its learning rate, mini-batches, weight decay chosen on the validation set from a declared grid, early stopping on validation log loss with a declared patience, at least three seeds. Why weight decay from the validation set: the MLP has the capacity to fit noise, and the penalty strength is the one knob that controls it, so it must be chosen on data the model did not train on and not on the test set. Why seeds: the MLP's loss is not convex, so its result depends on initialization, and seed spread is reported as its own number.
9. Baselines. Majority class: the floor. Claim-only logistic regression: how far the claim text alone gets you, which measures the annotation artifact; the distance from it to each system is what reading the evidence added. Both read against both systems. Why claim-only is logistic regression only: one run sets the level.
10. Metrics. Macro-F1 primary, because the labels are imbalanced and accuracy rewards the majority. Per-class precision, recall, F1 and support, and confusion matrices, to see which label each system misses. Log loss and calibration bins, to see whether a better argmax comes with better probabilities. Generalization gap, training loss minus validation loss at the stopping epoch, to see whether capacity went to noise. Accuracy reported, not primary.
11. Paired comparison and the confidence interval. The primary number is MLP macro-F1 minus logistic regression macro-F1 on the same test claims. Paired means both systems are scored on the same resampled pages each time, so the interval reflects the difference, not the two scores' separate noise. The interval: resample pages with replacement 2,000 times, recompute the difference each time, take the 2.5th and 97.5th percentiles. Decision rule: supported if the interval excludes zero in the MLP's favor, otherwise not established.
12. Checks before the freeze. No page in two splits. No vocabulary term that occurs only outside the training set. The torch logistic regression within 0.02 macro-F1 of scikit-learn's on the same features, to prove the model is implemented correctly. Shuffled training labels scoring at chance, to prove the pipeline cannot score without real labels. A seeded rerun reproducing the metrics. Why before the freeze: a check that fails after the test set is read cannot be fixed without reading it again.
13. Order of operations. Pilot everything on the validation set, write the sizing calculation, freeze claim.json, then read the test set once. Why: the frozen commit preceding the result commit is the proof that the claim was not fitted to the outcome.

## Coverage

| Module | Where X1 uses it |
| --- | --- |
| F01 | Feature tensors `(N, 2V + 2)`; the weight matrices `(F, 3)` and `(F, H)`, `(H, 3)`; matrix multiplication for the logits; softmax over 3 logits |
| F02 | The training loop: forward, cross-entropy, backward, optimizer step, per-epoch record of training loss, validation loss and validation macro-F1; mini-batches; early stopping |
| F03 | Softmax cross-entropy on logits (F03-2); the linear classifier (Part C); the one-hidden-layer MLP with ReLU and its backward pass (F03-3, Part D) |
| F04 | Page as the independent unit for the split and the confidence interval, with the reason stated; the split check on real data (F04A); vectorizer and scaler fitted on the training set only; weight decay chosen on the validation set from a grid (Part C); macro-F1, log loss and calibration; paired percentile bootstrap by page; the sizing calculation; frozen `claim.json`; one test-set read |

## Rule Set
### Claim
"The one-hidden-layer MLP outperforms multinomial logistic regression on FEVER claim verification with the page given, on the same TF-IDF features, measured by macro-F1 on held-out Wikipedia pages." Frozen in `claim.json` before the test set is read; the freeze commit is linked here once made.

### Primary Analysis
Macro-F1 Difference -- MLP macro-F1 minus logistic regression macro-F1 (`sklearn.metrics.f1_score(average="macro")`), same seed, same test claims. Reported per seed and as the mean over seeds, with the seed spread beside it.

95% confidence interval resampling pages with replacement on the test set -- percentile bootstrap, 2,000 resamples, one draw takes a page with all its claims, both systems scored on the same draw. Written with numpy in this folder. Reported with the unit (page), the number of units (615) and the number of resamples.

Decision rule -- supported when the 95 percent page-unit interval of the difference excludes 0 in the MLP's favor; otherwise not established.

Sizing calculation -- from a pilot on the validation set: the bootstrap standard error of the difference by page as a function of pages held out, and the interval half-width expected at 615 test pages. Recorded here before the freeze.

Pilot, 2026-10-15, validation split, 616 pages, chosen configurations, 2,000 resamples per point: standard error of the difference 0.0069 at 100 pages, 0.0052 at 200, 0.0041 at 300, 0.0036 at 400, 0.0030 at 500, 0.0028 at 616; least-squares fit se = 0.0705 / sqrt(pages); expected standard error at 615 test pages 0.0028, expected 95 percent half-width 0.0056. The hypothesis is a difference under 0.03, so 615 pages resolve it with room. Intra-page correlation of correctness 0.038 (logreg) and 0.041 (mlp), mean page size 40.6 claims, design effect 2.5: a claim-unit interval would be about 1.6 times too narrow. Pilot difference, MLP minus logreg, by seed: -0.005, -0.051, -0.023; mean -0.026, seed spread 0.023. Figures: `figures/sizing.png`, `figures/loss_over_epochs.png`; numbers in `results/validation_report.json`.

Weight decay (L2 regularization) selection -- chosen on the validation set from a declared grid, for each system separately, by mean validation macro-F1 over the three seeds; the grid and the chosen value for each system recorded here and in `claim.json`.

Pilot, 2026-10-15: grid 0 / 0.01 / 0.1 / 1.0 (AdamW, decoupled), hidden size 256, seeds 0, 1, 2. Mean validation macro-F1 over seeds, logreg: 0.539 / 0.538 / 0.552 / 0.391; mlp: 0.522 / 0.522 / 0.526 / 0.475. Chosen: 0.1 for both systems, interior to the grid. Seed standard deviation at the chosen value: 0.001 (logreg), 0.024 (mlp).

### Secondary Analysis
Confusion Matrices -- one per system and baseline on the test set.

Per-Class Precision, Recall, F1, Support, and overall Accuracy -- `classification_report`; accuracy reported, not primary.

Log Loss and calibration bins -- `sklearn.metrics.log_loss` and reliability bins with counts, per system.

Generalization gap -- training loss minus validation loss at the stopping epoch, per system and seed.

Distance to each baseline -- each system's macro-F1 minus the majority-class and claim-only macro-F1.

Intra-page correlation of correctness -- on the validation set, how much the page unit mattered; the claim-unit interval may appear beside the page-unit one as one line, not as a question.

Per-label and per-page-size tables -- macro-F1 by label and by how many claims the page carries.

## Checks
A check passes only if every row holds. Each runs before the test set is read and is declared in `claim.json`.

### Data assembly (run 2026-10-15, from `data/fever/data_card.json`)

| Check | Condition | Tolerance | Observed | Holds |
| --- | --- | --- | --- | --- |
| SHA-256 of `wiki-pages.zip` | equals the value recorded in the data card | exact | `4b06d95d…65f2` | pass |
| Claims loaded | 145,449 + 9,999 + 9,999 = 165,447 | exact | 165,447 | pass |
| Cited pages resolved in the Wikipedia text after NFC | 100% of pages cited by SUPPORTS and REFUTES claims | exact | 14,533 of 14,533; 0 missing | pass |
| Primary page assigned | every SUPPORTS and REFUTES claim | exact | 123,142 of 123,142 | pass |
| NOT ENOUGH INFO match rule, validated on labeled claims | match rate and accuracy reported | reported | match 0.9781, accuracy 0.9560 | reported |
| Pages in two splits | none | exact | 0 | pass |
| Claims in two splits | none | exact | 0 | pass |

### Features and training (run on the validation set before the freeze)

| Check | Condition | Tolerance | Observed | Holds |
| --- | --- | --- | --- | --- |
| Vectorizer vocabulary fitted on the training set only | no term that occurs only in the validation or test set is in it | exact | V = 293,437; terms absent from every training text: 0 | pass |
| Torch logistic regression against `sklearn.linear_model.LogisticRegression`, same features, validation macro-F1 | within 0.02 | 0.02 | torch 0.5530 (decay 0.1, best epoch 4), sklearn 0.5715 (C = 1, L-BFGS); gap 0.0185; 114,883 training rows | pass |
| Shuffled training labels, negative control | validation macro-F1 at chance: within 0.05 of 1/3 (uniform) or of the majority-class macro-F1 (prior only); see Deviations | 0.05 | macro-F1 0.2346; majority-class macro-F1 0.2295 (distance 0.005); 1/3 (distance 0.099); log loss 1.036 against a label-prior entropy of 1.018 | pass |
| Seeded rerun | the same seed reproduces the saved validation metrics | 1e-6 | max abs difference 0.0 over 7 epochs of train loss, val loss and val macro-F1; same epoch count | pass |

Run by `src/checks.py --n-train 0 --weight-decay 0.1` on 2026-10-15; values in `results/checks.json`. Baselines from the same run, validation split: majority class (SUPPORTS) macro-F1 0.2295, accuracy 0.5249; claim-only logistic regression macro-F1 0.5431, log loss 0.8565, macro-F1 on SUPPORTS and REFUTES claims alone 0.7147 (the tutor pilot's 0.71). Against the full-feature logistic regression's 0.5530, the evidence block adds 0.01 macro-F1.

## Result
**Not established.** The one-hidden-layer MLP did not outperform multinomial logistic regression. It scored lower on every seed, and the 95 percent page-unit interval of the mean difference excludes zero in logistic regression's favor.

Test split read once on 2026-10-15 (`run.py test`, execution `test_20261015_145814`), under `claim.json` as committed in `exp(X1): freeze claim`. 24,648 claims on 615 pages. Every number below is read from `results/test/`.

### Primary result
MLP macro-F1 minus logistic regression macro-F1, mean over seeds 0, 1, 2, each pair scored on the same test claims:

| Quantity | Value |
| --- | --- |
| Mean difference over seeds | **-0.0275** |
| 95% page-unit percentile bootstrap interval, 2,000 resamples of 615 pages | **[-0.0331, -0.0218]** |
| Bootstrap standard error | 0.0029 |
| Seed spread (standard deviation of the three per-seed differences) | 0.0194 |
| Verdict under the decision rule (supported if the lower bound is above 0) | not established |

The hypothesis predicted a small MLP advantage, under 0.03. The observed difference is about that size in the other direction. The pilot on the validation split predicted this (mean -0.026 over the same seeds); the test split confirms it.

| Seed | MLP macro-F1 | LogReg macro-F1 | Difference | 95% page-unit CI | 95% claim-unit CI (secondary) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.5488 | 0.5571 | -0.0084 | [-0.0146, -0.0025] | [-0.0126, -0.0041] |
| 1 | 0.5039 | 0.5510 | -0.0471 | [-0.0547, -0.0391] | [-0.0521, -0.0419] |
| 2 | 0.5312 | 0.5582 | -0.0269 | [-0.0348, -0.0182] | [-0.0322, -0.0219] |

Seed variation and case variation, separately: the three logistic regressions sit within 0.007 of each other; the three MLPs span 0.045. The seed spread of the difference (0.019) is seven times the bootstrap standard error (0.003), so which MLP you get matters more than which pages you test it on. The claim-unit intervals are narrower than the page-unit ones by about the design effect measured in the pilot (2.5 in variance, 1.6 in width), as expected; they are not the primary interval.

### The four systems on the test split
Systems at seed 0; every seed is in `results/test/test_summary.csv`. Per-class columns are precision / recall / F1. Support: SUPPORTS 12,881; REFUTES 5,535; NOT ENOUGH INFO 6,232.

| System | Macro-F1 | Accuracy | Log loss | SUPPORTS | REFUTES | NOT ENOUGH INFO |
| --- | --- | --- | --- | --- | --- | --- |
| MLP, hidden 256, decay 0.1 | 0.5488 | 0.6230 | 0.8667 | 0.636 / 0.857 / 0.730 | 0.706 / 0.438 / 0.541 | 0.491 / 0.304 / 0.376 |
| Logistic regression, decay 0.1 | 0.5571 | 0.6329 | 0.8677 | 0.641 / 0.871 / 0.738 | 0.704 / 0.460 / 0.557 | 0.520 / 0.295 / 0.376 |
| Majority class (SUPPORTS) | 0.2288 | 0.5226 | 6.60 | 0.523 / 1.000 / 0.686 | 0 / 0 / 0 | 0 / 0 / 0 |
| Claim-only logistic regression | 0.5444 | 0.6120 | 0.8562 | 0.633 / 0.831 / 0.718 | 0.730 / 0.461 / 0.565 | 0.433 / 0.294 / 0.350 |

Distance to the baselines: logistic regression is 0.328 above the majority class and **0.013 above the claim-only baseline**. The MLP is 0.320 above majority and 0.004 above claim-only. Reading the evidence adds about one point of macro-F1 over reading the claim alone, for either system.

Confusion matrices (rows true, columns predicted, order SUPPORTS / REFUTES / NOT ENOUGH INFO):

| | MLP | Logistic regression | Claim-only |
| --- | --- | --- | --- |
| SUPPORTS | 11,036 / 513 / 1,332 | 11,215 / 527 / 1,139 | 10,703 / 481 / 1,697 |
| REFUTES | 2,481 / 2,425 / 629 | 2,433 / 2,547 / 555 | 2,279 / 2,550 / 706 |
| NOT ENOUGH INFO | 3,839 / 498 / 1,895 | 3,851 / 543 / 1,838 | 3,939 / 461 / 1,832 |

Both systems fail the same way: they call most REFUTES and most NOT ENOUGH INFO claims SUPPORTS. Recall on NOT ENOUGH INFO is 0.30 for both. Neither label is where the MLP differs; it is slightly worse on all three.

Calibration (argmax probability, ten bins; count, mean confidence, accuracy; empty bins omitted):

| Bin | MLP | Logistic regression |
| --- | --- | --- |
| 0.3 to 0.4 | 537, 0.38, 0.37 | 708, 0.38, 0.40 |
| 0.4 to 0.5 | 2,998, 0.46, 0.44 | 3,674, 0.46, 0.46 |
| 0.5 to 0.6 | 4,259, 0.55, 0.52 | 4,202, 0.55, 0.53 |
| 0.6 to 0.7 | 4,553, 0.65, 0.58 | 4,138, 0.65, 0.62 |
| 0.7 to 0.8 | 4,990, 0.75, 0.65 | 4,316, 0.75, 0.67 |
| 0.8 to 0.9 | 4,805, 0.85, 0.76 | 4,436, 0.85, 0.75 |
| 0.9 to 1.0 | 2,506, 0.94, 0.86 | 3,174, 0.94, 0.84 |

Both are overconfident above 0.6 by 5 to 10 points; the MLP a little more so in the middle bins. Log loss is the same to three decimals (0.867 vs 0.868), so the MLP's probabilities are not better; its argmax is slightly worse.

Generalization gap (training loss minus validation loss at the stopping epoch): logistic regression -0.21, -0.19, -0.22; MLP -0.11, +0.06, +0.06. The MLP runs that stopped at epoch 1 (seeds 1 and 2) had not yet fitted the training set when validation loss turned; the one that reached epoch 2 (seed 0) is the best MLP and the closest to logistic regression. Figure: `figures/test_loss_over_epochs.png`.

### What the training curves say about the stopping rule
Early stopping on validation log loss, patience 3, was declared before the pilot and kept at the freeze. For the MLP it stops very early: best epoch 2, 1, 1 on the three test-mode runs. In the pilot, the MLP's validation macro-F1 kept rising for several epochs after its validation log loss had turned up (seed 0, decay 0: log loss 0.878 -> 0.984 over epochs 1 to 4 while macro-F1 went 0.531 -> 0.574). The MLP becomes overconfident on the claims it gets wrong faster than it gets more claims right, and log loss charges for that. So the stopped MLP is a model from the first or second epoch, and the comparison is between that model and a logistic regression that stopped at epoch 3 or 4.

This is a property of the declared rule, not a bug, and it was known before the freeze (Deviations, pilot observations). Stopping on macro-F1 instead would have selected later MLP epochs, which on validation reached about 0.57, the same region as logistic regression. The option was considered at the freeze and declined: the rule was declared, and changing it after seeing that it favored one system would make the result a choice rather than a test. A follow-up experiment could declare macro-F1 stopping up front; the pilot numbers suggest it would move the MLP to parity, not to an advantage.

### Discussion
The result is a negative, and the write-up's job is to say what it does and does not show.

What it shows: on these features, with this training procedure, a hidden layer does not help. Both systems land at 0.55 to 0.56 macro-F1, the claim-only baseline lands at 0.54, and the three seeds of the MLP are more spread than the gap between the systems. The 0.013 that the evidence block adds over the claim alone is the most informative number in the table: the models are barely reading the evidence.

Why, speculatively (this was not tested and is not a claim of this experiment): TF-IDF features cannot carry what the task needs. Verification is a relation between a statement and a passage: the claim says "born in 1985" and the evidence says "born in 1958", or the claim names a person and the evidence is about someone else with the same surname. In a bag of words, "1985" and "1958" are two unrelated columns, word order is gone, and the only relationship the input encodes is two scalars, the overlap count and the cosine, which measure how many words the sides share and not whether they agree. A hidden layer can form products of input positions, but the positions it is given do not contain the information a product would need. The MLP has 150 million parameters to combine 586,876 near-unique bigram columns, and what it learns to combine is which training claims those bigrams belonged to. That is the generalization gap in the pilot: training loss 0.5 against validation 0.9 by epoch 4. The linear model at least cannot memorize as fast.

If that is right, the experiment's design limited its outcome before any model was trained: a comparison of two classifiers on an input that neither can use is a comparison of how fast they overfit. The same experiment with inputs that encode the relation, token-level alignment features or sentence embeddings, would be a different experiment with a different question, and the FEVER literature's jump from bag-of-words baselines to attention models is consistent with that. None of this is established here; the only test-set read is the one reported above.

What the negative control and the baselines protect: the shuffled-label model scored at the prior (0.235), so the pipeline cannot score without real labels; the torch logistic regression is within 0.02 of scikit-learn's converged solution, so the training loop reaches the linear optimum; the seeded rerun reproduced every metric to zero difference; and the claim-only baseline reproduces the known FEVER annotation artifact (macro-F1 0.71 on SUPPORTS and REFUTES from the claim text alone on validation, matching the tutor pilot's 0.71). The systems were compared on the same features, the same claims, and the same pages, with the page as the unit throughout.

## Error Analysis
Read from `results/test/predictions/` (seed 0 of each system) joined to the test rows of `claims.parquet` by position. No model was scored again. Numbers in `results/test/error_analysis.json` and `error_analysis_cuts.csv`.

### Cuts
Accuracy is reported alongside macro-F1 because within a cut of one or two labels macro-F1 averages over absent classes.

| Cut | n | LogReg macro-F1 | LogReg acc | MLP macro-F1 | MLP acc | Claim-only acc |
| --- | --- | --- | --- | --- | --- | --- |
| All test claims | 24,648 | 0.557 | 0.633 | 0.549 | 0.623 | 0.612 |
| SUPPORTS/REFUTES, every gold sentence among the five retrieved | 15,286 | 0.475 | 0.745 | 0.465 | 0.726 | 0.713 |
| SUPPORTS/REFUTES, some gold retrieved | 1,946 | 0.519 | 0.798 | 0.515 | 0.793 | 0.795 |
| SUPPORTS/REFUTES, no gold sentence retrieved | 1,184 | 0.457 | 0.698 | 0.449 | 0.690 | 0.685 |
| SUPPORTS/REFUTES, gold on the primary page only | 15,219 | 0.475 | 0.741 | 0.466 | 0.727 | 0.706 |
| SUPPORTS/REFUTES, cites a second page that was not read | 3,197 | 0.500 | 0.776 | 0.487 | 0.751 | 0.787 |
| NOT ENOUGH INFO (page inferred by title match) | 6,232 | | 0.295 | | 0.304 | 0.294 |
| Pages with 20 claims or fewer | 2,100 | 0.510 | 0.551 | 0.502 | 0.541 | 0.543 |
| Pages with 21 to 60 claims | 12,443 | 0.576 | 0.648 | 0.568 | 0.636 | 0.630 |
| Pages with more than 60 claims | 10,105 | 0.542 | 0.632 | 0.532 | 0.624 | 0.604 |

Per-label accuracy (recall), logreg / MLP / claim-only: SUPPORTS 0.871 / 0.857 / 0.831; REFUTES 0.460 / 0.438 / 0.461; NOT ENOUGH INFO 0.295 / 0.304 / 0.294.

**The retrieval ceiling is not the ceiling.** The planned cut was claims whose gold sentence was not among the five retrieved. Those claims are scored 0.698 by logistic regression; claims with every gold sentence present are scored 0.745; the claim-only model, which reads no evidence at all, scores 0.713 on that same all-gold cut. Having the cited sentence in the input is worth about three points of accuracy over not reading the input. The 11 percent retrieval miss rate stated in Limits costs almost nothing because the models were not using the retrieved sentences much when they were there.

**Citing a second page did not hurt.** Claims whose gold evidence spans a second page, which was never read, are scored higher (0.776) than claims whose evidence is on the primary page alone (0.741), and the claim-only model is highest on them (0.787). These claims are not harder for these models; they are claims the claim text alone resolves. The stated limit holds as a limit of the input, not as a measured cost.

**NOT ENOUGH INFO is where both systems fail.** Recall 0.30 for both. The NEI cut cannot be split into right and wrong inferred pages because NEI claims have no gold page; what can be said is that the whole NEI label behaves the same way, whichever system, and the same as the claim-only model, so the inferred evidence is not what decides these claims.

**Small pages are harder.** Pages with 20 claims or fewer score 0.55, pages with 21 to 60 claims 0.65. Small pages are short articles with less text to retrieve from and fewer training claims about them.

### Where the two systems disagree
The systems disagree on 2,029 of 24,648 claims (8.2 percent). On those, logistic regression is right 975 times, the MLP 731, neither 323. The MLP's disagreements move claims across the SUPPORTS / NOT ENOUGH INFO boundary: logreg SUPPORTS -> MLP NEI 748 claims, logreg NEI -> MLP SUPPORTS 542. On REFUTES they barely differ. The MLP is not seeing something different in the evidence; it draws the same boundary in a slightly different place and loses a little on balance. Both are wrong on 8,317 claims (34 percent): 4,056 NEI, 2,869 REFUTES, 1,392 SUPPORTS.

### Real failure cases
Claim, the first of the five retrieved sentences, the true label, and each system's label with its confidence.

1. **Contradiction by one word, gold sentence present.** "Venus is called Earth's 'brother planet.'" Evidence: "Venus ... is sometimes called Earth's 'sister planet' because of their similar size, mass ..." True REFUTES; logreg SUPPORTS 0.79, MLP SUPPORTS 0.80. The claim and the evidence share "Venus", "Earth", "called", "planet". The one word that decides the label, "brother" against "sister", is one column in each block with no relation between them.
2. **Antonym, gold present.** "The Republican Party is socially liberal." Evidence: "... the Republican Party is socially conservative ..." True REFUTES; both SUPPORTS at 0.83 and 0.81. Same mechanism: "socially" and "Republican Party" overlap; "liberal" and "conservative" are unrelated columns.
3. **Wrong entity, gold present.** "The Flash is based on Hal Jordan." Evidence: "It is based on the DC Comics character Barry Allen / Flash ..." True REFUTES; both SUPPORTS at 0.70 and 0.69. "based on" overlaps; which name follows it does not register.
4. **NEI about the right page.** "Venus is a dwarf planet." Evidence: the same Venus sentences as case 1. True NOT ENOUGH INFO; logreg SUPPORTS 0.45, MLP SUPPORTS 0.55. The NEI claim was written about this page, the title-match rule found the page, and the retrieved sentences share its vocabulary, so the overlap looks like support. This is the dominant NEI failure: 3,539 of 6,232 NEI claims are called SUPPORTS by both systems.
5. **Supported by a sentence that was retrieved, called NEI.** "Urine contains bilirubin." Evidence: "Bilirubin is excreted in bile and urine ..." True SUPPORTS; logreg NEI 0.60, MLP NEI 0.80. A short claim with little overlap beyond "bilirubin" and "urine" reads as NEI to both, and the MLP is more confident in the wrong answer, which is the calibration pattern from the Result tables.
6. **The MLP overcommits.** "Amerigo Vespucci was from Europe." Evidence: "... was an Italian explorer ..." True SUPPORTS; logreg SUPPORTS 0.42, MLP NOT ENOUGH INFO 0.71. "Europe" does not appear in the evidence; "Italian" does. Logistic regression leans weakly toward support, the MLP commits to NEI. Among the fifty claims where logreg is right and the MLP is most confidently wrong, this shape repeats: the MLP's wrong label carries 0.65 to 0.75 confidence.
7. **No gold retrieved, still called SUPPORTS.** "Precious premiered in 1977." Evidence: "... is a 2009 American drama film ..." True REFUTES; both SUPPORTS at 0.75. The year that refutes the claim is present in the retrieved text but is just another token, and "1977" is a column the evidence block does not light.

### What they share
Every case is the same failure: the label depends on a relation between a specific claim token and a specific evidence token (brother/sister, liberal/conservative, Hal Jordan/Barry Allen, 1977/2009, Europe/Italian), and the input encodes which tokens are present on each side but not whether they agree. Overlap and cosine, the only cross-side features, rise whenever the claim is about the page at all, which is true for every FEVER claim including the NOT ENOUGH INFO ones, so they say "this claim is about this page", not "this page confirms this claim". The MLP, which could in principle combine positions, had no positions to combine that carry the relation. This is the same conclusion the Result's discussion reaches from the baselines, now seen claim by claim.

## Deviations
`claim.json` is not yet frozen. These are changes to the proposed training settings made during the pilot, with the observation that forced each one, so the design review sees them. The feature settings (`ngram_range` (1, 2), `min_df` 2), the split, the evidence rule and the systems are unchanged. Implementation changes that do not alter what is trained (sparse GPU batches, the dashboard) are not design deviations and are described under Reproduction.

### Pilot observations, 2026-10-15

All numbers are validation-split scores; the test split has not been read.

| Run | Training rows | Optimizer, lr, decay | Best epoch | Val log loss | Val macro-F1 |
| --- | --- | --- | --- | --- | --- |
| Limited pilot, logistic regression | 20,000 | Adam, 1e-3, 0 | still falling at 16 | 0.9295 at ep 16 | 0.41 at ep 16 |
| Full pilot, logistic regression, 12 runs | 114,883 | AdamW, 1e-2, 0 to 1e-3 | 1, every run | 0.880 to 0.884 | 0.538 to 0.541 |
| Subset sweep, logistic regression | 20,000 | AdamW, 3e-3 or 1e-2, 0 to 0.1 | 3 (1e-2) or 12 (3e-3) | 0.911 to 0.915 | 0.465 to 0.478 |
| Subset sweep, logistic regression | 20,000 | AdamW, either lr, 1.0 | 12 or 20 | 0.934 to 0.936 | 0.38 to 0.39 |
| Reference: `sklearn.linear_model.LogisticRegression`, C = 1, L-BFGS | 114,883 | converged, 280 iterations | | 0.857 | 0.572 |
| Reference: same, C = 0.1 | 114,883 | converged | | 0.859 | 0.506 |
| Reference: same, C = 10 | 114,883 | hit 300 iterations | | 1.143 | 0.573 |
| Majority class (SUPPORTS) | | | | | 0.230 |

### Changes

1. **Learning rate 1e-3 -> 1e-2.** At 1e-3 the limited pilot was still improving at epoch 16, with validation loss falling about 0.003 per epoch and macro-F1 0.41, rising from the majority floor of 0.23; early stopping would never have triggered before the 30-epoch cap. Unit-length TF-IDF rows put each input near 0.1, so 1e-3 under-steps. The subset sweep then showed 3e-3 and 1e-2 reach the same validation floor (0.915 vs 0.911), 3e-3 taking four times the epochs, so the rate changes speed and not the minimum. 1e-2 is kept.

2. **Optimizer Adam -> AdamW.** The spec names AdamW or SGD. Adam's `weight_decay` is a coupled L2 term added to the gradient and rescaled by the adaptive step; AdamW applies the decay to the weights directly, so one grid value means one thing. Changed before the grid was trained.

3. **Weight-decay grid 0 / 1e-5 / 1e-4 / 1e-3 -> 0 / 0.01 / 0.1 / 1.0.** In the first full pilot the three smallest decays gave validation losses identical to four decimals at every seed, because AdamW shrinks a weight by lr x decay per step and 1e-2 x 1e-4 is a millionth. The subset sweep over the new grid shows decay up to 0.1 leaves the validation floor unchanged and 1.0 raises it by 0.02, so the new grid brackets the optimum and the answer inside it is "weak or none". The pilot is rerun on this grid; the chosen value per system is declared in `claim.json` at the freeze.

4. **Shuffled-label check: chance has two references.** The row was written as "within 0.05 of 1/3". The shuffled-label model scored 0.2346 macro-F1 with log loss 1.036. That is the majority-class value (0.2295) and the entropy of the label prior (1.018): with 53 percent SUPPORTS, a model fitted to noise learns the class frequencies and predicts SUPPORTS for everything, and macro-F1 of that is 0.23, not 1/3. The model learned nothing, which is what the control tests, but the row as written fails. The condition is amended before the freeze to "within 0.05 of 1/3 (uniform guessing) or of the majority-class macro-F1 (prior only)", both recorded. The observed value is unchanged by the amendment.

### After the freeze

`claim.json` was committed on 2026-10-15 (`exp(X1): freeze claim`). Nothing in it changed afterward. Two things happened between the freeze and the completed test read:

5. **Test mode saves checkpoints.** As written at the freeze, test mode retrained the chosen configurations under the frozen settings and seeds (a reproduction of the pilot models, which the seeded-rerun check covers) and scored the test split, keeping the per-claim predictions but not the weights. The spec freezes the procedure, not the weights, and this was within it; but the artifact behind each test number should exist on disk. One line was added so each model's state dict is written to `results/test/checkpoints/` beside its predictions (committed as `exp(X1): test mode saves each model's checkpoint`, before the test read completed). The checkpoints are gitignored; the MLP ones are about 600 MB each.

6. **The first test run was stopped and rerun.** The first invocation of `run.py test` was stopped by hand after one of its seven models, logistic regression seed 0, had been trained and scored, to make the change above. Its test macro-F1 (0.5571) and log loss (0.8677) were seen in the console. No decision was made on them and no other model's test score existed. The partial `results/test/` was deleted and test mode was run again from the start under the same `claim.json`; the rerun reproduces that model exactly (same seed, same settings), so the number seen is the number reported. This is the only test-split exposure before the reported read.

### What the comparison to scikit-learn shows

The torch logistic regression and scikit-learn's `LogisticRegression` are the same model on the same features and the same 114,883 training rows. scikit-learn's L-BFGS runs to convergence of the L2-regularized loss; ours is AdamW with early stopping on validation loss.

- The linear ceiling on these features is about 0.57 macro-F1 (0.857 log loss). C = 0.1 is too strong (0.506); C = 10 keeps the F1 but the probabilities degrade (1.143 log loss). Weak regularization is right, which agrees with the sweep.
- Our torch model stops at epoch 1 with 0.541 macro-F1 and 0.880 log loss, 0.03 below the ceiling. That exceeds the 0.02 tolerance of the check row "torch logistic regression against scikit-learn". The subset sweep rules out the learning rate and the weight decay as the cause; the remaining difference is the stopping rule (a point chosen by validation loss after whole epochs at lr 1e-2, against a converged optimum). The check is run by `checks.py` after the pilot and its observed value is written in the table above whether or not it holds; if it does not hold, that is reported, not tuned away.
- Every logistic regression run overfits after its first epoch: training loss at the best epoch is 0.92 and falling while validation loss rises from 0.880. With 586,876 input columns and 114,883 rows, a linear model memorizes near-unique bigrams. `min_df` is a declared feature setting and was not changed in response.

## Limits
- Verification only, not retrieval. The page is given; the evidence is retrieved from the known page, not from all of Wikipedia. This is a fixed-corpus claim about the classifier.
- Retrieval ceiling. Gold recall at 5 is 0.8854, so about 11 percent of SUPPORTS and REFUTES claims are scored on evidence that does not contain the cited sentence. About 16 percent of claims cite a second page that is not read.
- Annotation artifact. FEVER claims were written by annotators from Wikipedia sentences and the claim text alone predicts the label well above chance. The claim-only baseline measures this; it does not remove it.
- Inferred groups. The NOT ENOUGH INFO page is inferred by a title-match rule that is right about 96 percent of the time where it can be checked; 898 unmatched claims were excluded. A wrong match can place near-copy claims in different splits.
- Data vintage and sampling. The text is the June 2017 dump; results do not transfer to current Wikipedia. The pages are a sample chosen by the FEVER authors, not a random sample of Wikipedia.
- Units and seeds. 615 test pages; at least three training seeds. The interval is only as wide as those allow.
- Exposure. `paper_test` labels are public. This experiment did not read them after the cut, but that is a procedural guarantee, not a cryptographic one.

## Reproduction
From a fresh clone, with `uv` installed:

```bash
uv sync
uv run python scripts/get_data.py            # all stages: download, claims, wiki, split, evidence, card
# or one stage at a time:
uv run python scripts/get_data.py download
uv run python scripts/get_data.py claims
uv run python scripts/get_data.py wiki
uv run python scripts/get_data.py split
uv run python scripts/get_data.py evidence
uv run python scripts/get_data.py card
```

Implementation note: `train.batches` sends each batch to the GPU as a `torch.sparse_csr_tensor` of its nonzeros, and the models multiply with `torch.sparse.mm`. The dense alternative, densifying each 256-row batch to a 1.2 GB float64 block, cost 550 ms of memory traffic per batch against a few milliseconds of arithmetic (38 s per epoch on a 20,000-row subset against 0.4 s sparse). Same weights, same logits; the seeded-rerun check covers whether the sparse backward is deterministic on this GPU.

The download is about 1.75 GB; the Wikipedia scan reads all 5,416,537 pages once. Output lands in `data/fever/`, which git ignores: `claims.parquet` (164,549 claims with split and evidence), `wiki_subset.parquet` (14,533 pages), `data_card.json`, `split_report.json`, `nei_rule_report.json`.

The feature, training, bootstrap and table steps (`run.py`, `test_x1.py`) and their runtime are added here once written. Hardware and library versions are in Setup; the lock file is `uv.lock`.
