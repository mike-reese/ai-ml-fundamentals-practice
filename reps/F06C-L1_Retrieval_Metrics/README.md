# F06C-L1 -- Retrieval Metrics from One Ranked List and Graded Judgments

## Mechanism
A score orders candidates; it does not say whether the order is good. That needs a recorded judgment per (query, document) pair and a rule that turns one ranked list plus the judgments into a number. Four rules are built by counting and summing: two about the top-k as a set (precision@k, recall@k) and two about position (reciprocal rank, nDCG@k). Every number carries three declarations: the binary threshold, the cutoff, and what an unanswerable query is credited with.

## Task
Create the toolkit functions that build the relevant set from graded judgments, compute precision@k, recall@k, reciprocal rank, DCG@k, ideal DCG@k and nDCG@k from one ranked list, and take the mean over the defined values with its count. The analyzer A1 and the index builder come from `evalkit.tf_idf_weights`; the lengths, the IDF table and the term-at-a-time scorer come from `evalkit.bm25_term_frequency`. None of them is rewritten. Inside the functions the toolbox is `math.log2`, sets, dicts, lists and `sorted`; `ranx` and `sklearn.metrics` appear in the tests only. No function holds global state: the threshold and the cutoff are arguments, not module constants.

## Interface
Build Judgments (helper)
name:
  build_judgments
arguments:
  rows: The judgment rows of relevance_judgments.json, each with query_id, doc_id, grade and reason
returns:
  judgments: A dict from query_id to a dict from doc_id to its grade (0, 1 or 2 under rule RJ-1)

Relevant Set
name:
  relevant_set
arguments:
  judgments: One query's judgments, a dict from doc_id to grade
  threshold: The binary relevance threshold; a document is relevant when grade >= threshold
returns:
  relevant: The set of relevant doc_ids; empty for an unanswerable query

Precision@k
name:
  precision_at_k
arguments:
  ranked: The ranked list of doc_ids, rank 1 first; a document outside it has no rank
  relevant: The query's relevant set
  k: The cutoff
returns:
  precision: The number of top-k entries in the relevant set, over k; the divisor stays k when the list is shorter

Recall@k
name:
  recall_at_k
arguments:
  ranked: The ranked list of doc_ids, rank 1 first
  relevant: The query's relevant set
  k: The cutoff
returns:
  recall: The number of top-k entries in the relevant set, over the size of the relevant set; None when that set is empty

Reciprocal Rank
name:
  reciprocal_rank
arguments:
  ranked: The ranked list of doc_ids, rank 1 first
  relevant: The query's relevant set
returns:
  rr: 1 / rank of the first relevant document; 0.0 when none is on the list; None when the relevant set is empty

DCG@k
name:
  dcg_at_k
arguments:
  ranked: The ranked list of doc_ids, rank 1 first
  judgments: One query's judgments, a dict from doc_id to grade; the grade of each listed document is its gain
  k: The cutoff
returns:
  dcg: The sum over ranks 1..k of grade / log2(rank + 1); an entry beyond the list's end adds 0

Ideal DCG@k
name:
  ideal_dcg_at_k
arguments:
  judgments: One query's judgments, a dict from doc_id to grade, every document of the corpus included
  k: The cutoff
returns:
  ideal_dcg: The DCG of the judged grades sorted from largest to smallest and cut at k; 0.0 when every grade is 0

nDCG@k
name:
  ndcg_at_k
arguments:
  ranked: The ranked list of doc_ids, rank 1 first
  judgments: One query's judgments, a dict from doc_id to grade
  k: The cutoff
returns:
  ndcg: dcg_at_k / ideal_dcg_at_k; None when the ideal is 0. Linear gain, logarithm base 2

Mean Over the Defined Values
name:
  mean_defined
arguments:
  values: A dict from query_id to a metric value, None where the metric is undefined for that query
returns:
  mean: The mean of the non-None values; None when there is none
  count: The number of queries the mean covers

## Pass Check
tests/test_retrieval_metrics.py passes: four tests, one per section. A test passes only if every row holds.

Inputs: fixture F05-v1, the five documents with `arrival = 0` from the F06A-L1 copy of `documents.json` (sha256 `06128ab7…`) and the eight validation queries `vq1` to `vq8` of `data/F05_v1/queries.json` (sha256 `d2574d07…`), written into the test as strings. Fixture F06-v1, `relevance_judgments.json` in this folder (sha256 `ee8b682b…`): 40 rows, one per (query, document) pair, graded under rule RJ-1 (`2` holds the answer span, `1` is about the task with no answer, `0` otherwise), reviewed on 2026-10-16 with 0 changes. The ranked lists are the BM25 lists of the F06B-L1 term-at-a-time function at `k1 = 1.2`, `b = 0.75`, Lucene IDF, `doc_id` only, in rank order; the test checks there are no tied scores. Policy 1: relevant means `grade ≥ 2`; threshold 1 is used for contrast. Policy 2: a query with no grade `≥ 1` (`vq6`) has recall, reciprocal rank and nDCG `None`, excluded from every mean; its precision is defined. Cutoffs `k = 1, 3, 5`. For `vq1`: list A is its BM25 list, B is A reversed, C is A without the answer document. Known values are full double-precision numbers computed on this machine (Python 3.12, scikit-learn 1.9.1, ranx 0.3.21) from the formulas in the spec; every one matches the spec's four-decimal value.

1. **Known values.** Grade counts 7 × 2, 8 × 1, 25 × 0; the relevant set of each query at threshold 2; `vq1`'s set at threshold 1 of size 3; `vq6` the only unanswerable query, all exact. The eight lists as `(doc_id, grade)` pairs, exact. `vq1` at threshold 2: precision@1, 3, 5 = `1.0`, `0.3333`, `0.2` with the 3-entry list dividing by 5; recall `1.0` at every `k`; the same counts and divisions written in the test, within `1e-9`. `vq1` at threshold 1: precision `1.0`, `1.0`, `0.6`, recall `0.3333`, `1.0`, `1.0`, within `1e-9`. Reciprocal rank `1.0` for the 7 answerable queries, `vq6` `None` exact; MRR `1.0` over 7, exact on the count; ranx's convention `0.875` over 8 written in the test. DCG@3 of A and B as the three quotients `2/log2(2) + 1/log2(3) + 1/log2(4)` = 3.1309 and `1/log2(2) + 1/log2(3) + 2/log2(4)` = 2.6309, written in the test; the ideal from the judged grades `2, 1, 0, 0, 1` sorted and cut to `2, 1, 1` = 3.1309; nDCG@3 A `1.0`, B `0.8403`, within `1e-9`. nDCG@3 `vq5` 0.8403, `vq8` 0.6388; `vq6` ideal `0.0` and nDCG `None`. Recall@3, reciprocal rank, nDCG@3 of A `1.0, 1.0, 1.0`, B `1.0, 0.3333, 0.8403`, C `0.0, 0.0, 0.5209`, with C's DCG `1 + 0.6309` over the same ideal. The per-query table (P@1, P@3, P@5, R@1, R@3, R@5, RR, nDCG@3) within `1e-9` and exact on `None`; means P@3 0.2917 over 8, R@3 1.0 over 7, MRR 1.0 over 7, nDCG@3 0.9256 over 7, exact on the counts. Reason for the tolerance: every value is a count over an integer or a short sum of grade over `log2` in double precision; two orders of summation differ by about `1e-16`. Sets, `None` and counts are exact.
2. **Against the libraries.** A ranx `Qrels` from the pairs with grade `> 0` and a `Run` from the BM25 `(doc_id, score)` pairs, evaluated per query on `precision@{1,3,5}-l2`, `recall@{1,3,5}-l2`, `mrr-l2`, `ndcg@3`: the eight metrics of the 7 answerable queries and precision@k of `vq6` equal ours within `1e-9`; ranx scores `vq6` as `0.0` in every metric, asserted. The ranx means over 8 (`mrr-l2` 0.875, `recall@3-l2` 0.875, `ndcg@3` 0.8099, `precision@3-l2` 0.2917) are asserted against their known values; their differences from the policy 2 means over 7 are printed, not asserted. scikit-learn on `vq1`: `y_true = [[2, 1, 0, 0, 1]]`, lists encoded as descending integers over the five documents (A `[5, 4, 2, 1, 3]`, B `[3, 4, 2, 1, 5]`, C `[3, 5, 2, 1, 4]`); `dcg_score` and `ndcg_score` at `k = 3` equal ours for A and B within `1e-9`; C's `0.8403` against our `0.5209` printed, not asserted; tied scores `[1, 1, 0, 0, 1]` give 0.9075 and an all-zero `y_true` gives `0.0`. Reason: the same counts, divisions and base-2 logarithms agree to about `1e-16`; the two reported rows are conventions. ranx keeps an unanswerable query in the mean as 0; scikit-learn ranks all five documents, so a third document always fills rank 3 of list C.
3. **The hazard: the ideal from the list's own grades.** A wrong ideal DCG@3 built from the grades of the documents on the list, written in the test: for A and B it equals the right one, 3.1309, and both nDCG@3 values are unchanged (`1.0`, `0.8403`); for C it is `1 + 0.6309` = 1.6309 and makes C perfect, `1.0`, against the right `0.5209` over 3.1309; `vq8`'s one-entry list becomes `1.0` against the right `0.6388`. All within `1e-9`. Reason: the ideal is the best order the judgments allow, so a document the system never retrieved still counts in it. The wrong version passes on every complete list, which is why a test on A and B alone cannot catch it.
4. **Unseen cases.** `vq1` with an empty list: precision@3, recall@3, reciprocal rank, nDCG@3 all `0.0` and none `None`, exact. `vq5` at threshold 1: relevant set of 3 with `KB-0131.v1` never a candidate, recall@5 `2/3`; at threshold 2 `1.0`. Mean recall@5 at threshold 1 `0.8571` over 7, with `vq8` at `1/3`, exact on the count. `vq1` list A at `k = 10`: precision `0.1`, recall `1.0`, nDCG `1.0`, with DCG@10 and the ideal@10 equal to their `k = 3` values. Reason: an empty list for an answerable query is a system failure and scores 0; an unanswerable query is a property of the data and scores `None`; the two must not share a value. A cutoff past the list's end adds non-relevant positions to precision's divisor and nothing to recall or DCG.

## Results
All tests passed.

Numbers from `results/F06C-L1.json`, four decimals shown.

`N = 5`, 40 judged pairs: 7 × grade 2, 8 × grade 1, 25 × grade 0. Every pair is judged. The relevant set at threshold 2 is one document per answerable query: `vq1` `KB-0101.v1`, `vq2` `KB-0107.v1`, `vq3` `KB-0112.v1`, `vq4` `KB-0120.v1`, `vq5` `KB-0107.v1`, `vq7` `KB-0112.v1`, `vq8` `KB-0101.v1`; `vq6` (`staging tuesday`) has no grade above 0 and is the one unanswerable query. At threshold 1, `vq1`'s set grows to `{KB-0101.v1, KB-0107.v1, KB-0131.v1}`.

The eight BM25 lists (`k1 = 1.2`, `b = 0.75`, Lucene IDF, no tied scores) as `doc_id:grade`:

| query | text | list |
|---|---|---|
| vq1 | `retry failed job` | `0101:2, 0107:1, 0131:1` |
| vq2 | `streaming importer retry` | `0107:2, 0101:1, 0131:1` |
| vq3 | `deploy reviewer` | `0112:2, 0131:0, 0120:0` |
| vq4 | `cache ttl` | `0120:2` |
| vq5 | `dead letter queue` | `0107:2, 0101:1` |
| vq6 | `staging tuesday` | `0112:0` |
| vq7 | `production deploy tuesday` | `0112:2, 0120:0, 0131:0` |
| vq8 | `wait between attempts` | `0101:2` |

Every answerable query has its answer document at rank 1. The lists hold 1 to 3 candidates, never 5: a document that shares no term with the query has no rank.

`vq1` precision@k and recall@k, the 3-entry list A:

| threshold | relevant set | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 |
|---|---|---|---|---|---|---|---|
| 2 | size 1 | 1 / 1 = 1.0 | 1 / 3 = 0.3333 | 1 / 5 = 0.2 | 1 / 1 = 1.0 | 1.0 | 1.0 |
| 1 | size 3 | 1 / 1 = 1.0 | 3 / 3 = 1.0 | 3 / 5 = 0.6 | 1 / 3 = 0.3333 | 1.0 | 1.0 |

Precision@5 divides by 5 although the list has 3 entries. The threshold changed precision@3 from 0.3333 to 1.0 on the same list.

Reciprocal rank is 1.0 for the 7 answerable queries and undefined for `vq6`. MRR = `7 / 7` = 1.0 over 7. With `vq6` counted as 0, ranx's convention, `7 / 8` = 0.875.

The DCG@3 trace of `vq1`'s lists A and B, gain over `log2(rank + 1)`:

| rank | list A | quotient | list B | quotient |
|---|---|---|---|---|
| 1 | `KB-0101.v1`, grade 2 | 2 / 1.0000 = 2.0000 | `KB-0131.v1`, grade 1 | 1 / 1.0000 = 1.0000 |
| 2 | `KB-0107.v1`, grade 1 | 1 / 1.5850 = 0.6309 | `KB-0107.v1`, grade 1 | 1 / 1.5850 = 0.6309 |
| 3 | `KB-0131.v1`, grade 1 | 1 / 2.0000 = 0.5000 | `KB-0101.v1`, grade 2 | 2 / 2.0000 = 1.0000 |
| DCG@3 | | 3.1309 | | 2.6309 |

The ideal order comes from the judged grades `2, 1, 0, 0, 1` sorted to `2, 1, 1, 0, 0` and cut at 3: `2.0000 + 0.6309 + 0.5000` = 3.1309. nDCG@3: A `3.1309 / 3.1309` = 1.0, B `2.6309 / 3.1309` = 0.8403. With the gain `2^grade − 1`, B would be 0.7579.

The three lists of `vq1` at `k = 3`:

| list | documents | recall@3 | RR | DCG@3 | nDCG@3 |
|---|---|---|---|---|---|
| A | `0101, 0107, 0131` | 1.0 | 1.0 | 3.1309 | 1.0 |
| B | `0131, 0107, 0101` | 1.0 | 0.3333 | 2.6309 | 0.8403 |
| C | `0107, 0131` | 0.0 | 0.0 | 1.6309 | 0.5209 |

A against B: the same set in the top-3, so recall@3 is equal; the answer moved from rank 1 to rank 3, so RR and nDCG@3 fell. A against C: the answer is not on the list, so recall@3 and RR fell to 0; nDCG@3 fell because the ideal still counts the answer.

The per-query table at threshold 2:

| query | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | RR | nDCG@3 |
|---|---|---|---|---|---|---|---|---|
| vq1 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| vq2 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| vq3 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| vq4 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| vq5 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8403 |
| vq6 | 0.0 | 0.0 | 0.0 | undefined | undefined | undefined | undefined | undefined |
| vq7 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| vq8 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.6388 |

Means over the defined values: P@3 0.2917 over 8, R@3 1.0 over 7, MRR 1.0 over 7, nDCG@3 0.9256 over 7. `vq5` and `vq8` lose nDCG without losing anything else: their grade-1 documents sit at rank 2 or are missing, while the ideal from the judgments has `2, 1, 1` for both. `vq8`'s one-entry list scores `2.0 / 3.1309` = 0.6388.

Against ranx, on the same lists and judgments: every defined per-query value agrees with a difference of 0.0, and `vq6` scores 0.0 in all eight metrics. The means differ by convention:

| metric | policy 2, over 7 | ranx, over 8 | difference |
|---|---|---|---|
| MRR | 1.0 | 0.875 | 0.125 |
| recall@3 | 1.0 | 0.875 | 0.125 |
| nDCG@3 | 0.9256 | 0.8099 | 0.1157 |
| precision@3 | 0.2917 over 8 | 0.2917 | 0.0 |

Precision is defined for `vq6` under both, so that mean is equal.

Against scikit-learn on `vq1`, `y_true = [[2, 1, 0, 0, 1]]`, each list encoded as descending integers over the five documents:

| list | encoding | `dcg_score` | ours | `ndcg_score` | ours |
|---|---|---|---|---|---|
| A | `[5, 4, 2, 1, 3]` | 3.1309 | 3.1309 (diff 4.4e-16) | 1.0 | 1.0 (diff 0.0) |
| B | `[3, 4, 2, 1, 5]` | 2.6309 | 2.6309 (diff 4.4e-16) | 0.8403 | 0.8403 (diff 0.0) |
| C | `[3, 5, 2, 1, 4]` | 2.6309 | 1.6309 | 0.8403 | 0.5209 |

List C's row is a convention, not an error: scikit-learn ranks all five documents, so `KB-0101.v1` at score 4 fills rank 3 and C becomes list B. Tied scores `[1, 1, 0, 0, 1]` give 0.9075, the gain averaged over the tie. An all-zero `y_true` gives 0.0 where ours is undefined.

The hazard, an ideal built from the list's own grades: A and B keep 1.0 and 0.8403, because their wrong ideal equals the right one. C's wrong ideal is `1 + 0.6309` = 1.6309 and its nDCG@3 becomes `1.6309 / 1.6309` = 1.0 against the right 0.5209. `vq8`'s wrong ideal is 2.0 and its nDCG@3 becomes 1.0 against the right 0.6388.

Unseen cases. `vq1` with an empty list: precision@3, recall@3, RR and nDCG@3 are all 0.0 and none is undefined. `vq5` at threshold 1: relevant set `{KB-0101.v1, KB-0107.v1, KB-0131.v1}`, `KB-0131.v1` never a candidate, recall@5 `2 / 3` = 0.6667 against 1.0 at threshold 2. Recall@5 at threshold 1 per query: 1.0 for `vq1` to `vq4` and `vq7`, `vq5` 0.6667, `vq8` 0.3333, `vq6` undefined; mean 0.8571 over 7. `vq1` list A at `k = 10`: precision 0.1, recall 1.0, nDCG 1.0.

## Why It Matters
BM25 produced a ranked list per query, and F06B-L1 checked that list against Tantivy's. Neither step said whether the list was any good. That takes a judgment per (query, document) pair, recorded under a written rule, and a metric that reads the list and the judgments together. Rule RJ-1 gives three grades and policy 1 picks the threshold that turns them into a relevant set. The threshold is a declaration, not a fact: `vq1` has one relevant document at threshold 2 and three at threshold 1, and precision@3 moves from 0.3333 to 1.0 on the same list. A number from this rep is not comparable to another until both name their threshold.

The four metrics disagree on purpose. Precision@k and recall@k read the top-k as a set, so A and its reverse B score the same on both. Reciprocal rank and nDCG@k read position, so B drops to 0.3333 and 0.8403. List C, which never retrieves the answer, is the one that recall catches: 0.0 where A and B had 1.0. When a colleague reports that nDCG@3 fell, recall@3 is the number to ask for first. Recall unchanged with nDCG down points at the ordering stage; recall down points at the candidate stage, because a document the retriever never surfaced cannot be reordered into place.

The hazard is the divisor of nDCG. The ideal order comes from the judgments, which include every document of the corpus, not from the list, which holds only what the system found. On a complete list the two agree, and A and B give the same ideal either way. On list C and on `vq8`'s one-entry list the wrong divisor awards a perfect 1.0 for leaving the answer out. scikit-learn's `ndcg_score` shows the other side of the same point: it ranks all five documents, so a document the retriever never surfaced still fills a rank, and list C scores 0.8403 there against the 0.5209 of a list that truly has no third document.

The unseen cases fix the policy for the edges. An empty list for an answerable query scores 0 everywhere; an unanswerable query scores `None` on recall, reciprocal rank and nDCG and keeps its precision. ranx collapses the two into 0 and keeps the query in the mean, so MRR reads 0.875 over 8 instead of 1.0 over 7 on the same lists. Both are defensible; they credit the system differently for `vq6`, and a mean reported without its count hides which convention it used. At threshold 1, `KB-0131.v1` is relevant to `vq5` by task but shares no term with it, so no lexical system can reach recall 1 there: the threshold also sets the ceiling.

## Toolkit
We added `relevant_set`, `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `dcg_at_k`, `ideal_dcg_at_k`, `ndcg_at_k` and `mean_defined`, with the helper `build_judgments`. The ranked lists come from `evalkit.bm25_term_frequency.score_term_at_a_time` over the `evalkit.tf_idf_weights` index. `build_judgments` folds the fixture rows into `query_id → {doc_id: grade}`. `relevant_set` applies the threshold to one query's judgments. `precision_at_k` and `recall_at_k` count the relevant documents among the top-k and divide by `k` and by the size of the relevant set; recall returns `None` for an empty set. `reciprocal_rank` returns `1 / rank` of the first relevant document, `0.0` when none is on the list, `None` for an empty set. `dcg_at_k`, `ideal_dcg_at_k` and `ndcg_at_k` are three functions so that each value exists for the trace; the ideal reads the judgments, not the list, and nDCG returns `None` when the ideal is 0. `mean_defined` returns the mean of the non-`None` values with the number of queries it covers. Every threshold and cutoff is an argument; no function holds global state.
