# F06B-L1 -- BM25 by Term-at-a-Time Accumulation

## Mechanism
Compute BM25 the way a search engine computes it: a Lucene IDF per term, a saturation-normalized term that replaces the raw count, and a scoring loop that walks the query's posting lists and adds each posting's contribution into a running total per document.

## Task
Create four toolkit functions that compute the document lengths, the Lucene IDF, the term contribution, and the term-at-a-time ranked list over the F06A-L1 index. The analyzer A1 and the index builder are imported from `evalkit.tf_idf_weights`, not rewritten.

## Interface
Document Lengths
name:
  document_lengths
arguments:
  forward_index: A dict from doc_id to a dict from term to tf, the F06A-L1 forward index
returns:
  dl: A dict from doc_id to its length in terms, the sum of its tf values, in index order
  avgdl: The mean of dl over the corpus

Lucene IDF
name:
  lucene_idf
arguments:
  df: The number of documents that contain the term
  N: The number of documents in the corpus
returns:
  idf: ln(1 + (N - df + 0.5) / (df + 0.5)), natural logarithm, always positive

Build IDF Table (helper)
name:
  build_idf_table
arguments:
  inverted_index: A dict from term to its postings, the F06A-L1 inverted index
  N: The number of documents in the corpus
  idf: The IDF function of df and N, lucene_idf unless another variant is passed
returns:
  idf_table: A dict from every vocabulary term to its idf, df taken as the posting-list length

Term Contribution
name:
  term_contribution
arguments:
  tf: The term's count in the document
  dl: The document's length in terms
  avgdl: The average document length
  idf: The term's idf
  k1: The saturation setting
  b: The length-normalization setting
returns:
  denominator: tf + k1 * (1 - b + b * dl / avgdl)
  saturation_normalized_term: tf * (k1 + 1) / denominator
  contribution: idf * saturation_normalized_term

Term-at-a-Time Scoring
name:
  score_term_at_a_time
arguments:
  query: The query string, analyzed with the same analyzer as the documents
  inverted_index: A dict from term to its postings, each posting a (doc_id, tf) pair
  idf_table: A dict from term to idf, so that another variant can be passed in
  dl: A dict from doc_id to its length in terms
  avgdl: The average document length
  k1: The saturation setting
  b: The length-normalization setting
  trace: Optional list; when given, a (term, postings read, accumulator copy) triple is appended after each posting list
returns:
  ranked: A list of (doc_id, score) pairs for every document that entered the accumulator, sorted by falling score then rising doc_id
  postings_read: The number of postings read over the query's distinct terms

## Pass Check
tests/test_bm25_term_frequency.py passes: four tests, one per section. A test passes only if every row holds.

Inputs: fixture F05-v1, the five documents with `arrival = 0` from the F06A-L1 copy of `documents.json` (sha256 `06128ab7…`). The index is the F06A-L1 forward and inverted index: 98 terms, 154 postings. `dl` is the sum of a document's `tf` values and `avgdl` their mean. IDF is Lucene's, `idf = ln(1 + (N − df + 0.5) / (df + 0.5))`. The saturation-normalized term is `tf · (k1 + 1) / (tf + k1 · (1 − b + b · dl / avgdl))` at `k1 = 1.2`, `b = 0.75`. Query terms are the distinct analyzed terms in sorted order. Ties break by rising `doc_id`. Known values are full double-precision numbers computed on this machine (Python 3.12, tantivy 0.26.2) from the formulas in the spec; every one matches the spec's four-decimal value.

1. **Known values.** `dl` per document exact and `avgdl = 37.0` within `1e-9`. `df → idf` for `df` 1 to 5 within `1e-9`, with 66, 17, 8, 5 and 2 terms at each `df`, exact. The denominator and saturation-normalized term at `tf = 1` per document within `1e-9`. The nine values of the `retry` in `KB-0107.v1` trace: `tf`, `df`, `dl` exact, the rest within `1e-9`. `wq1` order exact, scores and each candidate's two contributions within `1e-9`. `wq2` three distinct scores in the K6 order. `wq3` and `wq4` order exact, scores within `1e-9`. Postings read 5, 6, 4, 3, exact; `KB-0112.v1` and `KB-0120.v1` absent from `wq1`'s list. The term-at-a-time score equals a candidate-by-candidate sum of the same contributions, written in the test, within `1e-9`. Reason for the tolerance: every value is a logarithm, a division, a product or a short sum in double precision; the two orders of summation differ by about `1e-16`.
2. **Against the library.** An in-memory Tantivy index over the five texts with the `default` tokenizer and a stored `raw` `doc_id` field; each query run as an OR query over `body`. `searcher.doc_freq("body", "retry")` equals our posting-list length 3, exact. The candidate sets of the four queries equal the hits, exact. Our scores with Tantivy's stored lengths (`KB-0101.v1` 40, `avgdl` 37.0) equal the hit scores within `1e-5`. The four ranked lists, both sorted by `(-score, doc_id)`, are equal, exact. The exact-length difference is printed, not asserted. Reason: the same formula, but Tantivy keeps a 32-bit score, so `1e-5`; the exact-length gap is a one-byte stored length, not the formula.
3. **The hazard: the IDF variant.** Raw Okapi `ln((N − df + 0.5) / (df + 0.5))` over the vocabulary: `retry` −0.3365 and `failed` 0.3365 within `1e-9`, 15 negative terms exact, mean 0.6658 and `BM25Okapi`'s floor `0.25 × mean` = 0.1664 within `1e-9`. `wq1` under the `BM25Okapi` table (negatives replaced by the floor) scores 0.5546, 0.4816, 0.1895 and under the FTS5 table (values ≤ 0 replaced by `1e-6`) scores 0.3292, 0.3222, 1.1385e-6, within `1e-9`. The order is the same list under all three tables, exact. The ratio of the Lucene score to the FTS5 score for `KB-0107.v1`, about 4.8, is printed, not asserted.
4. **Unseen cases.** `wq1` at `k1 = 0`: every contribution equals its term's `idf`; the list is `KB-0101.v1` 1.4145, `KB-0107.v1` 1.4145, `KB-0131.v1` 0.5390, order exact, scores within `1e-9`. `cache kafka` returns `[("KB-0120.v1", 2.2717)]` with 1 posting read. `kafka` returns `[]` with 0 postings read and no exception. `retry retry failed` equals `wq1`'s list with 5 postings read.

## Results
All tests passed.

Numbers from `results/F06B-L1.json`, four decimals shown.

`N = 5`, 98 terms, 154 postings. `dl` is the sum of a document's `tf` values: `KB-0101.v1` 41, `KB-0107.v1` 39, `KB-0112.v1` 35, `KB-0120.v1` 44, `KB-0131.v1` 26. `avgdl = 185 / 5 = 37.0`. The `df → idf` table under Lucene's variant:

| df | argument | idf | terms |
|---|---|---|---|
| 1 | 4.0000 | 1.3863 | 66 |
| 2 | 2.4000 | 0.8755 | 17 |
| 3 | 1.7143 | 0.5390 | 8 |
| 4 | 1.3333 | 0.2877 | 5 (`a`, `applies`, `record`, `scope`, `this`) |
| 5 | 1.0909 | 0.0870 | 2 (`the`, `to`) |

Every value is positive: the `1 +` inside the logarithm keeps a term in every document above zero.

The saturation-normalized term at `tf = 1` per document, `(k1 + 1) / (1 + k1 · (1 − b + b · dl / avgdl))`:

| doc_id | dl | denominator | term |
|---|---|---|---|
| KB-0101.v1 | 41 | 2.2973 | 0.9576 |
| KB-0107.v1 | 39 | 2.2486 | 0.9784 |
| KB-0112.v1 | 35 | 2.1514 | 1.0226 |
| KB-0120.v1 | 44 | 2.3703 | 0.9282 |
| KB-0131.v1 | 26 | 1.9324 | 1.1385 |

Shorter than average is above 1, longer is below.

The trace of `retry` in `KB-0107.v1`, in execution order: `tf` 2, `df` 3, argument `1 + 2.5 / 3.5` = 1.7143, `idf` = 0.5390, `dl` 39, denominator `2 + 1.2 × (0.25 + 0.75 × 39 / 37)` = 3.2486, saturation `2 × 2.2 / 3.2486` = 1.3544, contribution `0.5390 × 1.3544` = 0.7300.

`wq1 = retry failed`, distinct terms `failed`, `retry`. The accumulator after each posting list:

| after | postings | accumulator |
|---|---|---|
| `failed` | 2 | `KB-0101.v1` 0.8384, `KB-0107.v1` 0.8565 |
| `retry` | 3 | `KB-0101.v1` 1.3546, `KB-0107.v1` 1.5866, `KB-0131.v1` 0.6136 |

| doc_id | failed + retry | score |
|---|---|---|
| KB-0107.v1 | 0.8565 + 0.7300 | 1.5866 |
| KB-0101.v1 | 0.8384 + 0.5162 | 1.3546 |
| KB-0131.v1 | 0 + 0.6136 | 0.6136 |

The work: 5 postings read in 2 lists, 3 of 5 documents touched, `KB-0112.v1` and `KB-0120.v1` never opened. The order is the same as both TF-IDF lists of F06A-L1. The term-at-a-time sums equal the candidate-by-candidate sums exactly on this fixture.

`wq2 = TR-0042`, terms `0042`, `tr`, 6 postings read:

| doc_id | dl | 0042 + tr | score |
|---|---|---|---|
| KB-0131.v1 | 26 | 0.6136 + 0.6136 | 1.2273 |
| KB-0101.v1 | 41 | 0.5162 + 0.5162 | 1.0323 |
| KB-0120.v1 | 44 | 0.5003 + 0.5003 | 1.0006 |

Raw TF-IDF tied all three at 2.8109 and needed L2 normalization to separate them. BM25 separates them with no normalization step: each document has `tf` 1 for both terms, so the length term alone decides, and the shortest document is first.

`wq3 = wait seconds attempts`, 4 postings read: `KB-0101.v1` 1.3276 + 0.8384 + 1.3276 = 3.4936, `KB-0120.v1` 0 + 0.8126 + 0 = 0.8126. `wq4 = failed message`, 3 postings read: `KB-0107.v1` 0.8565 + 1.8776 = 2.7341, `KB-0101.v1` 0.8384 + 0 = 0.8384.

Against Tantivy: `doc_freq("body", "retry")` is 3, equal to our posting list. All four candidate sets and all four ranked lists are equal. With Tantivy's stored lengths our scores agree within 2.3e-7, the 32-bit rounding. With the exact lengths, `KB-0101.v1` differs on every query it appears in:

| query | Tantivy | ours, stored length 40 | ours, exact length 41 |
|---|---|---|---|
| wq1 | 1.3691 | 1.3691 | 1.3546 |
| wq2 | 1.0434 | 1.0434 | 1.0323 |
| wq3 | 3.5309 | 3.5309 | 3.4936 |
| wq4 | 0.8474 | 0.8474 | 0.8384 |

The largest exact-length gap is 3.7e-2 on `wq3`. Every other candidate agrees within 6.1e-8. What differed was the stored length, not the formula, and no rank changed.

The IDF variants on `wq1`, `k1` and `b` fixed. Raw Okapi `ln((N − df + 0.5) / (df + 0.5))` gives `retry` (`df` 3) −0.3365 and `failed` (`df` 2) 0.3365; 15 of 98 terms are negative, every term with `df` 3 or more. The mean of the raw values is 0.6658 and `BM25Okapi`'s floor `0.25 × mean` is 0.1664.

| doc_id | Lucene | Okapi, BM25Okapi floor | Okapi, FTS5 floor 1e-6 |
|---|---|---|---|
| KB-0107.v1 | 1.5866 | 0.5546 | 0.3292 |
| KB-0101.v1 | 1.3546 | 0.4816 | 0.3222 |
| KB-0131.v1 | 0.6136 | 0.1895 | 0.0000011 |

Same order under all three. The Lucene score of `KB-0107.v1` is 4.8195 times the FTS5-variant score.

Unseen cases. `wq1` at `k1 = 0`: every contribution is the term's `idf`, so `KB-0101.v1` and `KB-0107.v1` both score `0.8755 + 0.5390` = 1.4145 and the tie-break rule lists `KB-0101.v1` first, then `KB-0131.v1` 0.5390. `cache kafka`: `kafka` has no posting list; one candidate, `KB-0120.v1` 2.2717 (`tf` 4, `dl` 44), 1 posting read. `kafka`: an empty list, 0 postings read, no exception. `retry retry failed`: one distinct `retry`, so the list equals `wq1`'s and 5 postings are read.

## Why It Matters
TF-IDF multiplied a raw count by a worth per term. BM25 keeps the worth per term, now Lucene's IDF, and replaces the raw count by a term that saturates. The second `retry` in `KB-0107.v1` adds less than the first, and no number of repeats can push the term past `k1 + 1`. The same term also shrinks with document length through `b`: a match in a document shorter than average counts for more than 1, in a longer one for less. `wq2` showed the length term working alone. Three documents each mention `TR-0042` once; raw TF-IDF could not tell them apart and needed L2 normalization, a separate step with its own hazard. BM25 ranked them by length in the score itself, shortest first, because the same single mention is a larger share of what a short document says.

The scoring loop is the engine's loop. Term-at-a-time walks the query's posting lists and adds each posting's contribution into an accumulator keyed by document. It never opens a document outside those lists: `wq1` read 5 postings and never touched `KB-0112.v1` or `KB-0120.v1`. The cost of a query is the total length of its posting lists, not the size of the corpus. A term with a long posting list and a small IDF, `the` here, costs the most and adds the least to any score. The candidate-by-candidate sum gives the same numbers in a different order of addition, within `1e-16`, which is why the test writes both.

The hazard is the IDF variant. Raw Okapi IDF goes negative once a term is in more than half the corpus: `retry` is in 3 of 5 and scores −0.3365. Every library patches that differently. Lucene adds 1 inside the logarithm, `BM25Okapi` replaces negatives by a quarter of the mean, FTS5 replaces non-positives by 1e-6. On this fixture the order survived all three, but the number for `KB-0107.v1` moved by a factor of 4.8, and `KB-0131.v1` went from 0.6136 to a millionth. A BM25 score carries the name of its variant and settings, and two systems' numbers cannot be compared or added until both are named.

Tantivy taught the second lesson about representation. Its formula is the one here, and with the lengths it actually stores the agreement is a 32-bit rounding. But it keeps a document's length in one byte, so 41 became 40, and every score of `KB-0101.v1` rose by about 1 percent. Nothing in the formula differed. Comparing against a library means matching its inputs, not only its equation.

The unseen cases mark the edges. At `k1 = 0` the saturation term is 1 for every `tf` and `dl`, so BM25 collapses to presence-only scoring and the two documents that contain both terms tie; the tie-break rule, not the score, orders them. A term the index never saw has no posting list, so it costs and scores nothing. A repeated query term is one distinct term here; `BM25Okapi.get_scores` would count it twice, one more convention that belongs in the score's name.

## Toolkit
We added `document_lengths`, `lucene_idf`, `term_contribution` and `score_term_at_a_time`, with the helper `build_idf_table`. The analyzer A1 and the index builder are imported from `evalkit.tf_idf_weights`. `document_lengths` reads the forward index and returns `dl` per document and `avgdl`. `lucene_idf` is the formula for one `df` and `N`; `build_idf_table` applies it, or any variant passed in, to every posting-list length of the inverted index. `term_contribution` returns the denominator, the saturation-normalized term and the product so that each value exists for the trace. `score_term_at_a_time` takes the query, the inverted index, the IDF table, the lengths, `avgdl`, `k1` and `b`, walks each distinct query term's posting list in sorted order, accumulates per `doc_id`, and returns the ranked list with the number of postings read; an optional `trace` list receives the accumulator after each posting list. No function holds global state.
