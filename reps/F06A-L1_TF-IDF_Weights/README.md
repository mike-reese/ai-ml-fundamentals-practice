# F06A-L1 -- TF-IDF Weights

## Mechanism
Calculate TF-IDF by hand and return a score per candidate that sorts into a ranked list.

## Task 
Create four toolkit functions that calculate the IDF, the Weight Matrix, the L2 Normalization, and the Score and Ranked list for terms in stored documents.

## Interface
Calculate IDF
name:
  calculate_idf
arguments:
  documents: A List of documents to calculate the term IDF from
returns:
  vocabulary: A List of the vocabulary across all documents
  idf_score: A List of the IDF of each vocabulary term in order, same shape and size as the vocabulary list.
  
Generate Weight Matrix
name:
  generate_weight_matrix
arguments:
  documents: A List of documents
  vocabulary: A List of the vocabulary across all documents
  idf_score: A List of the IDF of each vocabulary term in order, same shape and size as the vocabulary list.
returns:
  weight_matrix: A numpy array of shape (N, |V|), rows in the document order, columns in vocabulary order, each entry is tf * idf or 0 where the document doesn't contain the term.

L2 Normalization
name:
  l2_normalization
arguments:
  weight_matrix: A numpy array of shape (N, |V|), rows in the document order, columns in vocabulary order, each entry is tf * idf or 0 where the document doesn't contain the term.
returns:
  l2_normalized_weight_matrix: A numpy array with every row divided by its length, the square root of the sum of its squared entries over all columns.

Score and Rank
name:
  score_and_rank
arguments:
  query: The query string to score, analyzed with the same analyzer as the documents
  matrix: A numpy array of shape (N, |V|): the raw weight matrix or its L2-normalized form
  vocabulary: A List of the vocabulary across all documents, the column order of the matrix
  document_ids: A List of the document IDs, the row order of the matrix
returns:
  score: A list of (doc_id, score) pairs for every document with a score above 0, sorted by falling score then rising doc_id.


## Pass Check
tests/test_tf_idf_weights.py passes: four tests, one per section. A test passes only if every row holds.

Inputs: fixture F05-v1, the five documents with `arrival = 0` from `documents.json` (sha256 `06128ab7…`). Analyzer A1 lowercases the text and takes every run of letters and digits as a term. IDF is scikit-learn's smoothed variant, `idf = ln((1 + N) / (1 + df)) + 1`. Term frequency is the raw count. The query vector is binary. Ties break by rising `doc_id`. Known values are the full double-precision numbers from the course environment (Python 3.12, scikit-learn 1.9.1, numpy 2.5.3), which matched this machine's environment exactly.

1. **Known values.** Vocabulary has 98 terms, `'0'` first and `'would'` last, exact. `df → idf` for `df` 1 to 5 within `1e-9`, with 66, 17, 8, 5 and 2 terms at each `df`, exact. `W` has shape `(5, 98)`, 154 non-zero entries, `[37, 31, 28, 35, 23]` per row, exact. The three named weights of K4 within `1e-9`. Each document's vector length within `1e-9`, and every L2 row has length 1. `wq1` raw and L2: order exact, scores within `1e-9`. `wq2` raw: three equal scores in `doc_id` order; L2: the K9 order. `wq3` and `wq4` raw scores within `1e-9`. Reason for the tolerance: every value is a logarithm, a product, a short sum or a square root in double precision.
2. **Against the library.** `TfidfVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+", smooth_idf=True, sublinear_tf=False)` fit once with `norm=None` and once with `norm='l2'`. `get_feature_names_out()` equals our vocabulary, exact. `idf_` equals our IDF, `fit_transform(texts).toarray()` equals each matrix, and `X @ q` equals our scores for the four queries, all within `1e-9`. The 8 ranked lists, built from the positive entries sorted by `(-score, doc_id)`, are equal, exact. Reason: the same formula in double precision; `1e-9` still catches a wrong `df`, a wrong logarithm base or a missing `+ 1`.
3. **The hazard: the divisor of L2 normalization.** `wq2` L2 scores are three distinct values matching K9, and `wq1` L2 scores match K7, within `1e-9`. The wrong method, dividing each candidate's score by the length of its weights over the query's terms only, is computed and printed beside the right values but not asserted: it scores every `wq2` candidate `√2 = 1.4142` so the tie survives, and on `wq1` it puts `KB-0101.v1` at 1.4082 above `KB-0107.v1` at 1.3726, flipping the order.
4. **Unseen case.** `cache kafka` returns `[("KB-0120.v1", 8.3944)]` raw and `[("KB-0120.v1", 0.5635)]` L2, order exact and scores within `1e-9`. `kafka` alone returns `[]` with no exception. The library's `X @ q` with `q` built from the in-vocabulary terms and `norm='l2'` equals `[0, 0, 0, 0.5635, 0]` within `1e-9`.

## Results
All tests passed.

Numbers from `results/F06A-L1.json`, four decimals shown.

`N = 5`, vocabulary of 98 terms. The `df → idf` table:

| df | idf | terms |
|---|---|---|
| 1 | 2.0986 | 66 |
| 2 | 1.6931 | 17 |
| 3 | 1.4055 | 8 |
| 4 | 1.1823 | 5 (`a`, `applies`, `record`, `scope`, `this`) |
| 5 | 1.0000 | 2 (`the`, `to`) |

The floor is 1, never 0: a term in every document still carries its count.

Weights are `tf × idf`. The three named weights of K4: `W[KB-0107.v1, retry] = 2 × 1.4055 = 2.8109`, `W[KB-0101.v1, retry] = 1 × 1.4055 = 1.4055`, `W[KB-0120.v1, cache] = 4 × 2.0986 = 8.3944`.

Document vector lengths, over all 98 columns:

| doc_id | non-zero entries | ‖W[d]‖ | largest entry |
|---|---|---|---|
| KB-0101.v1 | 37 | 11.7224 | `job` 4.1972 |
| KB-0107.v1 | 31 | 11.5944 | `message` 4.1972 |
| KB-0112.v1 | 28 | 12.0856 | `deploy` 4.2164 |
| KB-0120.v1 | 35 | 14.8976 | `cache` 8.3944 |
| KB-0131.v1 | 23 | 9.3643 | `the` 3.0000 |

`wq1 = retry failed`, a score is the sum of the document's weights over the query's terms:

| doc_id | raw | L2 |
|---|---|---|
| KB-0107.v1 | 1.6931 + 2.8109 = 4.5041 | 0.3885 |
| KB-0101.v1 | 1.6931 + 1.4055 = 3.0986 | 0.2643 |
| KB-0131.v1 | 0 + 1.4055 = 1.4055 | 0.1501 |

`KB-0112.v1` and `KB-0120.v1` score 0 and are not candidates. The order is the same on both matrices.

`wq2 = TR-0042`, which the analyzer splits into `tr` and `0042`:

| doc_id | raw | L2 |
|---|---|---|
| KB-0101.v1 | 1.4055 + 1.4055 = 2.8109 | 0.2398 |
| KB-0120.v1 | 1.4055 + 1.4055 = 2.8109 | 0.1887 |
| KB-0131.v1 | 1.4055 + 1.4055 = 2.8109 | 0.3002 |

Raw: a three-way tie, broken by rising `doc_id`. L2: the tie is gone and `KB-0131.v1`, the shortest vector, is first. The same two matched terms count for more in a document that says less overall.

The wrong divisor, printed by the hazard test: dividing by the length of the query's slice scores every `wq2` candidate 1.4142 so the tie survives, and on `wq1` it scores `KB-0101.v1` 1.4082 above `KB-0107.v1` 1.3726, the opposite of the right order.

Unseen query `cache kafka`: `kafka` is outside the vocabulary and contributes nothing. One candidate, `KB-0120.v1`, raw 8.3944, L2 0.5635. `kafka` alone has no candidate and an empty ranked list, with no exception.

## Why It Matters
A raw term count says how often a word appears in one document. It says nothing about how useful that word is for telling documents apart. `the` and `to` appear in every one of our five documents, so matching on them separates nothing. `cache` appears in one, so a match on it is strong evidence. IDF encodes that: a term's weight falls as the number of documents containing it rises. The smoothed variant keeps a floor of 1 so a term in every document is not deleted, only flattened. Multiplying by the count gives the weight matrix, one number per (document, term) pair, and a query score is a sum of those weights over the query's terms.

Raw scores favor long documents. A document with more words has more chances to match, and more repeats to multiply. `wq2` showed the flip side: three documents that each mention `TR-0042` exactly once tie exactly, and the tie-break rule decides the order on `doc_id` alone, which is no evidence at all. L2 normalization divides each document vector by its own length, so each row has length 1 and a score becomes the cosine between the query and the document. The tie broke on content: `KB-0131.v1` is the shortest document, so the same match is a larger share of what it says.

The hazard is in the divisor. The length belongs to the whole document vector, over all 98 columns, and exists before any query arrives. Dividing by the length of only the query's slice gives every two-term match `√2` regardless of the document, so `wq2` stayed tied, and it inverted `wq1`. The error is quiet: the numbers look plausible and nothing raises. Only comparing against the library and the known values catches it.

The unseen case is the one real queries hit constantly. A term the corpus never saw has no column. Indexing it would raise; skipping it is the correct and universal behavior, and a query of only unseen terms returns an empty list rather than an error.

## Toolkit
We added `calculate_idf`, `generate_weight_matrix`, `l2_normalization` and `score_and_rank`, with the helpers `analyze`, `term_frequencies`, `invert` and `build_index`. `analyze` is analyzer A1. `term_frequencies` gives a document's raw counts. `build_index` produces the forward index (doc_id → term → tf) and, through `invert`, the inverted index (term → postings). `calculate_idf` sorts the inverted index into the vocabulary and takes each term's `df` as its posting count. `generate_weight_matrix` fills a `(N, |V|)` matrix with `tf × idf` per posting. `l2_normalization` divides each row by its full-vector length and leaves an all-zero row as zeros. `score_and_rank` builds the binary query vector, skips terms with no column, and returns the documents with a positive score sorted by falling score then rising `doc_id`.
