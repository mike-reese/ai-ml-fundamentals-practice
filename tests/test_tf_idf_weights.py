import json
import math
import re
from pathlib import Path

import pytest
import evalkit.tf_idf_weights as tf_idf_weights
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer


REP_DIR = Path(__file__).resolve().parents[1] / "reps" / "F06A-L1_TF-IDF_Weights"


@pytest.fixture
def inputs():
    # Fixture F05-v1: five documents with arrival = 0, copied from
    # data/F05_v1/documents.json in the course repository (sha256 06128ab7...).
    documents = [
        doc for doc in json.loads((REP_DIR / "documents.json").read_text())["documents"]
        if doc["arrival"] == 0
    ]
    doc_ids = ["KB-0101.v1", "KB-0107.v1", "KB-0112.v1", "KB-0120.v1", "KB-0131.v1"]
    N = 5

    # Analyzer A1: lowercase, then every run of letters and digits is one term
    analyzer = lambda text: re.findall(r"[a-z0-9]+", text.lower())

    # Vocabulary: distinct terms of the five documents, sorted; column order of W
    vocabulary = sorted({term for doc in documents for term in analyzer(doc["text"])})
    vocabulary_size = 98
    vocabulary_first = "0"
    vocabulary_last = "would"

    # IDF variant: scikit-learn's smoothed variant, natural logarithm
    idf_formula = lambda df, N: math.log((1 + N) / (1 + df)) + 1

    # Term frequency: the raw count, no logarithm
    sublinear_tf = False

    # Query vector: 1 for each distinct query term in the vocabulary, 0 elsewhere
    query_binary = True

    # Tie-break rule: equal scores ordered by rising doc_id
    tie_break_key = lambda doc_id_score: (-doc_id_score[1], doc_id_score[0])

    # Worked-example queries, from data/F05_v1/queries.json (sha256 d2574d07...)
    wq1 = "retry failed"
    wq2 = "TR-0042"
    wq3 = "wait seconds attempts"
    wq4 = "failed message"

    # Unseen query: one term in the vocabulary, one outside it
    unseen_query = "cache kafka"

    return (
        documents, doc_ids, N, analyzer,
        vocabulary, vocabulary_size, vocabulary_first, vocabulary_last,
        idf_formula, sublinear_tf, query_binary, tie_break_key,
        wq1, wq2, wq3, wq4, unseen_query,
    )


@pytest.fixture
def known_results():
    # Verified in the course environment (Python 3.12, scikit-learn 1.9.1,
    # numpy 2.5.3) on 17 October 2026. Floats are the full double-precision values
    # from that environment; the rep README shows them to four decimals.

    # K1: document length in terms (dl) and distinct terms, per document
    known_dl = {"KB-0101.v1": 41, "KB-0107.v1": 39, "KB-0112.v1": 35, "KB-0120.v1": 44, "KB-0131.v1": 26}
    known_distinct = {"KB-0101.v1": 37, "KB-0107.v1": 31, "KB-0112.v1": 28, "KB-0120.v1": 35, "KB-0131.v1": 23}
    known_postings = 154

    # K2: idf by df, and the number of vocabulary terms at each df
    known_idf_by_df = {1: 2.09861228866811, 2: 1.6931471805599454, 3: 1.4054651081081644, 4: 1.1823215567939547, 5: 1.0}
    known_terms_per_df = {1: 66, 2: 17, 3: 8, 4: 5, 5: 2}
    known_df4_terms = ["a", "applies", "record", "scope", "this"]
    known_df5_terms = ["the", "to"]
    known_idf_floor = 1.0

    # K3: df and idf of the query terms
    known_query_term_df = {"retry": 3, "failed": 2, "tr": 3, "0042": 3, "wait": 1, "seconds": 2, "attempts": 1, "message": 1}
    known_query_term_idf = {"retry": 1.4054651081081644, "failed": 1.6931471805599454, "tr": 1.4054651081081644, "0042": 1.4054651081081644, "wait": 2.09861228866811, "seconds": 1.6931471805599454, "attempts": 2.09861228866811, "message": 2.09861228866811}

    # K4: weight matrix W, shape (5, 98), one non-zero entry per posting
    known_W_shape = (5, 98)
    known_W_nonzero = 154
    known_W_nonzero_per_row = [37, 31, 28, 35, 23]
    known_W_entries = {("KB-0107.v1", "retry"): 2.8109302162163288, ("KB-0101.v1", "retry"): 1.4054651081081644, ("KB-0120.v1", "cache"): 8.39444915467244}

    # K5: document vector lengths; after L2 normalization every row has length 1
    known_lengths = {"KB-0101.v1": 11.722389668632664, "KB-0107.v1": 11.594364328787, "KB-0112.v1": 12.085629084525332, "KB-0120.v1": 14.897619616323807, "KB-0131.v1": 9.364341298285462}
    known_l2_row_length = 1.0

    # K6: wq1 raw scores and ranked list; KB-0112.v1 and KB-0120.v1 are not on the list
    known_wq1_raw = {"KB-0107.v1": 4.504077396776275, "KB-0101.v1": 3.09861228866811, "KB-0131.v1": 1.4054651081081644}
    known_wq1_raw_ranked = ["KB-0107.v1", "KB-0101.v1", "KB-0131.v1"]

    # K7: wq1 L2 scores, same order as K6
    known_wq1_l2 = {"KB-0107.v1": 0.38847126664748255, "KB-0101.v1": 0.26433281747658727, "KB-0131.v1": 0.15008691624316325}
    known_wq1_l2_ranked = ["KB-0107.v1", "KB-0101.v1", "KB-0131.v1"]

    # K8: wq2 raw scores, a three-way tie broken by rising doc_id
    known_wq2_raw = {"KB-0101.v1": 2.8109302162163288, "KB-0120.v1": 2.8109302162163288, "KB-0131.v1": 2.8109302162163288}
    known_wq2_raw_ranked = ["KB-0101.v1", "KB-0120.v1", "KB-0131.v1"]

    # K9: wq2 L2 scores; the tie is gone and the shortest vector is first
    known_wq2_l2 = {"KB-0131.v1": 0.3001738324863265, "KB-0101.v1": 0.23979156943894742, "KB-0120.v1": 0.18868317815930147}
    known_wq2_l2_ranked = ["KB-0131.v1", "KB-0101.v1", "KB-0120.v1"]

    # K10: wq3 and wq4 raw scores
    known_wq3_raw = {"KB-0101.v1": 5.890371757896165, "KB-0120.v1": 1.6931471805599454}
    known_wq4_raw = {"KB-0107.v1": 5.890371757896165, "KB-0101.v1": 1.6931471805599454}

    # K11: unseen query "cache kafka"; "kafka" is outside the vocabulary
    known_unseen_raw = {"KB-0120.v1": 8.39444915467244}
    known_unseen_l2 = {"KB-0120.v1": 0.5634758686867243}
    known_unseen_ranked = ["KB-0120.v1"]
    known_kafka_only_ranked = []

    return (
        known_dl, known_distinct, known_postings,
        known_idf_by_df, known_terms_per_df, known_df4_terms, known_df5_terms, known_idf_floor,
        known_query_term_df, known_query_term_idf,
        known_W_shape, known_W_nonzero, known_W_nonzero_per_row, known_W_entries,
        known_lengths, known_l2_row_length,
        known_wq1_raw, known_wq1_raw_ranked,
        known_wq1_l2, known_wq1_l2_ranked,
        known_wq2_raw, known_wq2_raw_ranked,
        known_wq2_l2, known_wq2_l2_ranked,
        known_wq3_raw, known_wq4_raw,
        known_unseen_raw, known_unseen_l2, known_unseen_ranked, known_kafka_only_ranked,
    )


TOL = 1e-9


def _build(documents, doc_ids):
    """Vocabulary, IDF, raw W and L2-normalized W from the toolkit functions."""
    vocabulary, idf_score = tf_idf_weights.calculate_idf(documents)
    W = tf_idf_weights.generate_weight_matrix(documents, vocabulary, idf_score)
    W_l2 = tf_idf_weights.l2_normalization(W)
    return vocabulary, numpy.asarray(idf_score, dtype=float), numpy.asarray(W, dtype=float), numpy.asarray(W_l2, dtype=float)


def _vectorizers():
    kwargs = dict(lowercase=True, token_pattern=r"[a-z0-9]+", smooth_idf=True, sublinear_tf=False)
    return TfidfVectorizer(norm=None, **kwargs), TfidfVectorizer(norm="l2", **kwargs)


def _binary_query_vector(query, analyzer, vocabulary):
    column = {term: i for i, term in enumerate(vocabulary)}
    q = numpy.zeros(len(vocabulary))
    for term in set(analyzer(query)):
        if term in column:
            q[column[term]] = 1.0
    return q


def _library_ranked(X, q, doc_ids, tie_break_key):
    scores = X @ q
    return sorted(
        [(doc_ids[i], float(scores[i])) for i in range(len(doc_ids)) if scores[i] > 0],
        key=tie_break_key,
    )


def _assert_ranked(actual, known_scores, known_order):
    """Exact on order, 1e-9 on scores, against a dict of known scores and a known order."""
    assert [doc_id for doc_id, _ in actual] == known_order
    for doc_id, score in actual:
        assert abs(score - known_scores[doc_id]) <= TOL, (doc_id, score, known_scores[doc_id])


def _assert_ranked_equal(actual, expected):
    """Exact on order, 1e-9 on scores, between two ranked lists."""
    assert [doc_id for doc_id, _ in actual] == [doc_id for doc_id, _ in expected]
    for (_, a), (_, b) in zip(actual, expected):
        assert abs(a - b) <= TOL


# 1. Known values

def test_known_values(inputs, known_results):
    (
        documents, doc_ids, N, analyzer,
        vocabulary_in, vocabulary_size, vocabulary_first, vocabulary_last,
        idf_formula, sublinear_tf, query_binary, tie_break_key,
        wq1, wq2, wq3, wq4, unseen_query,
    ) = inputs
    (
        known_dl, known_distinct, known_postings,
        known_idf_by_df, known_terms_per_df, known_df4_terms, known_df5_terms, known_idf_floor,
        known_query_term_df, known_query_term_idf,
        known_W_shape, known_W_nonzero, known_W_nonzero_per_row, known_W_entries,
        known_lengths, known_l2_row_length,
        known_wq1_raw, known_wq1_raw_ranked,
        known_wq1_l2, known_wq1_l2_ranked,
        known_wq2_raw, known_wq2_raw_ranked,
        known_wq2_l2, known_wq2_l2_ranked,
        known_wq3_raw, known_wq4_raw,
        known_unseen_raw, known_unseen_l2, known_unseen_ranked, known_kafka_only_ranked,
    ) = known_results

    vocabulary, idf_score, W, W_l2 = _build(documents, doc_ids)

    # vocabulary: length, first and last term (exact)
    assert len(vocabulary) == vocabulary_size
    assert vocabulary[0] == vocabulary_first
    assert vocabulary[-1] == vocabulary_last
    assert list(vocabulary) == vocabulary_in
    assert len(idf_score) == vocabulary_size

    # df -> idf for df 1 to 5 (1e-9), and the number of terms at each df (exact)
    df = {term: sum(term in set(analyzer(doc["text"])) for doc in documents) for term in vocabulary}
    for d in range(1, 6):
        terms_at_df = [term for term in vocabulary if df[term] == d]
        assert len(terms_at_df) == known_terms_per_df[d], d
        for term in terms_at_df:
            assert abs(idf_score[vocabulary.index(term)] - known_idf_by_df[d]) <= TOL, (term, d)
    assert [term for term in vocabulary if df[term] == 4] == known_df4_terms
    assert [term for term in vocabulary if df[term] == 5] == known_df5_terms
    assert idf_score.min() >= known_idf_floor - TOL

    # W.shape, number of non-zero entries, non-zero entries per row (exact)
    assert W.shape == known_W_shape
    assert int((W != 0).sum()) == known_W_nonzero
    assert [int(n) for n in (W != 0).sum(axis=1)] == known_W_nonzero_per_row

    # the three named weights (1e-9)
    for (doc_id, term), value in known_W_entries.items():
        assert abs(W[doc_ids.index(doc_id), vocabulary.index(term)] - value) <= TOL, (doc_id, term)

    # ||W[d]|| per document; every normalized row's length is 1.0 (1e-9)
    lengths = numpy.linalg.norm(W, axis=1)
    for i, doc_id in enumerate(doc_ids):
        assert abs(lengths[i] - known_lengths[doc_id]) <= TOL, doc_id
    assert W_l2.shape == W.shape
    for length in numpy.linalg.norm(W_l2, axis=1):
        assert abs(length - known_l2_row_length) <= TOL

    # wq1 ranked list, raw and L2: order exact, scores 1e-9
    _assert_ranked(tf_idf_weights.score_and_rank(wq1, W, vocabulary, doc_ids), known_wq1_raw, known_wq1_raw_ranked)
    _assert_ranked(tf_idf_weights.score_and_rank(wq1, W_l2, vocabulary, doc_ids), known_wq1_l2, known_wq1_l2_ranked)

    # wq2 raw: three equal scores, list in doc_id order; L2: the list of K9
    wq2_raw = tf_idf_weights.score_and_rank(wq2, W, vocabulary, doc_ids)
    _assert_ranked(wq2_raw, known_wq2_raw, known_wq2_raw_ranked)
    wq2_raw_scores = [score for _, score in wq2_raw]
    assert max(wq2_raw_scores) - min(wq2_raw_scores) <= TOL
    assert [doc_id for doc_id, _ in wq2_raw] == sorted(doc_id for doc_id, _ in wq2_raw)
    _assert_ranked(tf_idf_weights.score_and_rank(wq2, W_l2, vocabulary, doc_ids), known_wq2_l2, known_wq2_l2_ranked)

    # wq3, wq4 raw scores (1e-9)
    for query, known_raw in ((wq3, known_wq3_raw), (wq4, known_wq4_raw)):
        ranked = tf_idf_weights.score_and_rank(query, W, vocabulary, doc_ids)
        assert {doc_id for doc_id, _ in ranked} == set(known_raw)
        for doc_id, score in ranked:
            assert abs(score - known_raw[doc_id]) <= TOL, (query, doc_id)


# 2. Against the library: scikit-learn's TfidfVectorizer

def test_against_library(inputs):
    (
        documents, doc_ids, N, analyzer,
        vocabulary_in, vocabulary_size, vocabulary_first, vocabulary_last,
        idf_formula, sublinear_tf, query_binary, tie_break_key,
        wq1, wq2, wq3, wq4, unseen_query,
    ) = inputs
    texts = [doc["text"] for doc in documents]

    vocabulary, idf_score, W, W_l2 = _build(documents, doc_ids)

    raw_vectorizer, l2_vectorizer = _vectorizers()
    X_raw = raw_vectorizer.fit_transform(texts).toarray()
    X_l2 = l2_vectorizer.fit_transform(texts).toarray()

    # vocabulary (exact), IDF in vocabulary order, W, normalized W (1e-9)
    assert list(raw_vectorizer.get_feature_names_out()) == list(vocabulary)
    assert numpy.max(numpy.abs(idf_score - raw_vectorizer.idf_)) <= TOL
    assert numpy.max(numpy.abs(W - X_raw)) <= TOL
    assert numpy.max(numpy.abs(W_l2 - X_l2)) <= TOL

    # raw and L2 scores for the 4 queries (1e-9); the 8 ranked lists (exact)
    ranked_lists_checked = 0
    for query in (wq1, wq2, wq3, wq4):
        q = _binary_query_vector(query, analyzer, vocabulary)
        for X, matrix in ((X_raw, W), (X_l2, W_l2)):
            ours = tf_idf_weights.score_and_rank(query, matrix, vocabulary, doc_ids)
            library = _library_ranked(X, q, doc_ids, tie_break_key)
            _assert_ranked_equal(ours, library)
            # X @ q scores all five documents; the 0 entries are the non-candidates our list omits
            library_scores = X @ q
            assert {doc_id for doc_id, _ in ours} == {doc_ids[i] for i in range(N) if library_scores[i] > 0}
            ranked_lists_checked += 1
    assert ranked_lists_checked == 8


# 3. The hazard: the divisor of L2 normalization

def _wrong_divisor_scores(query, analyzer, W, vocabulary, doc_ids, tie_break_key):
    """Divide each candidate's score by the length of its weights over the query's terms only."""
    columns = [vocabulary.index(term) for term in set(analyzer(query)) if term in vocabulary]
    wrong = []
    for i, doc_id in enumerate(doc_ids):
        query_slice = W[i, columns]
        if query_slice.sum() > 0:
            wrong.append((doc_id, float(query_slice.sum() / numpy.linalg.norm(query_slice))))
    return sorted(wrong, key=tie_break_key)


def test_hazard(inputs, known_results):
    (
        documents, doc_ids, N, analyzer,
        vocabulary_in, vocabulary_size, vocabulary_first, vocabulary_last,
        idf_formula, sublinear_tf, query_binary, tie_break_key,
        wq1, wq2, wq3, wq4, unseen_query,
    ) = inputs
    known_wq1_l2, known_wq1_l2_ranked = known_results[18], known_results[19]
    known_wq2_l2, known_wq2_l2_ranked = known_results[22], known_results[23]

    vocabulary, idf_score, W, W_l2 = _build(documents, doc_ids)

    # wq2 L2 scores: three distinct values, K9 (1e-9)
    wq2_right = tf_idf_weights.score_and_rank(wq2, W_l2, vocabulary, doc_ids)
    _assert_ranked(wq2_right, known_wq2_l2, known_wq2_l2_ranked)
    wq2_scores = [score for _, score in wq2_right]
    assert len(wq2_scores) == 3
    for a, b in zip(wq2_scores, wq2_scores[1:]):
        assert abs(a - b) > TOL

    # wq1 L2 scores: K7 (1e-9)
    wq1_right = tf_idf_weights.score_and_rank(wq1, W_l2, vocabulary, doc_ids)
    _assert_ranked(wq1_right, known_wq1_l2, known_wq1_l2_ranked)

    # Reported, not asserted: the wrong divisor scores every two-term match sqrt(2) on wq2,
    # so the tie survives; on wq1 KB-0101.v1 (1.4082) ranks above KB-0107.v1 (1.3726), the order flips
    wq2_wrong = _wrong_divisor_scores(wq2, analyzer, W, vocabulary, doc_ids, tie_break_key)
    wq1_wrong = _wrong_divisor_scores(wq1, analyzer, W, vocabulary, doc_ids, tie_break_key)
    print(f"\nwq2 L2 right: {wq2_right}")
    print(f"wq2 L2 wrong divisor (query slice only): {wq2_wrong}")
    print(f"wq1 L2 right: {wq1_right}")
    print(f"wq1 L2 wrong divisor (query slice only): {wq1_wrong}")


# 4. Unseen case: a query term outside the vocabulary

def test_unseen_case(inputs, known_results):
    (
        documents, doc_ids, N, analyzer,
        vocabulary_in, vocabulary_size, vocabulary_first, vocabulary_last,
        idf_formula, sublinear_tf, query_binary, tie_break_key,
        wq1, wq2, wq3, wq4, unseen_query,
    ) = inputs
    known_unseen_raw, known_unseen_l2, known_unseen_ranked, known_kafka_only_ranked = known_results[26:30]
    texts = [doc["text"] for doc in documents]

    vocabulary, idf_score, W, W_l2 = _build(documents, doc_ids)

    assert "cache" in vocabulary
    assert "kafka" not in vocabulary

    # cache kafka: ranked list, raw and L2 (order exact, scores 1e-9)
    _assert_ranked(tf_idf_weights.score_and_rank(unseen_query, W, vocabulary, doc_ids), known_unseen_raw, known_unseen_ranked)
    _assert_ranked(tf_idf_weights.score_and_rank(unseen_query, W_l2, vocabulary, doc_ids), known_unseen_l2, known_unseen_ranked)

    # kafka: an empty list, no exception
    assert tf_idf_weights.score_and_rank("kafka", W, vocabulary, doc_ids) == known_kafka_only_ranked
    assert tf_idf_weights.score_and_rank("kafka", W_l2, vocabulary, doc_ids) == known_kafka_only_ranked

    # library: X @ q with q built from the in-vocabulary terms, norm='l2' -> [0, 0, 0, 0.5635, 0]
    _, l2_vectorizer = _vectorizers()
    X_l2 = l2_vectorizer.fit_transform(texts).toarray()
    q = _binary_query_vector(unseen_query, analyzer, list(l2_vectorizer.get_feature_names_out()))
    expected = numpy.zeros(N)
    expected[doc_ids.index("KB-0120.v1")] = known_unseen_l2["KB-0120.v1"]
    assert numpy.max(numpy.abs(X_l2 @ q - expected)) <= TOL
