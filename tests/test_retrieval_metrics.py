import json
import math
import warnings
from collections import Counter
from pathlib import Path

import pytest
from ranx import Qrels, Run, evaluate
from sklearn.metrics import dcg_score, ndcg_score
import evalkit.tf_idf_weights as tf_idf_weights
import evalkit.bm25_term_frequency as bm25
import evalkit.retrieval_metrics as rm


REPS_DIR = Path(__file__).resolve().parents[1] / "reps"
F06A_DIR = REPS_DIR / "F06A-L1_TF-IDF_Weights"
REP_DIR = REPS_DIR / "F06C-L1_Retrieval_Metrics"


@pytest.fixture
def inputs():
    # Fixture F05-v1: five documents with arrival = 0, the copy in the F06A-L1 folder
    # (data/F05_v1/documents.json in the course repository, sha256 06128ab7...).
    documents = [
        doc for doc in json.loads((F06A_DIR / "documents.json").read_text())["documents"]
        if doc["arrival"] == 0
    ]
    doc_ids = ["KB-0101.v1", "KB-0107.v1", "KB-0112.v1", "KB-0120.v1", "KB-0131.v1"]
    N = 5

    # The eight validation queries (role = "validation") of data/F05_v1/queries.json (sha256 d2574d07...)
    queries = {
        "vq1": "retry failed job",
        "vq2": "streaming importer retry",
        "vq3": "deploy reviewer",
        "vq4": "cache ttl",
        "vq5": "dead letter queue",
        "vq6": "staging tuesday",
        "vq7": "production deploy tuesday",
        "vq8": "wait between attempts",
    }
    query_ids = list(queries)

    # Fixture F06-v1: relevance_judgments.json (sha256 ee8b682b...), 40 rows, one per (query, document) pair,
    # graded under rule RJ-1; reviewed on 2026-10-16 with 0 changes
    rows = json.loads((REP_DIR / "relevance_judgments.json").read_text())["judgments"]

    # BM25 settings of the ranked lists: Lucene IDF, k1 = 1.2, b = 0.75, the F06B-L1 term-at-a-time function
    k1 = 1.2
    b = 0.75

    # Policy 1: a document is relevant when grade >= threshold; threshold 2 is the default, threshold 1 for contrast
    threshold = 2
    contrast_threshold = 1

    # Cutoffs
    cutoffs = (1, 3, 5)

    # The chosen lists for vq1 are built from A in the tests: B = A reversed, C = A without the answer document
    answer_doc_vq1 = "KB-0101.v1"

    return documents, doc_ids, N, queries, query_ids, rows, k1, b, threshold, contrast_threshold, cutoffs, answer_doc_vq1


@pytest.fixture
def known_results():
    # Full double-precision values computed on this machine (Python 3.12, scikit-learn 1.9.1, ranx 0.3.21) on
    # 17 October 2026 from the formulas in the spec; each matches the spec's four-decimal value.

    # K1: grade counts; the relevant set per query at threshold 2; vq1 at threshold 1; the unanswerable query
    known_grade_counts = {2: 7, 1: 8, 0: 25}
    known_relevant = {
        "vq1": {"KB-0101.v1"}, "vq2": {"KB-0107.v1"}, "vq3": {"KB-0112.v1"}, "vq4": {"KB-0120.v1"},
        "vq5": {"KB-0107.v1"}, "vq6": set(), "vq7": {"KB-0112.v1"}, "vq8": {"KB-0101.v1"},
    }
    known_relevant_vq1_t1 = {"KB-0101.v1", "KB-0107.v1", "KB-0131.v1"}
    known_unanswerable = ["vq6"]

    # K2: the eight BM25 lists as (doc_id, grade) pairs, in rank order
    known_lists = {
        "vq1": [("KB-0101.v1", 2), ("KB-0107.v1", 1), ("KB-0131.v1", 1)],
        "vq2": [("KB-0107.v1", 2), ("KB-0101.v1", 1), ("KB-0131.v1", 1)],
        "vq3": [("KB-0112.v1", 2), ("KB-0131.v1", 0), ("KB-0120.v1", 0)],
        "vq4": [("KB-0120.v1", 2)],
        "vq5": [("KB-0107.v1", 2), ("KB-0101.v1", 1)],
        "vq6": [("KB-0112.v1", 0)],
        "vq7": [("KB-0112.v1", 2), ("KB-0120.v1", 0), ("KB-0131.v1", 0)],
        "vq8": [("KB-0101.v1", 2)],
    }

    # K3: vq1 at threshold 2, k = 1, 3, 5; the list has 3 entries, precision@5 still divides by 5
    known_vq1_precision = {1: 1.0, 3: 1 / 3, 5: 0.2}
    known_vq1_recall = {1: 1.0, 3: 1.0, 5: 1.0}

    # K4: vq1 at threshold 1, relevant set of 3
    known_vq1_precision_t1 = {1: 1.0, 3: 1.0, 5: 0.6}
    known_vq1_recall_t1 = {1: 1 / 3, 3: 1.0, 5: 1.0}

    # K5: reciprocal rank per query; MRR over the defined values; ranx's convention counts vq6 as 0
    known_rr = {"vq1": 1.0, "vq2": 1.0, "vq3": 1.0, "vq4": 1.0, "vq5": 1.0, "vq6": None, "vq7": 1.0, "vq8": 1.0}
    known_mrr = (1.0, 7)
    known_mrr_ranx_convention = 0.875

    # K6: DCG@3 of lists A and B as three quotients grade / log2(rank + 1)
    known_dcg_terms_A = [2.0, 0.6309297535714575, 0.5]
    known_dcg_terms_B = [1.0, 0.6309297535714575, 1.0]
    known_dcg_A = 3.1309297535714578
    known_dcg_B = 2.6309297535714578

    # K7: ideal DCG@3 of vq1 from the judged grades 2, 1, 0, 0, 1 sorted and cut to 2, 1, 1; nDCG@3 of A and B
    known_ideal_vq1 = 3.1309297535714578
    known_ndcg_A = 1.0
    known_ndcg_B = 0.8403030283801005
    known_ndcg_B_burges = 0.7579237460681981

    # K8: nDCG@3 of vq5 and vq8; vq6 has ideal 0 and is undefined
    known_dcg_vq5 = 2.6309297535714578
    known_ndcg_vq5 = 0.8403030283801005
    known_dcg_vq8 = 2.0
    known_ndcg_vq8 = 0.6387878864795979
    known_ideal_vq6 = 0.0
    known_ndcg_vq6 = None

    # K9: the three lists of vq1 at k = 3: recall@3, reciprocal rank, nDCG@3; C's DCG over the same ideal
    known_three_lists = {
        "A": (1.0, 1.0, 1.0),
        "B": (1.0, 1 / 3, 0.8403030283801005),
        "C": (0.0, 0.0, 0.5209090851403014),
    }
    known_dcg_C = 1.6309297535714575

    # K10: the per-query table at threshold 2 and the means over the defined values with their counts
    answerable_row = {"P1": 1.0, "P3": 1 / 3, "P5": 0.2, "R1": 1.0, "R3": 1.0, "R5": 1.0, "RR": 1.0, "nDCG3": 1.0}
    known_table = {query_id: dict(answerable_row) for query_id in ("vq1", "vq2", "vq3", "vq4", "vq5", "vq7", "vq8")}
    known_table["vq5"]["nDCG3"] = 0.8403030283801005
    known_table["vq8"]["nDCG3"] = 0.6387878864795979
    known_table["vq6"] = {"P1": 0.0, "P3": 0.0, "P5": 0.0, "R1": None, "R3": None, "R5": None, "RR": None, "nDCG3": None}
    known_means = {
        "P3": (0.29166666666666663, 8),
        "R3": (1.0, 7),
        "RR": (1.0, 7),
        "nDCG3": (0.9255844164085284, 7),
    }

    # K11: ranx means over 8, vq6 scored 0 in every metric
    known_ranx_means = {"mrr-l2": 0.875, "recall@3-l2": 0.875, "ndcg@3": 0.8098863643574623, "precision@3-l2": 0.29166666666666663}

    # K12: scikit-learn on vq1; y_true is one grade per document in corpus order; lists encoded as descending integers
    known_sklearn_y_true = [2, 1, 0, 0, 1]
    known_sklearn_encoding = {"A": [5, 4, 2, 1, 3], "B": [3, 4, 2, 1, 5], "C": [3, 5, 2, 1, 4]}
    known_sklearn_dcg = {"A": 3.1309297535714573, "B": 2.6309297535714573}
    known_sklearn_ndcg = {"A": 1.0, "B": 0.8403030283801005, "C": 0.8403030283801005}
    known_sklearn_tied = 0.9074747423469345
    known_sklearn_all_zero = 0.0

    # K13: the hazard, an ideal built from the list's own grades
    known_wrong_ideal = {"A": 3.1309297535714578, "B": 3.1309297535714578, "C": 1.6309297535714575}
    known_wrong_ndcg = {"A": 1.0, "B": 0.8403030283801005, "C": 1.0}
    known_wrong_ideal_vq8 = 2.0
    known_wrong_ndcg_vq8 = 1.0

    # K14: unseen cases
    known_empty_list = {"P3": 0.0, "R3": 0.0, "RR": 0.0, "nDCG3": 0.0}
    known_vq5_relevant_t1 = {"KB-0101.v1", "KB-0107.v1", "KB-0131.v1"}
    known_vq5_non_candidate = "KB-0131.v1"
    known_vq5_recall5_t1 = 2 / 3
    known_vq5_recall5_t2 = 1.0
    known_vq8_recall5_t1 = 1 / 3
    known_mean_recall5_t1 = (0.8571428571428571, 7)
    known_k10 = {"P": 0.1, "R": 1.0, "nDCG": 1.0}

    return {
        "grade_counts": known_grade_counts, "relevant": known_relevant, "relevant_vq1_t1": known_relevant_vq1_t1,
        "unanswerable": known_unanswerable, "lists": known_lists,
        "vq1_precision": known_vq1_precision, "vq1_recall": known_vq1_recall,
        "vq1_precision_t1": known_vq1_precision_t1, "vq1_recall_t1": known_vq1_recall_t1,
        "rr": known_rr, "mrr": known_mrr, "mrr_ranx_convention": known_mrr_ranx_convention,
        "dcg_terms_A": known_dcg_terms_A, "dcg_terms_B": known_dcg_terms_B, "dcg_A": known_dcg_A, "dcg_B": known_dcg_B,
        "ideal_vq1": known_ideal_vq1, "ndcg_A": known_ndcg_A, "ndcg_B": known_ndcg_B, "ndcg_B_burges": known_ndcg_B_burges,
        "dcg_vq5": known_dcg_vq5, "ndcg_vq5": known_ndcg_vq5, "dcg_vq8": known_dcg_vq8, "ndcg_vq8": known_ndcg_vq8,
        "ideal_vq6": known_ideal_vq6, "ndcg_vq6": known_ndcg_vq6,
        "three_lists": known_three_lists, "dcg_C": known_dcg_C,
        "table": known_table, "means": known_means,
        "ranx_means": known_ranx_means,
        "sklearn_y_true": known_sklearn_y_true, "sklearn_encoding": known_sklearn_encoding,
        "sklearn_dcg": known_sklearn_dcg, "sklearn_ndcg": known_sklearn_ndcg,
        "sklearn_tied": known_sklearn_tied, "sklearn_all_zero": known_sklearn_all_zero,
        "wrong_ideal": known_wrong_ideal, "wrong_ndcg": known_wrong_ndcg,
        "wrong_ideal_vq8": known_wrong_ideal_vq8, "wrong_ndcg_vq8": known_wrong_ndcg_vq8,
        "empty_list": known_empty_list,
        "vq5_relevant_t1": known_vq5_relevant_t1, "vq5_non_candidate": known_vq5_non_candidate,
        "vq5_recall5_t1": known_vq5_recall5_t1, "vq5_recall5_t2": known_vq5_recall5_t2,
        "vq8_recall5_t1": known_vq8_recall5_t1, "mean_recall5_t1": known_mean_recall5_t1,
        "k10": known_k10,
    }


TOL = 1e-9
RANX_METRICS = [
    "precision@1-l2", "precision@3-l2", "precision@5-l2",
    "recall@1-l2", "recall@3-l2", "recall@5-l2",
    "mrr-l2", "ndcg@3",
]
# Our table's column for each ranx metric name
RANX_COLUMNS = dict(zip(RANX_METRICS, ("P1", "P3", "P5", "R1", "R3", "R5", "RR", "nDCG3")))


def _close(actual, known, context=None):
    """1e-9 on a number; exact on None, so that an undefined value and a 0 never pass for each other."""
    if known is None or actual is None:
        assert actual is known, (context, actual, known)
    else:
        assert abs(actual - known) <= TOL, (context, actual, known)


def _bm25_runs(documents, queries, k1, b):
    """The BM25 (doc_id, score) list of every query from the F06A-L1 index and the F06B-L1 scorer; no ties."""
    forward, inverted = tf_idf_weights.build_index(documents)
    dl, avgdl = bm25.document_lengths(forward)
    idf_table = bm25.build_idf_table(inverted, len(documents))
    runs = {}
    for query_id, text in queries.items():
        ranked, _ = bm25.score_term_at_a_time(text, inverted, idf_table, dl, avgdl, k1, b)
        assert len({score for _, score in ranked}) == len(ranked), (query_id, "tied scores")
        runs[query_id] = ranked
    return runs


def _ranked_lists(runs):
    """doc_id only, in rank order: the input the metric functions take."""
    return {query_id: [doc_id for doc_id, _ in ranked] for query_id, ranked in runs.items()}


def _three_lists(list_A, answer_doc):
    """STATE 13's lists for vq1: A as given, B reversed, C without the answer document in A's order."""
    list_B = list(reversed(list_A))
    list_C = [doc_id for doc_id in list_A if doc_id != answer_doc]
    return {"A": list(list_A), "B": list_B, "C": list_C}


def _table_row(ranked, judgments, threshold):
    """P@1, P@3, P@5, R@1, R@3, R@5, RR, nDCG@3 of one query from the toolkit functions."""
    relevant = rm.relevant_set(judgments, threshold)
    return {
        "P1": rm.precision_at_k(ranked, relevant, 1),
        "P3": rm.precision_at_k(ranked, relevant, 3),
        "P5": rm.precision_at_k(ranked, relevant, 5),
        "R1": rm.recall_at_k(ranked, relevant, 1),
        "R3": rm.recall_at_k(ranked, relevant, 3),
        "R5": rm.recall_at_k(ranked, relevant, 5),
        "RR": rm.reciprocal_rank(ranked, relevant),
        "nDCG3": rm.ndcg_at_k(ranked, judgments, 3),
    }


def _wrong_ideal_dcg(ranked, judgments, k):
    """The hazard, written in the test: the ideal from the grades of the documents on the list, not the judgments."""
    grades = sorted((judgments[doc_id] for doc_id in ranked), reverse=True)[:k]
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def _sklearn_encoding(ranked, doc_ids):
    """One descending integer per document: the list's entries first, then the others, lower still and distinct."""
    scores = {doc_id: len(doc_ids) - rank for rank, doc_id in enumerate(ranked)}
    for offset, doc_id in enumerate(doc_id for doc_id in doc_ids if doc_id not in scores):
        scores[doc_id] = len(doc_ids) - len(ranked) - offset
    return [scores[doc_id] for doc_id in doc_ids]


# 1. Known values

def test_known_values(inputs, known_results):
    documents, doc_ids, N, queries, query_ids, rows, k1, b, threshold, contrast_threshold, cutoffs, answer_doc = inputs
    known = known_results

    judgments = rm.build_judgments(rows)
    runs = _bm25_runs(documents, queries, k1, b)
    lists = _ranked_lists(runs)

    # K1: grade counts (exact); every pair judged
    assert set(judgments) == set(query_ids)
    assert all(set(judgments[query_id]) == set(doc_ids) for query_id in query_ids)
    assert dict(Counter(row["grade"] for row in rows)) == known["grade_counts"]
    assert dict(Counter(grade for per_query in judgments.values() for grade in per_query.values())) == known["grade_counts"]

    # K1: relevant set of each query at threshold 2; vq1 at threshold 1; the unanswerable query (exact)
    for query_id in query_ids:
        assert rm.relevant_set(judgments[query_id], threshold) == known["relevant"][query_id], query_id
    assert rm.relevant_set(judgments["vq1"], contrast_threshold) == known["relevant_vq1_t1"]
    unanswerable = [query_id for query_id in query_ids if not rm.relevant_set(judgments[query_id], 1)]
    assert unanswerable == known["unanswerable"]

    # K2: the eight lists as (doc_id, grade) pairs (exact)
    for query_id in query_ids:
        assert [(doc_id, judgments[query_id][doc_id]) for doc_id in lists[query_id]] == known["lists"][query_id], query_id

    # K3: vq1 precision@k and recall@k at threshold 2 (1e-9); a 3-entry list, precision@5 divides by 5
    list_A = lists["vq1"]
    relevant_vq1 = rm.relevant_set(judgments["vq1"], threshold)
    assert len(list_A) == 3
    for k in cutoffs:
        hits = sum(doc_id in relevant_vq1 for doc_id in list_A[:k])
        _close(rm.precision_at_k(list_A, relevant_vq1, k), known["vq1_precision"][k], ("P", k))
        _close(rm.precision_at_k(list_A, relevant_vq1, k), hits / k, ("P written", k))
        _close(rm.recall_at_k(list_A, relevant_vq1, k), known["vq1_recall"][k], ("R", k))
        _close(rm.recall_at_k(list_A, relevant_vq1, k), hits / len(relevant_vq1), ("R written", k))

    # K4: vq1 at threshold 1 for contrast (1e-9)
    relevant_vq1_t1 = rm.relevant_set(judgments["vq1"], contrast_threshold)
    assert len(relevant_vq1_t1) == 3
    for k in cutoffs:
        _close(rm.precision_at_k(list_A, relevant_vq1_t1, k), known["vq1_precision_t1"][k], ("P t1", k))
        _close(rm.recall_at_k(list_A, relevant_vq1_t1, k), known["vq1_recall_t1"][k], ("R t1", k))

    # K5: reciprocal rank per query, vq6 None (exact); MRR and its count (1e-9, exact on the count)
    rr = {query_id: rm.reciprocal_rank(lists[query_id], rm.relevant_set(judgments[query_id], threshold)) for query_id in query_ids}
    for query_id in query_ids:
        _close(rr[query_id], known["rr"][query_id], query_id)
    assert rr["vq6"] is None
    mrr, mrr_count = rm.mean_defined(rr)
    _close(mrr, known["mrr"][0])
    assert mrr_count == known["mrr"][1]
    _close(sum(value for value in rr.values() if value is not None) / mrr_count, known["mrr"][0])
    _close(sum(value or 0.0 for value in rr.values()) / len(query_ids), known["mrr_ranx_convention"])

    # K6: DCG@3 of A and B as the sum of three quotients, written here (1e-9)
    three = _three_lists(list_A, answer_doc)
    list_B = three["B"]
    grades_A = [judgments["vq1"][doc_id] for doc_id in list_A]
    grades_B = [judgments["vq1"][doc_id] for doc_id in list_B]
    assert grades_A == [2, 1, 1]
    assert grades_B == [1, 1, 2]
    terms_A = [grades_A[0] / math.log2(2), grades_A[1] / math.log2(3), grades_A[2] / math.log2(4)]
    terms_B = [grades_B[0] / math.log2(2), grades_B[1] / math.log2(3), grades_B[2] / math.log2(4)]
    for actual, expected in zip(terms_A + terms_B, known["dcg_terms_A"] + known["dcg_terms_B"]):
        _close(actual, expected)
    _close(rm.dcg_at_k(list_A, judgments["vq1"], 3), sum(terms_A), "DCG A written")
    _close(rm.dcg_at_k(list_B, judgments["vq1"], 3), sum(terms_B), "DCG B written")
    _close(rm.dcg_at_k(list_A, judgments["vq1"], 3), known["dcg_A"], "DCG A")
    _close(rm.dcg_at_k(list_B, judgments["vq1"], 3), known["dcg_B"], "DCG B")

    # K7: ideal DCG@3 from the judged grades 2, 1, 0, 0, 1 sorted and cut; both nDCG@3 (1e-9)
    assert sorted(judgments["vq1"].values(), reverse=True)[:3] == [2, 1, 1]
    ideal_vq1 = rm.ideal_dcg_at_k(judgments["vq1"], 3)
    _close(ideal_vq1, known["ideal_vq1"], "ideal vq1")
    _close(ideal_vq1, 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4), "ideal written")
    _close(rm.ndcg_at_k(list_A, judgments["vq1"], 3), known["ndcg_A"], "nDCG A")
    _close(rm.ndcg_at_k(list_B, judgments["vq1"], 3), known["ndcg_B"], "nDCG B")
    _close(rm.ndcg_at_k(list_B, judgments["vq1"], 3), rm.dcg_at_k(list_B, judgments["vq1"], 3) / ideal_vq1, "nDCG B ratio")

    # K8: nDCG@3 of vq5 and vq8 (1e-9); vq6 ideal 0 and None (exact)
    _close(rm.dcg_at_k(lists["vq5"], judgments["vq5"], 3), known["dcg_vq5"], "DCG vq5")
    _close(rm.ideal_dcg_at_k(judgments["vq5"], 3), known["ideal_vq1"], "ideal vq5")
    _close(rm.ndcg_at_k(lists["vq5"], judgments["vq5"], 3), known["ndcg_vq5"], "nDCG vq5")
    _close(rm.dcg_at_k(lists["vq8"], judgments["vq8"], 3), known["dcg_vq8"], "DCG vq8")
    _close(rm.ndcg_at_k(lists["vq8"], judgments["vq8"], 3), known["ndcg_vq8"], "nDCG vq8")
    _close(rm.ideal_dcg_at_k(judgments["vq6"], 3), known["ideal_vq6"], "ideal vq6")
    assert rm.ndcg_at_k(lists["vq6"], judgments["vq6"], 3) is None

    # K9: recall@3, reciprocal rank, nDCG@3 of A, B, C (1e-9); C's DCG over the same ideal
    for name, ranked in three.items():
        values = (
            rm.recall_at_k(ranked, relevant_vq1, 3),
            rm.reciprocal_rank(ranked, relevant_vq1),
            rm.ndcg_at_k(ranked, judgments["vq1"], 3),
        )
        for actual, expected in zip(values, known["three_lists"][name]):
            _close(actual, expected, name)
    _close(rm.dcg_at_k(three["C"], judgments["vq1"], 3), known["dcg_C"], "DCG C")
    _close(rm.dcg_at_k(three["C"], judgments["vq1"], 3), 1 / math.log2(2) + 1 / math.log2(3), "DCG C written")

    # K10: the per-query table (1e-9, exact on None) and the four means with their counts
    table = {query_id: _table_row(lists[query_id], judgments[query_id], threshold) for query_id in query_ids}
    for query_id in query_ids:
        for column, expected in known["table"][query_id].items():
            _close(table[query_id][column], expected, (query_id, column))
    for column, (expected_mean, expected_count) in known["means"].items():
        actual_mean, actual_count = rm.mean_defined({query_id: table[query_id][column] for query_id in query_ids})
        _close(actual_mean, expected_mean, column)
        assert actual_count == expected_count, column


# 2. Against the libraries: ranx and scikit-learn

def test_against_libraries(inputs, known_results):
    documents, doc_ids, N, queries, query_ids, rows, k1, b, threshold, contrast_threshold, cutoffs, answer_doc = inputs
    known = known_results

    judgments = rm.build_judgments(rows)
    runs = _bm25_runs(documents, queries, k1, b)
    lists = _ranked_lists(runs)
    table = {query_id: _table_row(lists[query_id], judgments[query_id], threshold) for query_id in query_ids}

    # ranx: Qrels from the pairs with grade > 0, Run from the BM25 (doc_id, score) pairs; no ties checked in _bm25_runs
    qrels = Qrels({query_id: {doc_id: grade for doc_id, grade in judgments[query_id].items() if grade > 0} for query_id in query_ids})
    run = Run({query_id: dict(runs[query_id]) for query_id in query_ids})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numba cast warning from the precision kernel; the values are unchanged
        per_query = evaluate(qrels, run, RANX_METRICS, return_mean=False)
        ranx_means = evaluate(qrels, run, RANX_METRICS)

    # the eight metrics for the 7 answerable queries and precision@k for vq6 (1e-9); the largest difference is printed
    largest = 0.0
    for metric, values in per_query.items():
        assert len(values) == len(query_ids)
        column = RANX_COLUMNS[metric]
        for query_id, library_value in zip(query_ids, values):
            ours = table[query_id][column]
            if ours is None:
                assert query_id == "vq6"
                assert float(library_value) == 0.0, (metric, query_id)  # ranx's convention: 0 for a query with no relevant document
                continue
            assert abs(ours - float(library_value)) <= TOL, (metric, query_id, ours, float(library_value))
            largest = max(largest, abs(ours - float(library_value)))
    print(f"\nlargest |ours - ranx| over the defined per-query values: {largest:.1e}")

    # reported, not asserted: policy 2 means over 7 against ranx's means over 8 (vq6 kept as 0)
    for metric, column in (("mrr-l2", "RR"), ("recall@3-l2", "R3"), ("ndcg@3", "nDCG3"), ("precision@3-l2", "P3")):
        ours_mean, ours_count = rm.mean_defined({query_id: table[query_id][column] for query_id in query_ids})
        library_mean = float(ranx_means[metric])
        assert abs(library_mean - known["ranx_means"][metric]) <= TOL, metric
        print(f"{column}: policy 2 mean {ours_mean:.4f} over {ours_count}; ranx {library_mean:.4f} over {len(query_ids)}; "
              f"difference {ours_mean - library_mean:.4f} (ranx scores vq6 as 0 and keeps it)")

    # scikit-learn on vq1: y_true one grade per document in corpus order; each list as descending integers
    three = _three_lists(lists["vq1"], answer_doc)
    y_true = [[judgments["vq1"][doc_id] for doc_id in doc_ids]]
    assert y_true[0] == known["sklearn_y_true"]
    for name, ranked in three.items():
        encoding = _sklearn_encoding(ranked, doc_ids)
        assert encoding == known["sklearn_encoding"][name], name
        assert sorted(encoding) == list(range(1, N + 1)), name  # distinct, no ties
    for name in ("A", "B"):
        encoding = [_sklearn_encoding(three[name], doc_ids)]
        library_dcg = float(dcg_score(y_true, encoding, k=3))
        library_ndcg = float(ndcg_score(y_true, encoding, k=3))
        ours_dcg = rm.dcg_at_k(three[name], judgments["vq1"], 3)
        ours_ndcg = rm.ndcg_at_k(three[name], judgments["vq1"], 3)
        assert abs(library_dcg - known["sklearn_dcg"][name]) <= TOL, name
        assert abs(library_ndcg - known["sklearn_ndcg"][name]) <= TOL, name
        assert abs(ours_dcg - library_dcg) <= TOL, (name, ours_dcg, library_dcg)
        assert abs(ours_ndcg - library_ndcg) <= TOL, (name, ours_ndcg, library_ndcg)
        print(f"list {name}: |ours - sklearn| DCG@3 {abs(ours_dcg - library_dcg):.1e}, nDCG@3 {abs(ours_ndcg - library_ndcg):.1e}")

    # reported, not asserted: list C, where scikit-learn ranks all five documents and a third one fills rank 3
    library_ndcg_C = float(ndcg_score(y_true, [_sklearn_encoding(three["C"], doc_ids)], k=3))
    ours_ndcg_C = rm.ndcg_at_k(three["C"], judgments["vq1"], 3)
    assert abs(library_ndcg_C - known["sklearn_ndcg"]["C"]) <= TOL
    print(f"list C: ours nDCG@3 {ours_ndcg_C:.4f} against sklearn {library_ndcg_C:.4f} "
          f"(sklearn cannot represent a document that was not retrieved)")

    # tied scores average the gain; an all-zero y_true scores 0 where we return None
    assert abs(float(ndcg_score(y_true, [[1, 1, 0, 0, 1]], k=3)) - known["sklearn_tied"]) <= TOL
    assert float(ndcg_score([[0, 0, 0, 0, 0]], [_sklearn_encoding(three["A"], doc_ids)], k=3)) == known["sklearn_all_zero"]


# 3. The hazard: the ideal order comes from the judgments, not from the list

def test_hazard(inputs, known_results):
    documents, doc_ids, N, queries, query_ids, rows, k1, b, threshold, contrast_threshold, cutoffs, answer_doc = inputs
    known = known_results

    judgments = rm.build_judgments(rows)
    lists = _ranked_lists(_bm25_runs(documents, queries, k1, b))
    three = _three_lists(lists["vq1"], answer_doc)

    # the wrong ideal from the list's own grades, and each list's DCG@3 divided by it (1e-9)
    for name, ranked in three.items():
        wrong_ideal = _wrong_ideal_dcg(ranked, judgments["vq1"], 3)
        wrong_ndcg = rm.dcg_at_k(ranked, judgments["vq1"], 3) / wrong_ideal
        _close(wrong_ideal, known["wrong_ideal"][name], ("wrong ideal", name))
        _close(wrong_ndcg, known["wrong_ndcg"][name], ("wrong nDCG", name))

    # A and B: the wrong ideal equals the right one, both nDCG@3 unchanged
    right_ideal = rm.ideal_dcg_at_k(judgments["vq1"], 3)
    for name in ("A", "B"):
        _close(_wrong_ideal_dcg(three[name], judgments["vq1"], 3), right_ideal, name)
        _close(rm.ndcg_at_k(three[name], judgments["vq1"], 3), known["wrong_ndcg"][name], name)

    # C: the wrong ideal is 1 + 0.6309 and makes C perfect; the right ideal is 3.1309 and gives 0.5209
    _close(_wrong_ideal_dcg(three["C"], judgments["vq1"], 3), 1 / math.log2(2) + 1 / math.log2(3), "wrong ideal C written")
    _close(right_ideal, known["ideal_vq1"], "right ideal C")
    _close(rm.ndcg_at_k(three["C"], judgments["vq1"], 3), known["three_lists"]["C"][2], "right nDCG C")
    assert rm.ndcg_at_k(three["C"], judgments["vq1"], 3) < known["wrong_ndcg"]["C"]

    # vq8: a one-entry list becomes perfect under the wrong ideal (right: 0.6388)
    wrong_ideal_vq8 = _wrong_ideal_dcg(lists["vq8"], judgments["vq8"], 3)
    _close(wrong_ideal_vq8, known["wrong_ideal_vq8"], "wrong ideal vq8")
    _close(rm.dcg_at_k(lists["vq8"], judgments["vq8"], 3) / wrong_ideal_vq8, known["wrong_ndcg_vq8"], "wrong nDCG vq8")
    _close(rm.ndcg_at_k(lists["vq8"], judgments["vq8"], 3), known["ndcg_vq8"], "right nDCG vq8")


# 4. Unseen cases: an empty list, a threshold that reveals a non-candidate, and a cutoff past the list

def test_unseen_cases(inputs, known_results):
    documents, doc_ids, N, queries, query_ids, rows, k1, b, threshold, contrast_threshold, cutoffs, answer_doc = inputs
    known = known_results

    judgments = rm.build_judgments(rows)
    lists = _ranked_lists(_bm25_runs(documents, queries, k1, b))

    # vq1 with an empty list: every metric 0.0, none None (exact); a system failure, not an unanswerable query
    relevant_vq1 = rm.relevant_set(judgments["vq1"], threshold)
    empty = {
        "P3": rm.precision_at_k([], relevant_vq1, 3),
        "R3": rm.recall_at_k([], relevant_vq1, 3),
        "RR": rm.reciprocal_rank([], relevant_vq1),
        "nDCG3": rm.ndcg_at_k([], judgments["vq1"], 3),
    }
    for column, expected in known["empty_list"].items():
        assert empty[column] is not None, column
        assert empty[column] == expected, column

    # vq5 at threshold 1: a relevant set of 3 with a document that is never a candidate; recall@5 2/3 against 1.0 at threshold 2
    relevant_vq5_t1 = rm.relevant_set(judgments["vq5"], contrast_threshold)
    assert relevant_vq5_t1 == known["vq5_relevant_t1"]
    assert len(relevant_vq5_t1) == 3
    assert known["vq5_non_candidate"] in relevant_vq5_t1
    assert known["vq5_non_candidate"] not in lists["vq5"]
    _close(rm.recall_at_k(lists["vq5"], relevant_vq5_t1, 5), known["vq5_recall5_t1"], "R5 vq5 t1")
    _close(rm.recall_at_k(lists["vq5"], rm.relevant_set(judgments["vq5"], threshold), 5), known["vq5_recall5_t2"], "R5 vq5 t2")

    # mean recall@5 at threshold 1 over the 7 answerable queries, vq8 at 1/3 (1e-9, exact on the count)
    recall5_t1 = {
        query_id: rm.recall_at_k(lists[query_id], rm.relevant_set(judgments[query_id], contrast_threshold), 5)
        for query_id in query_ids
    }
    _close(recall5_t1["vq8"], known["vq8_recall5_t1"], "R5 vq8 t1")
    assert recall5_t1["vq6"] is None
    mean_recall5_t1, count = rm.mean_defined(recall5_t1)
    _close(mean_recall5_t1, known["mean_recall5_t1"][0])
    assert count == known["mean_recall5_t1"][1]

    # vq1 list A at k = 10: precision's divisor grows, recall and nDCG do not change (1e-9)
    list_A = lists["vq1"]
    _close(rm.precision_at_k(list_A, relevant_vq1, 10), known["k10"]["P"], "P@10")
    _close(rm.recall_at_k(list_A, relevant_vq1, 10), known["k10"]["R"], "R@10")
    _close(rm.ndcg_at_k(list_A, judgments["vq1"], 10), known["k10"]["nDCG"], "nDCG@10")
    _close(rm.dcg_at_k(list_A, judgments["vq1"], 10), rm.dcg_at_k(list_A, judgments["vq1"], 3), "DCG@10 = DCG@3")
    _close(rm.ideal_dcg_at_k(judgments["vq1"], 10), rm.ideal_dcg_at_k(judgments["vq1"], 3), "ideal@10 = ideal@3")
