import math

from evalkit.tf_idf_weights import analyze, build_index  # noqa: F401  (A1 and the index builder come from F06A-L1)
from evalkit.bm25_term_frequency import document_lengths, build_idf_table, score_term_at_a_time  # noqa: F401  (ranked lists come from F06B-L1)


def build_judgments(rows: list[dict]) -> dict[str, dict[str, int]]:
    """
    Build Judgments (helper): Fold the fixture rows into query_id -> {doc_id: grade}; every (query, document) pair appears once

    Parameters:
    ------
    rows: The judgment rows of relevance_judgments.json, each with query_id, doc_id, grade and reason

    Returns:
    ------
    judgments: A dict from query_id to a dict from doc_id to its grade (0, 1 or 2 under rule RJ-1)
    """
    judgments: dict[str, dict[str, int]] = {}
    for row in rows:
        per_query = judgments.setdefault(row["query_id"], {})
        if row["doc_id"] in per_query:
            raise ValueError(f"duplicate judgment for ({row['query_id']}, {row['doc_id']})")
        per_query[row["doc_id"]] = row["grade"]
    return judgments


def relevant_set(judgments: dict[str, int], threshold: int) -> set[str]:
    """
    Relevant Set: The set of doc_ids whose grade is at or above the threshold (policy 1: threshold 2 means the document holds the answer)

    Parameters:
    ------
    judgments: One query's judgments, a dict from doc_id to grade
    threshold: The binary relevance threshold; a document is relevant when grade >= threshold

    Returns:
    ------
    relevant: The set of relevant doc_ids; empty for an unanswerable query
    """
    return {doc_id for doc_id, grade in judgments.items() if grade >= threshold}


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@k: The count of relevant documents among the top-k, divided by k; the divisor stays k when the list is shorter

    Defined for every query, an unanswerable one included: with an empty relevant set it is 0.0.

    Parameters:
    ------
    ranked: The ranked list of doc_ids, rank 1 first; a document outside it has no rank
    relevant: The query's relevant set
    k: The cutoff

    Returns:
    ------
    precision: The number of top-k entries in the relevant set, over k
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return sum(doc_id in relevant for doc_id in ranked[:k]) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    """
    Recall@k: The count of relevant documents among the top-k, divided by the size of the relevant set; None when that set is empty

    Parameters:
    ------
    ranked: The ranked list of doc_ids, rank 1 first
    relevant: The query's relevant set
    k: The cutoff

    Returns:
    ------
    recall: The number of top-k entries in the relevant set, over the size of the relevant set; None for an unanswerable query (policy 2)
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not relevant:
        return None
    return sum(doc_id in relevant for doc_id in ranked[:k]) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float | None:
    """
    Reciprocal Rank: 1 / rank of the first relevant document; 0.0 when none is on the list; None when the relevant set is empty

    An empty list for an answerable query scores 0.0, a system failure; an unanswerable query scores None, a property
    of the data. The two never share a value.

    Parameters:
    ------
    ranked: The ranked list of doc_ids, rank 1 first
    relevant: The query's relevant set

    Returns:
    ------
    rr: 1 / rank of the first relevant document, 0.0 if no relevant document is on the list, None if the relevant set is empty
    """
    if not relevant:
        return None
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


def dcg_at_k(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    """
    DCG@k: The sum over ranks 1..k of grade / log2(rank + 1); linear gain, an entry beyond the list's end adds 0

    Parameters:
    ------
    ranked: The ranked list of doc_ids, rank 1 first
    judgments: One query's judgments, a dict from doc_id to grade; the grade of each listed document is its gain
    k: The cutoff

    Returns:
    ------
    dcg: The discounted cumulative gain of the top-k
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return sum(judgments[doc_id] / math.log2(rank + 1) for rank, doc_id in enumerate(ranked[:k], start=1))


def ideal_dcg_at_k(judgments: dict[str, int], k: int) -> float:
    """
    Ideal DCG@k: The DCG of the query's judged grades sorted from largest to smallest and cut at k; the judgments, not the list

    A document the system never retrieved still counts here. This is the divisor of nDCG and the hazard of the rep:
    an ideal built from the list's own grades rewards a list for leaving the answer out.

    Parameters:
    ------
    judgments: One query's judgments, a dict from doc_id to grade, every document of the corpus included
    k: The cutoff

    Returns:
    ------
    ideal_dcg: The DCG of the best order the judgments allow; 0.0 when every grade is 0
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    ideal_grades = sorted(judgments.values(), reverse=True)[:k]
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))


def ndcg_at_k(ranked: list[str], judgments: dict[str, int], k: int) -> float | None:
    """
    nDCG@k: DCG@k divided by ideal DCG@k; None when the ideal is 0

    Parameters:
    ------
    ranked: The ranked list of doc_ids, rank 1 first
    judgments: One query's judgments, a dict from doc_id to grade
    k: The cutoff

    Returns:
    ------
    ndcg: dcg_at_k(ranked, judgments, k) / ideal_dcg_at_k(judgments, k); None for a query with no grade above 0 (policy 2)
    """
    ideal = ideal_dcg_at_k(judgments, k)
    if ideal == 0:
        return None
    return dcg_at_k(ranked, judgments, k) / ideal


def mean_defined(values: dict[str, float | None]) -> tuple[float | None, int]:
    """
    Mean Over the Defined Values: The mean of the values that are not None, with the number of queries it covers

    Parameters:
    ------
    values: A dict from query_id to a metric value, None where the metric is undefined for that query

    Returns:
    ------
    mean: The mean of the non-None values; None when there is none
    count: The number of queries the mean covers
    """
    defined = [value for value in values.values() if value is not None]
    if not defined:
        return None, 0
    return sum(defined) / len(defined), len(defined)
