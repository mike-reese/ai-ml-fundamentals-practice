import math
from collections.abc import Callable

from evalkit.tf_idf_weights import analyze, build_index  # noqa: F401  (A1 and the index builder come from F06A-L1)


def document_lengths(forward_index: dict[str, dict[str, int]]) -> tuple[dict[str, int], float]:
    """
    Document Lengths: Return each document's length in terms, the sum of its tf values, and their mean avgdl

    Parameters:
    ------
    forward_index: A dict from doc_id to a dict from term to tf (the F06A-L1 forward index)

    Returns:
    ------
    dl: A dict from doc_id to its length in terms, in index order
    avgdl: The mean of dl over the corpus
    """
    if not forward_index:
        raise ValueError("forward_index is empty: avgdl is undefined")

    dl = {doc_id: sum(terms.values()) for doc_id, terms in forward_index.items()}
    avgdl = sum(dl.values()) / len(dl)

    return dl, avgdl


def lucene_idf(df: int, N: int) -> float:
    """
    Lucene IDF: idf = ln(1 + (N - df + 0.5) / (df + 0.5)), natural logarithm, always positive

    Parameters:
    ------
    df: The number of documents that contain the term
    N: The number of documents in the corpus

    Returns:
    ------
    idf: The term's worth under Lucene's variant, used by Tantivy
    """
    if N < 1:
        raise ValueError(f"N must be at least 1, got {N}")
    if not 0 <= df <= N:
        raise ValueError(f"df must be between 0 and N = {N}, got {df}")

    return math.log(1 + (N - df + 0.5) / (df + 0.5))


def build_idf_table(
    inverted_index: dict[str, list[tuple[str, int]]], N: int, idf: Callable[[int, int], float] = lucene_idf
) -> dict[str, float]:
    """
    Build IDF Table: One idf per vocabulary term, df taken as the length of the term's posting list

    Parameters:
    ------
    inverted_index: A dict from term to its postings, each posting a (doc_id, tf) pair (the F06A-L1 inverted index)
    N: The number of documents in the corpus
    idf: The IDF function of df and N; Lucene's unless another variant is passed

    Returns:
    ------
    idf_table: A dict from every term of the index to its idf
    """
    return {term: idf(len(postings), N) for term, postings in inverted_index.items()}


def term_contribution(tf: int, dl: int, avgdl: float, idf: float, k1: float, b: float) -> tuple[float, float, float]:
    """
    Term Contribution: idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)), returned with its two intermediate values

    The denominator shrinks the term for a document shorter than average and grows it for a longer one; the
    saturation-normalized term rises with tf but never beyond k1 + 1; the contribution is that term times the idf.

    Parameters:
    ------
    tf: The term's count in the document
    dl: The document's length in terms
    avgdl: The average document length over the corpus
    idf: The term's idf
    k1: The saturation setting; 0 makes the term 1 for every tf and dl
    b: The length-normalization setting; 0 ignores the document length, 1 normalizes fully

    Returns:
    ------
    denominator: tf + k1 * (1 - b + b * dl / avgdl)
    saturation_normalized_term: tf * (k1 + 1) / denominator
    contribution: idf * saturation_normalized_term
    """
    if avgdl <= 0:
        raise ValueError(f"avgdl must be positive, got {avgdl}")

    denominator = tf + k1 * (1 - b + b * dl / avgdl)
    saturation_normalized_term = tf * (k1 + 1) / denominator
    contribution = idf * saturation_normalized_term

    return denominator, saturation_normalized_term, contribution


def score_term_at_a_time(
    query: str,
    inverted_index: dict[str, list[tuple[str, int]]],
    idf_table: dict[str, float],
    dl: dict[str, int],
    avgdl: float,
    k1: float,
    b: float,
    trace: list | None = None,
) -> tuple[list[tuple[str, float]], int]:
    """
    Term-at-a-Time Scoring: Walk each distinct query term's posting list, add every posting's contribution into an accumulator per doc_id, and return the ranked list and the number of postings read

    Terms outside, postings inside: the query's distinct terms are visited in sorted order; a term with no posting
    list costs and scores nothing. The list holds exactly the documents that entered the accumulator, sorted by
    falling score, then rising doc_id on a tie.

    Parameters:
    ------
    query: The query string, analyzed with the same analyzer as the documents
    inverted_index: A dict from term to its postings, each posting a (doc_id, tf) pair
    idf_table: A dict from term to idf; another variant can be passed in
    dl: A dict from doc_id to its length in terms
    avgdl: The average document length over the corpus
    k1: The saturation setting
    b: The length-normalization setting
    trace: Optional list; when given, (term, postings read in that list, a copy of the accumulator) is appended after each posting list

    Returns:
    ------
    ranked: A list of (doc_id, score) pairs sorted by falling score, then rising doc_id
    postings_read: The number of postings read over the query's distinct terms
    """
    accumulator: dict[str, float] = {}
    postings_read = 0

    for term in sorted(set(analyze(query))):
        postings = inverted_index.get(term, [])
        for doc_id, tf in postings:
            _, _, contribution = term_contribution(tf, dl[doc_id], avgdl, idf_table[term], k1, b)
            accumulator[doc_id] = accumulator.get(doc_id, 0.0) + contribution
        postings_read += len(postings)
        if trace is not None:
            trace.append((term, len(postings), dict(accumulator)))

    ranked = sorted(accumulator.items(), key=lambda pair: (-pair[1], pair[0]))

    return ranked, postings_read
