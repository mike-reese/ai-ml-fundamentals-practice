import math
import re

import numpy

TERM_PATTERN = r"[a-z0-9]+"


def analyze(text: str) -> list[str]:
    """
    Analyzer A1: lowercase the text, then every run of letters and digits is one term.
    """
    return re.findall(TERM_PATTERN, text.lower())


def term_frequencies(text: str) -> dict[str, int]:
    """
    Raw term count of one text: a dict from term to the number of times it occurs (sublinear_tf=False).
    """
    tf = {}
    for term in analyze(text):
        tf[term] = tf.get(term, 0) + 1
    return tf


def invert(forward: dict[str, dict[str, int]], doc_ids: list[str]) -> dict[str, list[tuple[str, int]]]:
    """
    Inverted index from a forward index: a dict from term to its postings, each posting a (doc_id, tf) pair in document order.
    """
    index = {}
    for doc_id in doc_ids:
        for term, tf in forward[doc_id].items():
            index.setdefault(term, []).append((doc_id, tf))
    return index


def build_index(documents: list[dict]) -> tuple[dict, dict]:
    """
    Forward index (doc_id -> term -> tf) and inverted index (term -> postings) of a list of documents.
    """
    doc_ids = [d["doc_id"] for d in documents]
    forward = {d["doc_id"]: term_frequencies(d["text"]) for d in documents}
    return forward, invert(forward, doc_ids)


def calculate_idf(documents: list[dict]) -> tuple[list[str], list[float]]:
    """
    Calculate IDF: Return the sorted vocabulary of a document list and the smoothed IDF of each term

    The vocabulary is every distinct term of every document, sorted. The IDF is scikit-learn's smoothed variant,
    idf = ln((1 + N) / (1 + df)) + 1, natural logarithm, where N is the number of documents and df the number of
    documents that contain the term. The floor is 1: a term in every document still counts.

    Parameters:
    ------
    documents: A list of documents. Each document is a dict with the fields doc_id and text

    Returns:
    ------
    vocabulary: A sorted list of the distinct terms across all documents. Column order of the weight matrix
    idf_score: A list of the IDF of each vocabulary term, in vocabulary order, the same length as the vocabulary
    """
    forward, index = build_index(documents)
    N = len(documents)

    vocabulary = sorted(index)
    idf_score = [math.log((1 + N) / (1 + len(index[term]))) + 1 for term in vocabulary]

    return vocabulary, idf_score


def generate_weight_matrix(documents: list[dict], vocabulary: list[str], idf_score: list[float]) -> numpy.ndarray:
    """
    Generate Weight Matrix: Return the TF-IDF weight of every (document, term) pair as a matrix

    Each entry is the raw term count in the document times the IDF of the term, or 0 where the document does not
    contain the term. One non-zero entry per posting.

    Parameters:
    ------
    documents: A list of documents. Each document is a dict with the fields doc_id and text
    vocabulary: A sorted list of the distinct terms across all documents
    idf_score: A list of the IDF of each vocabulary term, in vocabulary order

    Returns:
    ------
    weight_matrix: A numpy array of shape (N, |V|), rows in document order, columns in vocabulary order
    """
    if len(vocabulary) != len(idf_score):
        raise ValueError(f"vocabulary has {len(vocabulary)} terms but idf_score has {len(idf_score)} values")

    column = {term: i for i, term in enumerate(vocabulary)}
    weight_matrix = numpy.zeros((len(documents), len(vocabulary)), dtype=numpy.float64)

    for row, document in enumerate(documents):
        for term, tf in term_frequencies(document["text"]).items():
            if term in column:
                weight_matrix[row, column[term]] = tf * idf_score[column[term]]

    return weight_matrix


def l2_normalization(weight_matrix: numpy.ndarray) -> numpy.ndarray:
    """
    L2 Normalization: Divide every row of a weight matrix by its length

    The length of a row is the square root of the sum of its squared entries over all columns: the length of the
    whole document vector, which exists before any query. A row of all zeros is left as zeros.

    Parameters:
    ------
    weight_matrix: A numpy array of shape (N, |V|), each entry tf * idf or 0

    Returns:
    ------
    l2_normalized_weight_matrix: A numpy array of shape (N, |V|) where every non-zero row has length 1
    """
    weight_matrix = numpy.asarray(weight_matrix, dtype=numpy.float64)
    if weight_matrix.ndim != 2:
        raise ValueError(f"weight_matrix must have shape (N, |V|). Current shape: {weight_matrix.shape}")

    lengths = numpy.linalg.norm(weight_matrix, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0

    l2_normalized_weight_matrix = weight_matrix / lengths

    return l2_normalized_weight_matrix


def score_and_rank(query: str, matrix: numpy.ndarray, vocabulary: list[str], document_ids: list[str]) -> list[tuple[str, float]]:
    """
    Score and Rank: Score every document against a query and return the candidates as a ranked list

    The query vector is binary: 1 for each distinct query term that is in the vocabulary, 0 elsewhere. A query term
    outside the vocabulary has no column and contributes nothing. A document's score is the sum of its weights over
    the query's terms. Documents with a score of 0 are not candidates and are left off the list.

    Parameters:
    ------
    query: The query string, analyzed with the same analyzer as the documents
    matrix: A numpy array of shape (N, |V|): the raw weight matrix or its L2-normalized form
    vocabulary: A sorted list of the distinct terms across all documents, the column order of the matrix
    document_ids: A list of the N document IDs, the row order of the matrix

    Returns:
    ------
    score: A list of (doc_id, score) pairs sorted by falling score, then rising doc_id on a tie
    """
    matrix = numpy.asarray(matrix, dtype=numpy.float64)
    if matrix.shape != (len(document_ids), len(vocabulary)):
        raise ValueError(f"matrix shape {matrix.shape} does not match {len(document_ids)} documents by {len(vocabulary)} terms")

    column = {term: i for i, term in enumerate(vocabulary)}
    query_vector = numpy.zeros(len(vocabulary), dtype=numpy.float64)
    for term in set(analyze(query)):
        if term in column:
            query_vector[column[term]] = 1.0

    scores = matrix @ query_vector

    score = [(doc_id, float(scores[i])) for i, doc_id in enumerate(document_ids) if scores[i] > 0]
    score.sort(key=lambda pair: (-pair[1], pair[0]))

    return score
