import json
import math
from pathlib import Path

import numpy
import evalkit.tf_idf_weights as tf_idf_weights

REP_DIR = Path(__file__).parent


def load_documents():
    # Fixture F05-v1: the five documents with arrival = 0, in file order
    data = json.loads((REP_DIR / "documents.json").read_text())
    return [doc for doc in data["documents"] if doc["arrival"] == 0]


def worked_example_queries():
    wq1 = "retry failed"
    wq2 = "TR-0042"
    wq3 = "wait seconds attempts"
    wq4 = "failed message"
    unseen_query = "cache kafka"

    return wq1, wq2, wq3, wq4, unseen_query


def document_frequency(documents, vocabulary):
    return {term: sum(term in set(tf_idf_weights.analyze(doc["text"])) for doc in documents) for term in vocabulary}


def ranked_list_lines(query, matrix, vocabulary, doc_ids, W_for_terms, label):
    """Print each candidate's score as its sum of weights over the query's terms, then the ranked list."""
    query_terms = sorted(term for term in set(tf_idf_weights.analyze(query)) if term in vocabulary)
    columns = [vocabulary.index(term) for term in query_terms]
    ranked = tf_idf_weights.score_and_rank(query, matrix, vocabulary, doc_ids)
    print(f"  {label}")
    for doc_id, score in ranked:
        row = doc_ids.index(doc_id)
        parts = " + ".join(f"{matrix[row, c]:.4f}" for c in columns)
        print(f"    {doc_id}: {parts} = {score:.4f}")
    print(f"    ranked list: {[(doc_id, round(score, 4)) for doc_id, score in ranked]}")
    return ranked


if __name__ == "__main__":
    documents = load_documents()
    doc_ids = [doc["doc_id"] for doc in documents]
    N = len(documents)
    wq1, wq2, wq3, wq4, unseen_query = worked_example_queries()

    vocabulary, idf_score = tf_idf_weights.calculate_idf(documents)
    W = tf_idf_weights.generate_weight_matrix(documents, vocabulary, idf_score)
    W_l2 = tf_idf_weights.l2_normalization(W)
    column = {term: i for i, term in enumerate(vocabulary)}
    df = document_frequency(documents, vocabulary)

    # 1. Vocabulary, IDF and the df -> idf table
    print("1. Vocabulary and IDF")
    print(f"N = {N} documents: {doc_ids}")
    print(f"vocabulary size = {len(vocabulary)}, first term {vocabulary[0]!r}, last term {vocabulary[-1]!r}")
    print("idf = ln((1 + N) / (1 + df)) + 1")
    print(f"{'df':>4} | {'idf':>8} | {'terms':>5} | example terms")
    df_to_idf = {}
    for d in range(1, N + 1):
        terms_at_df = [term for term in vocabulary if df[term] == d]
        idf = math.log((1 + N) / (1 + d)) + 1
        df_to_idf[d] = {"idf": idf, "terms": len(terms_at_df)}
        examples = ", ".join(terms_at_df) if len(terms_at_df) <= 5 else ", ".join(terms_at_df[:5]) + ", ..."
        print(f"{d:>4} | {idf:>8.4f} | {len(terms_at_df):>5} | {examples}")

    # 2. The weight matrix restricted to the terms of wq1, and the three K4 weights
    print("")
    print(f"2. Weight matrix on the terms of wq1 ({wq1!r}): w = tf * idf")
    wq1_terms = sorted(set(tf_idf_weights.analyze(wq1)))
    header = " | ".join(f"{term + ' tf':>10} {term + ' idf':>12} {term + ' w':>10}" for term in wq1_terms)
    print(f"{'doc_id':<12} | {header}")
    wq1_table = []
    for row, doc in enumerate(documents):
        tf = tf_idf_weights.term_frequencies(doc["text"])
        cells = []
        entry = {"doc_id": doc["doc_id"]}
        for term in wq1_terms:
            t, i, w = tf.get(term, 0), idf_score[column[term]], W[row, column[term]]
            cells.append(f"{t:>10} {i:>12.4f} {w:>10.4f}")
            entry[term] = {"tf": t, "idf": i, "w": w}
        wq1_table.append(entry)
        print(f"{doc['doc_id']:<12} | {' | '.join(cells)}")
    print("K4 named weights:")
    k4_weights = []
    for doc_id, term in (("KB-0107.v1", "retry"), ("KB-0101.v1", "retry"), ("KB-0120.v1", "cache")):
        row = doc_ids.index(doc_id)
        t = tf_idf_weights.term_frequencies(documents[row]["text"])[term]
        i, w = idf_score[column[term]], W[row, column[term]]
        k4_weights.append({"doc_id": doc_id, "term": term, "tf": t, "idf": i, "w": w})
        print(f"  W[{doc_id}, {term}] = {t} x {i:.4f} = {w:.4f}")

    # 3. Document vector lengths
    print("")
    print("3. Document vector lengths: ||W[d]|| = sqrt(sum of squared entries over the k non-zero entries)")
    lengths = numpy.linalg.norm(W, axis=1)
    l2_lengths = numpy.linalg.norm(W_l2, axis=1)
    length_table = []
    for row, doc_id in enumerate(doc_ids):
        nonzero = numpy.flatnonzero(W[row])
        top = sorted(nonzero, key=lambda c: -W[row, c])[:3]
        top_terms = " + ".join(f"{W[row, c]:.4f}^2 [{vocabulary[c]}]" for c in top)
        print(f"  {doc_id}: sqrt({top_terms} + ... over {len(nonzero)} non-zero entries) = {lengths[row]:.4f}; after L2 the row length is {l2_lengths[row]:.4f}")
        length_table.append({
            "doc_id": doc_id,
            "non_zero_entries": int(len(nonzero)),
            "largest_entries": [{"term": vocabulary[c], "w": W[row, c]} for c in top],
            "length": lengths[row],
            "l2_row_length": l2_lengths[row],
        })

    # 4. wq1 and wq2: scores as sums of weights, raw and L2
    print("")
    print("4. Scores and ranked lists: score = sum of the document's weights over the query's terms")
    ranked_lists = {}
    for name, query in (("wq1", wq1), ("wq2", wq2)):
        print(f"{name} = {query!r}")
        ranked_lists[f"{name}_raw"] = ranked_list_lines(query, W, vocabulary, doc_ids, W, "raw W")
        ranked_lists[f"{name}_l2"] = ranked_list_lines(query, W_l2, vocabulary, doc_ids, W_l2, "L2-normalized W")
    for name, query in (("wq3", wq3), ("wq4", wq4)):
        ranked_lists[f"{name}_raw"] = tf_idf_weights.score_and_rank(query, W, vocabulary, doc_ids)
        ranked_lists[f"{name}_l2"] = tf_idf_weights.score_and_rank(query, W_l2, vocabulary, doc_ids)

    # 5. The unseen query
    print("")
    print(f"5. Unseen query {unseen_query!r}")
    unseen_terms = tf_idf_weights.analyze(unseen_query)
    for term in unseen_terms:
        status = f"in the vocabulary, df {df[term]}, idf {idf_score[column[term]]:.4f}" if term in column else "outside the vocabulary: no column, contributes 0"
        print(f"  {term!r}: {status}")
    unseen_raw = tf_idf_weights.score_and_rank(unseen_query, W, vocabulary, doc_ids)
    unseen_l2 = tf_idf_weights.score_and_rank(unseen_query, W_l2, vocabulary, doc_ids)
    kafka_only = tf_idf_weights.score_and_rank("kafka", W, vocabulary, doc_ids)
    print(f"  raw ranked list: {[(d, round(s, 4)) for d, s in unseen_raw]}")
    print(f"  L2 ranked list:  {[(d, round(s, 4)) for d, s in unseen_l2]}")
    print(f"  'kafka' alone:   {kafka_only} (no candidate, no exception)")

    # 6. Results file: the IDF, the two matrices and the eight ranked lists
    json_schema = {
        "Inputs": {
            "N": N,
            "doc_ids": doc_ids,
            "analyzer": tf_idf_weights.TERM_PATTERN,
            "idf": "ln((1 + N) / (1 + df)) + 1",
            "queries": {"wq1": wq1, "wq2": wq2, "wq3": wq3, "wq4": wq4, "unseen": unseen_query},
        },
        "Vocabulary": list(vocabulary),
        "IDF": {
            "by_df": {str(d): v for d, v in df_to_idf.items()},
            "by_term": dict(zip(vocabulary, idf_score)),
        },
        "Weight Matrix on wq1 Terms": wq1_table,
        "K4 Weights": k4_weights,
        "Vector Lengths": length_table,
        "Weight Matrix": {"shape": list(W.shape), "rows": doc_ids, "columns": list(vocabulary), "values": W.tolist()},
        "L2 Normalized Weight Matrix": {"shape": list(W_l2.shape), "rows": doc_ids, "columns": list(vocabulary), "values": W_l2.tolist()},
        "Ranked Lists": {name: [{"doc_id": d, "score": s} for d, s in ranked] for name, ranked in ranked_lists.items()},
        "Unseen Query": {
            "query": unseen_query,
            "raw": [{"doc_id": d, "score": s} for d, s in unseen_raw],
            "l2": [{"doc_id": d, "score": s} for d, s in unseen_l2],
            "kafka_only": kafka_only,
        },
    }

    path = REP_DIR / "results" / "F06A-L1.json"
    with open(path, "w") as f:
        json.dump(json_schema, f, indent=2)
    print("")
    print(f"6. Wrote {path.relative_to(REP_DIR.parent.parent)}")
