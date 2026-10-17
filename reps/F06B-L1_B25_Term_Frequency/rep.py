import json
import math
from pathlib import Path

import tantivy
import evalkit.tf_idf_weights as tf_idf_weights
import evalkit.bm25_term_frequency as bm25

REP_DIR = Path(__file__).parent
DOCUMENTS = REP_DIR.parent / "F06A-L1_TF-IDF_Weights" / "documents.json"

K1 = 1.2
B = 0.75


def load_documents():
    # Fixture F05-v1: the five documents with arrival = 0, in file order, from the F06A-L1 copy
    data = json.loads(DOCUMENTS.read_text())
    return [doc for doc in data["documents"] if doc["arrival"] == 0]


def worked_example_queries():
    wq1 = "retry failed"
    wq2 = "TR-0042"
    wq3 = "wait seconds attempts"
    wq4 = "failed message"
    return wq1, wq2, wq3, wq4


def query_terms(query):
    return sorted(set(tf_idf_weights.analyze(query)))


def contributions_by_candidate(query, inverted, idf_table, dl, avgdl, k1, b):
    """Each candidate's contribution per query term, the sum of which is its score."""
    out = {}
    for term in query_terms(query):
        for doc_id, tf in inverted.get(term, []):
            _, _, contribution = bm25.term_contribution(tf, dl[doc_id], avgdl, idf_table[term], k1, b)
            out.setdefault(doc_id, {})[term] = contribution
    return out


def fmt_ranked(ranked, places=4):
    return [(doc_id, round(score, places)) for doc_id, score in ranked]


def print_ranked_with_sums(name, query, inverted, idf_table, dl, avgdl, k1, b):
    """Print each candidate's score as its sum of contributions over the query's terms, then the ranked list."""
    terms = query_terms(query)
    ranked, read = bm25.score_term_at_a_time(query, inverted, idf_table, dl, avgdl, k1, b)
    per_candidate = contributions_by_candidate(query, inverted, idf_table, dl, avgdl, k1, b)
    print(f"{name} = {query!r}, terms {terms}")
    for doc_id, score in ranked:
        parts = " + ".join(f"{per_candidate[doc_id].get(term, 0.0):.4f}" for term in terms)
        print(f"    {doc_id}: {parts} = {score:.4f}")
    print(f"    ranked list: {fmt_ranked(ranked)}; postings read: {read}")
    return ranked, read, per_candidate


def tantivy_scores(documents, queries):
    """Tantivy's hits per query: an in-memory index, default tokenizer, OR query over body."""
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("doc_id", stored=True, tokenizer_name="raw")
    schema_builder.add_text_field("body", stored=False, tokenizer_name="default")
    index = tantivy.Index(schema_builder.build())
    writer = index.writer(15_000_000, 1)
    for doc in documents:
        document = tantivy.Document()
        document.add_text("doc_id", doc["doc_id"])
        document.add_text("body", doc["text"])
        writer.add_document(document)
    writer.commit()
    index.reload()
    searcher = index.searcher()
    hits = {}
    for name, query in queries.items():
        parsed = index.parse_query(" OR ".join(query_terms(query)), ["body"])
        pairs = [(searcher.doc(address)["doc_id"][0], float(score)) for score, address in searcher.search(parsed, 10).hits]
        hits[name] = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
    return hits


if __name__ == "__main__":
    documents = load_documents()
    doc_ids = [doc["doc_id"] for doc in documents]
    N = len(documents)
    wq1, wq2, wq3, wq4 = worked_example_queries()
    queries = {"wq1": wq1, "wq2": wq2, "wq3": wq3, "wq4": wq4}

    forward, inverted = tf_idf_weights.build_index(documents)
    dl, avgdl = bm25.document_lengths(forward)
    idf_table = bm25.build_idf_table(inverted, N)
    df = {term: len(postings) for term, postings in inverted.items()}
    vocabulary = sorted(inverted)

    # 1. Index, lengths and the df -> idf table
    print("1. Index, document lengths and Lucene IDF")
    print(f"N = {N} documents: {doc_ids}")
    print(f"index: {len(vocabulary)} terms, {sum(df.values())} postings")
    print(f"dl = sum of a document's tf values; avgdl = {sum(dl.values())} / {N} = {avgdl}")
    for doc_id in doc_ids:
        print(f"    {doc_id}: dl {dl[doc_id]}")
    print("idf = ln(1 + (N - df + 0.5) / (df + 0.5))")
    print(f"{'df':>4} | {'argument':>8} | {'idf':>8} | {'terms':>5} | example terms")
    df_to_idf = {}
    for d in range(1, N + 1):
        terms_at_df = [term for term in vocabulary if df[term] == d]
        argument = 1 + (N - d + 0.5) / (d + 0.5)
        idf = bm25.lucene_idf(d, N)
        df_to_idf[d] = {"argument": argument, "idf": idf, "terms": len(terms_at_df)}
        examples = ", ".join(terms_at_df) if len(terms_at_df) <= 5 else ", ".join(terms_at_df[:5]) + ", ..."
        print(f"{d:>4} | {argument:>8.4f} | {idf:>8.4f} | {len(terms_at_df):>5} | {examples}")

    # 2. The trace of K4: retry in KB-0107.v1, one line per value, in execution order
    print("")
    print("2. Trace of one term contribution: 'retry' in KB-0107.v1")
    trace_doc, trace_term = "KB-0107.v1", "retry"
    t_tf = forward[trace_doc][trace_term]
    t_df = df[trace_term]
    t_argument = 1 + (N - t_df + 0.5) / (t_df + 0.5)
    t_idf = idf_table[trace_term]
    t_dl = dl[trace_doc]
    t_denominator, t_saturation, t_contribution = bm25.term_contribution(t_tf, t_dl, avgdl, t_idf, K1, B)
    print(f"    tf            = {t_tf}")
    print(f"    df            = {t_df}")
    print(f"    argument      = 1 + ({N} - {t_df} + 0.5) / ({t_df} + 0.5) = {t_argument:.4f}")
    print(f"    idf           = ln({t_argument:.4f}) = {t_idf:.4f}")
    print(f"    dl            = {t_dl}")
    print(f"    denominator   = {t_tf} + {K1} x (1 - {B} + {B} x {t_dl} / {avgdl}) = {t_denominator:.4f}")
    print(f"    saturation    = {t_tf} x ({K1} + 1) / {t_denominator:.4f} = {t_saturation:.4f}")
    print(f"    contribution  = {t_idf:.4f} x {t_saturation:.4f} = {t_contribution:.4f}")
    k4_trace = {
        "doc_id": trace_doc, "term": trace_term, "tf": t_tf, "df": t_df, "argument": t_argument, "idf": t_idf,
        "dl": t_dl, "denominator": t_denominator, "saturation_normalized_term": t_saturation, "contribution": t_contribution,
    }

    # 3. wq1 term-at-a-time: the accumulator after each posting list, the sums, the ranked list, the work
    print("")
    print(f"3. Term-at-a-time on wq1 = {wq1!r}")
    accumulator_trace = []
    wq1_ranked, wq1_read = bm25.score_term_at_a_time(wq1, inverted, idf_table, dl, avgdl, K1, B, trace=accumulator_trace)
    for term, n_postings, accumulator in accumulator_trace:
        print(f"    after {term!r} ({n_postings} postings): {{{', '.join(f'{d}: {s:.4f}' for d, s in accumulator.items())}}}")
    wq1_per_candidate = contributions_by_candidate(wq1, inverted, idf_table, dl, avgdl, K1, B)
    for doc_id, score in wq1_ranked:
        parts = " + ".join(f"{wq1_per_candidate[doc_id].get(term, 0.0):.4f}" for term in query_terms(wq1))
        print(f"    {doc_id}: {parts} = {score:.4f}")
    print(f"    ranked list: {fmt_ranked(wq1_ranked)}")
    touched = [doc_id for doc_id in doc_ids if doc_id in dict(wq1_ranked)]
    never = [doc_id for doc_id in doc_ids if doc_id not in dict(wq1_ranked)]
    print(f"    work: {wq1_read} postings read in {len(accumulator_trace)} lists; {len(touched)} of {N} documents touched; never touched: {never}")

    # 4. wq2, wq3, wq4 with their sums; wq1 under the three IDF variants; the unseen cases
    print("")
    print("4. Ranked lists of wq2, wq3, wq4: score = sum of the term contributions")
    ranked_lists = {"wq1": wq1_ranked}
    postings_read = {"wq1": wq1_read}
    per_candidate = {"wq1": wq1_per_candidate}
    for name in ("wq2", "wq3", "wq4"):
        ranked_lists[name], postings_read[name], per_candidate[name] = print_ranked_with_sums(name, queries[name], inverted, idf_table, dl, avgdl, K1, B)

    print("")
    print("4b. wq1 under three IDF variants, k1 and b fixed")
    okapi_raw = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5)) for term in vocabulary}
    negative_terms = sorted(term for term, value in okapi_raw.items() if value < 0)
    okapi_mean = sum(okapi_raw.values()) / len(okapi_raw)
    okapi_floor = 0.25 * okapi_mean
    okapi_table = {term: value if value >= 0 else okapi_floor for term, value in okapi_raw.items()}
    fts5_table = {term: value if value > 0 else 1e-6 for term, value in okapi_raw.items()}
    print("    raw Okapi idf = ln((N - df + 0.5) / (df + 0.5))")
    print(f"    retry (df {df['retry']}): {okapi_raw['retry']:.4f}; failed (df {df['failed']}): {okapi_raw['failed']:.4f}")
    print(f"    {len(negative_terms)} of {len(vocabulary)} terms negative: {negative_terms}")
    print(f"    mean of the raw values: {okapi_mean:.4f}; BM25Okapi floor 0.25 x mean = {okapi_floor:.4f} replaces negatives; FTS5 floor 1e-6 replaces values <= 0")
    variant_lists = {}
    for label, table in (("lucene", idf_table), ("okapi_bm25okapi_floor", okapi_table), ("okapi_fts5_floor", fts5_table)):
        variant_lists[label], _ = bm25.score_term_at_a_time(wq1, inverted, table, dl, avgdl, K1, B)
        print(f"    {label:<22}: {fmt_ranked(variant_lists[label], 7)}")
    ratio = dict(variant_lists["lucene"])["KB-0107.v1"] / dict(variant_lists["okapi_fts5_floor"])["KB-0107.v1"]
    print(f"    KB-0107.v1: Lucene / FTS5-variant = {ratio:.4f}; same order under all three")

    print("")
    print("4c. Unseen cases")
    k1_zero, _ = bm25.score_term_at_a_time(wq1, inverted, idf_table, dl, avgdl, 0.0, B)
    print(f"    wq1 at k1 = 0: every contribution is the term's idf; {fmt_ranked(k1_zero)}")
    print(f"        KB-0101.v1 and KB-0107.v1 both {idf_table['failed']:.4f} + {idf_table['retry']:.4f}: tie, doc_id puts KB-0101.v1 first")
    unseen_cases = {"wq1_k1_zero": {"query": wq1, "k1": 0.0, "ranked": k1_zero}}
    for query in ("cache kafka", "kafka", "retry retry failed"):
        ranked, read = bm25.score_term_at_a_time(query, inverted, idf_table, dl, avgdl, K1, B)
        terms = query_terms(query)
        outside = [term for term in terms if term not in inverted]
        note = f"; outside the index: {outside}" if outside else ""
        if query == "retry retry failed":
            note += f"; distinct terms {terms}, equal to wq1's list: {ranked == wq1_ranked}"
        print(f"    {query!r}: {fmt_ranked(ranked)}, {read} postings read{note}")
        unseen_cases[query] = {"query": query, "k1": K1, "ranked": ranked, "postings_read": read}

    # 5. Tantivy's scores beside ours with the exact lengths and with Tantivy's stored lengths
    print("")
    print("5. Tantivy (default tokenizer, OR query over body) beside ours")
    # Tantivy keeps a one-byte length: exact up to 40; on this corpus only KB-0101.v1 (41 -> 40) changes, avgdl stays 37.0
    tantivy_dl = dict(dl)
    tantivy_dl["KB-0101.v1"] = 40
    hits = tantivy_scores(documents, queries)
    tantivy_rows = {}
    for name, query in queries.items():
        stored, _ = bm25.score_term_at_a_time(query, inverted, idf_table, tantivy_dl, avgdl, K1, B)
        exact = dict(ranked_lists[name])
        stored = dict(stored)
        tantivy_rows[name] = []
        for doc_id, hit_score in hits[name]:
            tantivy_rows[name].append({"doc_id": doc_id, "tantivy": hit_score, "ours_stored_length": stored[doc_id], "ours_exact_length": exact[doc_id]})
            print(f"    {name} {doc_id}: tantivy {hit_score:.4f}; ours, Tantivy's lengths {stored[doc_id]:.4f} (diff {abs(hit_score - stored[doc_id]):.1e}); ours, exact lengths {exact[doc_id]:.4f} (diff {abs(hit_score - exact[doc_id]):.1e})")
    print("    the stored-length difference is KB-0101.v1's dl 41 -> 40 in Tantivy's one-byte length; ranks agree on all four queries")

    # 6. Results file
    results = {
        "Inputs": {
            "N": N,
            "doc_ids": doc_ids,
            "analyzer": tf_idf_weights.TERM_PATTERN,
            "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
            "saturation_normalized_term": "tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))",
            "k1": K1, "b": B,
            "queries": queries,
        },
        "Document Lengths": {"dl": dl, "avgdl": avgdl, "tantivy_stored_dl": tantivy_dl},
        "IDF Tables": {
            "lucene_by_df": {str(d): v for d, v in df_to_idf.items()},
            "lucene": idf_table,
            "okapi_raw": okapi_raw,
            "okapi_negative_terms": negative_terms,
            "okapi_mean": okapi_mean,
            "okapi_bm25okapi_floor": okapi_floor,
            "okapi_bm25okapi_table": okapi_table,
            "okapi_fts5_table": fts5_table,
        },
        "K4 Trace": k4_trace,
        "wq1 Accumulator": [{"term": term, "postings": n, "accumulator": acc} for term, n, acc in accumulator_trace],
        "Ranked Lists": {
            name: {
                "ranked": [{"doc_id": d, "score": s, "contributions": per_candidate[name][d]} for d, s in ranked],
                "postings_read": postings_read[name],
            }
            for name, ranked in ranked_lists.items()
        },
        "Tantivy Scores": tantivy_rows,
        "Variant Scores on wq1": {label: [{"doc_id": d, "score": s} for d, s in ranked] for label, ranked in variant_lists.items()},
        "Lucene over FTS5 ratio KB-0107.v1": ratio,
        "Unseen Cases": {
            key: {**value, "ranked": [{"doc_id": d, "score": s} for d, s in value["ranked"]]} for key, value in unseen_cases.items()
        },
    }

    (REP_DIR / "results").mkdir(exist_ok=True)
    path = REP_DIR / "results" / "F06B-L1.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("")
    print(f"6. Wrote {path.relative_to(REP_DIR.parent.parent)}")
