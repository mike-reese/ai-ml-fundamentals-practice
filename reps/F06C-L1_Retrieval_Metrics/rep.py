import json
import math
import warnings
from collections import Counter
from pathlib import Path

from ranx import Qrels, Run, evaluate
from sklearn.metrics import dcg_score, ndcg_score
import evalkit.tf_idf_weights as tf_idf_weights
import evalkit.bm25_term_frequency as bm25
import evalkit.retrieval_metrics as rm

REP_DIR = Path(__file__).parent
DOCUMENTS = REP_DIR.parent / "F06A-L1_TF-IDF_Weights" / "documents.json"
JUDGMENTS = REP_DIR / "relevance_judgments.json"

K1 = 1.2
B = 0.75
THRESHOLD = 2           # policy 1: relevant when grade >= 2
CONTRAST_THRESHOLD = 1
CUTOFFS = (1, 3, 5)
ANSWER_DOC_VQ1 = "KB-0101.v1"

RANX_METRICS = [
    "precision@1-l2", "precision@3-l2", "precision@5-l2",
    "recall@1-l2", "recall@3-l2", "recall@5-l2",
    "mrr-l2", "ndcg@3",
]
RANX_COLUMNS = dict(zip(RANX_METRICS, ("P@1", "P@3", "P@5", "R@1", "R@3", "R@5", "RR", "nDCG@3")))
COLUMNS = list(RANX_COLUMNS.values())


def load_documents():
    # Fixture F05-v1: the five documents with arrival = 0, in file order, from the F06A-L1 copy
    data = json.loads(DOCUMENTS.read_text())
    return [doc for doc in data["documents"] if doc["arrival"] == 0]


def validation_queries():
    # The eight validation queries of data/F05_v1/queries.json (sha256 d2574d07...)
    return {
        "vq1": "retry failed job",
        "vq2": "streaming importer retry",
        "vq3": "deploy reviewer",
        "vq4": "cache ttl",
        "vq5": "dead letter queue",
        "vq6": "staging tuesday",
        "vq7": "production deploy tuesday",
        "vq8": "wait between attempts",
    }


def load_judgment_rows():
    # Fixture F06-v1: 40 rows, one per (query, document) pair, reviewed 2026-10-16 with 0 changes
    return json.loads(JUDGMENTS.read_text())["judgments"]


def bm25_runs(documents, queries):
    """The BM25 (doc_id, score) list per query from the F06A-L1 index and the F06B-L1 scorer."""
    forward, inverted = tf_idf_weights.build_index(documents)
    dl, avgdl = bm25.document_lengths(forward)
    idf_table = bm25.build_idf_table(inverted, len(documents))
    runs = {}
    for query_id, text in queries.items():
        ranked, _ = bm25.score_term_at_a_time(text, inverted, idf_table, dl, avgdl, K1, B)
        if len({score for _, score in ranked}) != len(ranked):
            raise ValueError(f"{query_id}: tied BM25 scores; ranx would order them itself")
        runs[query_id] = ranked
    return runs


def three_lists(list_A):
    """STATE 13's lists for vq1: A as given, B reversed, C without the answer document in A's order."""
    return {"A": list(list_A), "B": list(reversed(list_A)), "C": [d for d in list_A if d != ANSWER_DOC_VQ1]}


def table_row(ranked, judgments, threshold):
    relevant = rm.relevant_set(judgments, threshold)
    return {
        "P@1": rm.precision_at_k(ranked, relevant, 1),
        "P@3": rm.precision_at_k(ranked, relevant, 3),
        "P@5": rm.precision_at_k(ranked, relevant, 5),
        "R@1": rm.recall_at_k(ranked, relevant, 1),
        "R@3": rm.recall_at_k(ranked, relevant, 3),
        "R@5": rm.recall_at_k(ranked, relevant, 5),
        "RR": rm.reciprocal_rank(ranked, relevant),
        "nDCG@3": rm.ndcg_at_k(ranked, judgments, 3),
    }


def fmt(value, places=4):
    return "undefined" if value is None else f"{value:.{places}f}"


def short(doc_id):
    # KB-0101.v1 -> 0101, for the K2 line
    return doc_id.split("-")[1].split(".")[0]


def print_dcg_trace(name, ranked, judgments, k):
    """One line per rank: gain, discount, quotient; then the sum."""
    terms = []
    for rank, doc_id in enumerate(ranked[:k], start=1):
        gain = judgments[doc_id]
        discount = math.log2(rank + 1)
        quotient = gain / discount
        terms.append({"rank": rank, "doc_id": doc_id, "gain": gain, "discount": discount, "quotient": quotient})
        print(f"    list {name} rank {rank}: {doc_id} gain {gain} / log2({rank + 1}) = {gain} / {discount:.4f} = {quotient:.4f}")
    dcg = rm.dcg_at_k(ranked, judgments, k)
    print(f"    list {name} DCG@{k} = {' + '.join(f'{t['quotient']:.4f}' for t in terms)} = {dcg:.4f}")
    return terms, dcg


def sklearn_encoding(ranked, doc_ids):
    """One descending integer per document: the list's entries first, then the others, lower still and distinct."""
    scores = {doc_id: len(doc_ids) - rank for rank, doc_id in enumerate(ranked)}
    for offset, doc_id in enumerate(d for d in doc_ids if d not in scores):
        scores[doc_id] = len(doc_ids) - len(ranked) - offset
    return [scores[doc_id] for doc_id in doc_ids]


if __name__ == "__main__":
    documents = load_documents()
    doc_ids = [doc["doc_id"] for doc in documents]
    N = len(documents)
    queries = validation_queries()
    query_ids = list(queries)
    rows = load_judgment_rows()

    # 1. Judgments, the eight BM25 lists, K1 and K2
    print("1. Judgments and the eight BM25 lists")
    judgments = rm.build_judgments(rows)
    runs = bm25_runs(documents, queries)
    lists = {query_id: [doc_id for doc_id, _ in ranked] for query_id, ranked in runs.items()}
    grade_counts = Counter(row["grade"] for row in rows)
    print(f"N = {N} documents: {doc_ids}")
    print(f"{len(rows)} judged pairs: " + ", ".join(f"{grade_counts[g]} x grade {g}" for g in (2, 1, 0)))
    relevant = {query_id: rm.relevant_set(judgments[query_id], THRESHOLD) for query_id in query_ids}
    print(f"relevant set at threshold {THRESHOLD} (policy 1):")
    for query_id in query_ids:
        print(f"    {query_id}: {sorted(relevant[query_id])}")
    print(f"    vq1 at threshold {CONTRAST_THRESHOLD}: {sorted(rm.relevant_set(judgments['vq1'], CONTRAST_THRESHOLD))}")
    unanswerable = [query_id for query_id in query_ids if not rm.relevant_set(judgments[query_id], 1)]
    answerable = [query_id for query_id in query_ids if query_id not in unanswerable]
    print(f"unanswerable (no grade >= 1, policy 2): {unanswerable}")
    print(f"BM25 lists (k1 = {K1}, b = {B}, Lucene IDF) as doc_id:grade:")
    for query_id in query_ids:
        pairs = ", ".join(f"{short(d)}:{judgments[query_id][d]}" for d in lists[query_id])
        print(f"    {query_id} ({queries[query_id]!r}): {pairs}")

    # 2. vq1: the counts and divisions of K3 at each cutoff, then K4 for contrast
    print("")
    print("2. vq1 precision@k and recall@k: counts and divisions")
    list_A = lists["vq1"]
    vq1_pr = {}
    for threshold in (THRESHOLD, CONTRAST_THRESHOLD):
        rel = rm.relevant_set(judgments["vq1"], threshold)
        print(f"    threshold {threshold}: relevant set {sorted(rel)} (size {len(rel)}); list {list_A} ({len(list_A)} entries)")
        vq1_pr[threshold] = {}
        for k in CUTOFFS:
            hits = sum(d in rel for d in list_A[:k])
            p = rm.precision_at_k(list_A, rel, k)
            r = rm.recall_at_k(list_A, rel, k)
            vq1_pr[threshold][k] = {"hits": hits, "precision": p, "recall": r}
            print(f"        k = {k}: {hits} relevant in top-{k}; precision {hits} / {k} = {p:.4f}; recall {hits} / {len(rel)} = {fmt(r)}")
    print("    precision@5 divides by 5 although the list has 3 entries; recall does not change past the list's end")

    # 3. Reciprocal rank per query, MRR as sum over count, and ranx's convention
    print("")
    print("3. Reciprocal rank and MRR")
    rr = {query_id: rm.reciprocal_rank(lists[query_id], relevant[query_id]) for query_id in query_ids}
    for query_id in query_ids:
        print(f"    {query_id}: {fmt(rr[query_id])}")
    mrr, mrr_count = rm.mean_defined(rr)
    defined_sum = sum(v for v in rr.values() if v is not None)
    mrr_ranx = sum(v or 0.0 for v in rr.values()) / len(query_ids)
    print(f"    MRR (policy 2) = {defined_sum:.4f} / {mrr_count} = {mrr:.4f} over {mrr_count} answerable queries")
    print(f"    ranx's convention, vq6 counted as 0 = {defined_sum:.4f} / {len(query_ids)} = {mrr_ranx:.4f}")

    # 4. DCG@3 traces of A and B, the ideal order and its sum, both ratios, then the three-list table
    print("")
    print("4. DCG@3 trace of vq1's lists A and B; ideal DCG@3; nDCG@3")
    lists_vq1 = three_lists(list_A)
    traces = {}
    dcg = {}
    for name in ("A", "B"):
        print(f"    list {name} = {lists_vq1[name]}, grades {[judgments['vq1'][d] for d in lists_vq1[name]]}")
        traces[name], dcg[name] = print_dcg_trace(name, lists_vq1[name], judgments["vq1"], 3)
    judged_sorted = sorted(judgments["vq1"].values(), reverse=True)
    ideal_terms = [g / math.log2(i + 1) for i, g in enumerate(judged_sorted[:3], start=1)]
    ideal_vq1 = rm.ideal_dcg_at_k(judgments["vq1"], 3)
    print(f"    ideal: judged grades {list(judgments['vq1'].values())} sorted {judged_sorted}, cut at 3: {judged_sorted[:3]}")
    print(f"    ideal DCG@3 = {' + '.join(f'{t:.4f}' for t in ideal_terms)} = {ideal_vq1:.4f}")
    ndcg = {name: rm.ndcg_at_k(lists_vq1[name], judgments["vq1"], 3) for name in lists_vq1}
    for name in ("A", "B"):
        print(f"    nDCG@3 list {name} = {dcg[name]:.4f} / {ideal_vq1:.4f} = {ndcg[name]:.4f}")
    burges = {d: 2 ** g - 1 for d, g in judgments["vq1"].items()}
    ndcg_B_burges = rm.ndcg_at_k(lists_vq1["B"], burges, 3)
    print(f"    with the gain 2^grade - 1, list B would be {ndcg_B_burges:.4f}")
    dcg["C"] = rm.dcg_at_k(lists_vq1["C"], judgments["vq1"], 3)
    print("")
    print("    the three lists of vq1 at k = 3")
    print(f"    {'list':<5} | {'documents':<40} | {'recall@3':>8} | {'RR':>6} | {'nDCG@3':>6}")
    three_table = {}
    for name, ranked in lists_vq1.items():
        row = {
            "recall@3": rm.recall_at_k(ranked, relevant["vq1"], 3),
            "RR": rm.reciprocal_rank(ranked, relevant["vq1"]),
            "DCG@3": dcg[name],
            "nDCG@3": ndcg[name],
        }
        three_table[name] = row
        print(f"    {name:<5} | {str(ranked):<40} | {row['recall@3']:>8.4f} | {row['RR']:>6.4f} | {row['nDCG@3']:>6.4f}")
    print(f"    list C: DCG@3 {dcg['C']:.4f} over the same ideal {ideal_vq1:.4f}")
    print("    A against B: the same set in the top-3, so recall@3 is equal; the answer moved from rank 1 to rank 3, so RR and nDCG@3 fell")
    print("    A against C: the answer is not on the list, so recall@3 fell to 0 and RR to 0; nDCG@3 fell because the ideal still counts the answer")

    # 5. Per-query table and the means with their counts (K10), then the unseen cases (K14)
    print("")
    print(f"5. Per-query table at threshold {THRESHOLD}")
    table = {query_id: table_row(lists[query_id], judgments[query_id], THRESHOLD) for query_id in query_ids}
    print(f"    {'query':<6} | " + " | ".join(f"{c:>9}" for c in COLUMNS))
    for query_id in query_ids:
        print(f"    {query_id:<6} | " + " | ".join(f"{fmt(table[query_id][c]):>9}" for c in COLUMNS))
    means = {}
    for column in COLUMNS:
        mean, count = rm.mean_defined({query_id: table[query_id][column] for query_id in query_ids})
        means[column] = {"mean": mean, "count": count}
    print("    means over the defined values: " + "; ".join(f"{c} {means[c]['mean']:.4f} over {means[c]['count']}" for c in COLUMNS))

    print("")
    print("5b. Unseen cases")
    empty = {
        "P@3": rm.precision_at_k([], relevant["vq1"], 3),
        "R@3": rm.recall_at_k([], relevant["vq1"], 3),
        "RR": rm.reciprocal_rank([], relevant["vq1"]),
        "nDCG@3": rm.ndcg_at_k([], judgments["vq1"], 3),
    }
    print("    vq1 with an empty list: " + ", ".join(f"{c} {fmt(v)}" for c, v in empty.items()) + "; none undefined (a system failure, not an unanswerable query)")
    rel_vq5_t1 = rm.relevant_set(judgments["vq5"], CONTRAST_THRESHOLD)
    non_candidates = sorted(rel_vq5_t1 - set(lists["vq5"]))
    recall5_vq5_t1 = rm.recall_at_k(lists["vq5"], rel_vq5_t1, 5)
    recall5_vq5_t2 = rm.recall_at_k(lists["vq5"], relevant["vq5"], 5)
    print(f"    vq5 at threshold {CONTRAST_THRESHOLD}: relevant set {sorted(rel_vq5_t1)} (size {len(rel_vq5_t1)}); never a candidate: {non_candidates}; "
          f"recall@5 = {len(rel_vq5_t1) - len(non_candidates)} / {len(rel_vq5_t1)} = {recall5_vq5_t1:.4f}; at threshold {THRESHOLD}: {recall5_vq5_t2:.4f}")
    recall5_t1 = {query_id: rm.recall_at_k(lists[query_id], rm.relevant_set(judgments[query_id], CONTRAST_THRESHOLD), 5) for query_id in query_ids}
    mean_recall5_t1, count_recall5_t1 = rm.mean_defined(recall5_t1)
    print(f"    recall@5 at threshold {CONTRAST_THRESHOLD}: " + ", ".join(f"{q} {fmt(v)}" for q, v in recall5_t1.items()))
    print(f"    mean recall@5 at threshold {CONTRAST_THRESHOLD} = {mean_recall5_t1:.4f} over {count_recall5_t1}")
    k10 = {
        "precision@10": rm.precision_at_k(list_A, relevant["vq1"], 10),
        "recall@10": rm.recall_at_k(list_A, relevant["vq1"], 10),
        "nDCG@10": rm.ndcg_at_k(list_A, judgments["vq1"], 10),
    }
    print("    vq1 list A at k = 10: " + ", ".join(f"{c} {fmt(v)}" for c, v in k10.items()) + " (the divisor of precision grows; nothing is added to recall or DCG)")

    # 6. The libraries beside ours, then the results file
    print("")
    print("6. ranx and scikit-learn beside ours")
    qrels = Qrels({query_id: {d: g for d, g in judgments[query_id].items() if g > 0} for query_id in query_ids})
    run = Run({query_id: dict(runs[query_id]) for query_id in query_ids})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numba cast warning from the precision kernel; the values are unchanged
        ranx_per_query = evaluate(qrels, run, RANX_METRICS, return_mean=False)
        ranx_means = evaluate(qrels, run, RANX_METRICS)
    ranx_table = {query_id: {} for query_id in query_ids}
    largest = 0.0
    for metric, values in ranx_per_query.items():
        column = RANX_COLUMNS[metric]
        for query_id, value in zip(query_ids, values):
            ranx_table[query_id][column] = float(value)
            if table[query_id][column] is not None:
                largest = max(largest, abs(table[query_id][column] - float(value)))
    print(f"    ranx per query: largest |ours - ranx| over the defined entries {largest:.1e}; vq6 under ranx: "
          + ", ".join(f"{c} {ranx_table['vq6'][c]:.1f}" for c in COLUMNS))
    ranx_mean_rows = {}
    for metric, column in RANX_COLUMNS.items():
        ours = means[column]
        library = float(ranx_means[metric])
        ranx_mean_rows[column] = {"ours": ours["mean"], "ours_count": ours["count"], "ranx": library, "ranx_count": len(query_ids), "difference": ours["mean"] - library}
    for column in ("RR", "R@3", "nDCG@3", "P@3"):
        r = ranx_mean_rows[column]
        print(f"    {column}: policy 2 mean {r['ours']:.4f} over {r['ours_count']}; ranx {r['ranx']:.4f} over {r['ranx_count']}; difference {r['difference']:.4f}"
              + (" (ranx scores vq6 as 0 and keeps it in the mean)" if r["ours_count"] < len(query_ids) else " (precision is defined for vq6 under both)"))

    y_true = [[judgments["vq1"][d] for d in doc_ids]]
    print(f"    scikit-learn on vq1: y_true {y_true[0]} (one grade per document in corpus order)")
    sklearn_rows = {}
    for name, ranked in lists_vq1.items():
        encoding = sklearn_encoding(ranked, doc_ids)
        lib_dcg = float(dcg_score(y_true, [encoding], k=3))
        lib_ndcg = float(ndcg_score(y_true, [encoding], k=3))
        sklearn_rows[name] = {"encoding": encoding, "dcg_score": lib_dcg, "ndcg_score": lib_ndcg, "ours_dcg": dcg[name], "ours_ndcg": ndcg[name]}
        note = "" if name != "C" else "  <- convention: scikit-learn ranks all five documents, a third one fills rank 3"
        print(f"    list {name} encoded {encoding}: dcg_score {lib_dcg:.4f} (ours {dcg[name]:.4f}, diff {abs(lib_dcg - dcg[name]):.1e}); "
              f"ndcg_score {lib_ndcg:.4f} (ours {ndcg[name]:.4f}, diff {abs(lib_ndcg - ndcg[name]):.1e}){note}")
    sklearn_tied = float(ndcg_score(y_true, [[1, 1, 0, 0, 1]], k=3))
    sklearn_all_zero = float(ndcg_score([[0] * N], [sklearn_encoding(lists_vq1["A"], doc_ids)], k=3))
    print(f"    tied scores [1, 1, 0, 0, 1]: ndcg_score {sklearn_tied:.4f} (gain averaged over the tie); all-zero y_true: {sklearn_all_zero:.1f} where ours is undefined")

    # the hazard, for the results file: the ideal from the list's own grades
    hazard = {}
    for name, ranked in {**lists_vq1, "vq8": lists["vq8"]}.items():
        j = judgments["vq1"] if name != "vq8" else judgments["vq8"]
        wrong_ideal = sum(g / math.log2(i + 1) for i, g in enumerate(sorted((j[d] for d in ranked), reverse=True)[:3], start=1))
        hazard[name] = {"wrong_ideal": wrong_ideal, "wrong_ndcg": rm.dcg_at_k(ranked, j, 3) / wrong_ideal, "right_ideal": rm.ideal_dcg_at_k(j, 3), "right_ndcg": rm.ndcg_at_k(ranked, j, 3)}
    print("    the hazard, ideal from the list's own grades: " + "; ".join(f"{n} wrong {h['wrong_ndcg']:.4f} right {h['right_ndcg']:.4f}" for n, h in hazard.items()))

    results = {
        "Inputs": {
            "N": N, "doc_ids": doc_ids, "queries": queries,
            "judgments_file": JUDGMENTS.name, "judged_pairs": len(rows), "grade_counts": {str(g): grade_counts[g] for g in (2, 1, 0)},
            "rule": "RJ-1: 2 holds the answer span, 1 about the task with no answer, 0 otherwise",
            "policy_1": f"relevant when grade >= {THRESHOLD}",
            "policy_2": "a query with no grade >= 1 has recall, RR and nDCG undefined and excluded from every mean; precision is defined",
            "bm25": {"k1": K1, "b": B, "idf": "lucene"},
            "cutoffs": list(CUTOFFS),
            "ndcg": "linear gain, log base 2, ideal from the judged grades",
        },
        "Judgments": judgments,
        "Relevant Sets": {
            f"threshold_{THRESHOLD}": {q: sorted(s) for q, s in relevant.items()},
            f"vq1_threshold_{CONTRAST_THRESHOLD}": sorted(rm.relevant_set(judgments["vq1"], CONTRAST_THRESHOLD)),
            "unanswerable": unanswerable,
            "answerable": answerable,
        },
        "Ranked Lists": {q: [{"doc_id": d, "score": s, "grade": judgments[q][d]} for d, s in runs[q]] for q in query_ids},
        "vq1 Precision and Recall": {str(t): {str(k): v for k, v in per_k.items()} for t, per_k in vq1_pr.items()},
        "Reciprocal Rank": {"per_query": rr, "mrr": mrr, "count": mrr_count, "ranx_convention_over_8": mrr_ranx},
        "vq1 DCG Trace": {
            "lists": lists_vq1,
            "dcg_terms": traces,
            "dcg": dcg,
            "ideal_grades_sorted": judged_sorted,
            "ideal_terms": ideal_terms,
            "ideal_dcg": ideal_vq1,
            "ndcg": ndcg,
            "ndcg_B_gain_2_pow_grade_minus_1": ndcg_B_burges,
            "three_lists": three_table,
        },
        "Per-Query Table": table,
        "Means": means,
        "Unseen Cases": {
            "vq1_empty_list": empty,
            "vq5_threshold_1": {"relevant_set": sorted(rel_vq5_t1), "never_a_candidate": non_candidates, "recall@5": recall5_vq5_t1, "recall@5_threshold_2": recall5_vq5_t2},
            "recall@5_threshold_1": {"per_query": recall5_t1, "mean": mean_recall5_t1, "count": count_recall5_t1},
            "vq1_list_A_k10": k10,
        },
        "Hazard": hazard,
        "ranx": {"per_query": ranx_table, "means_over_8": {RANX_COLUMNS[m]: float(v) for m, v in ranx_means.items()}, "means_beside_ours": ranx_mean_rows, "largest_per_query_difference": largest},
        "scikit-learn": {"y_true": y_true[0], "lists": sklearn_rows, "tied_scores_ndcg": sklearn_tied, "all_zero_y_true_ndcg": sklearn_all_zero},
    }

    (REP_DIR / "results").mkdir(exist_ok=True)
    path = REP_DIR / "results" / "F06C-L1.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("")
    print(f"7. Wrote {path.relative_to(REP_DIR.parent.parent)}")
