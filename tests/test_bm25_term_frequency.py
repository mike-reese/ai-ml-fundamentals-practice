import json
import math
from pathlib import Path

import pytest
import tantivy
import evalkit.tf_idf_weights as tf_idf_weights
import evalkit.bm25_term_frequency as bm25


REP_DIR = Path(__file__).resolve().parents[1] / "reps" / "F06A-L1_TF-IDF_Weights"


@pytest.fixture
def inputs():
    # Fixture F05-v1: five documents with arrival = 0, the copy in the F06A-L1 folder
    # (data/F05_v1/documents.json in the course repository, sha256 06128ab7...).
    documents = [
        doc for doc in json.loads((REP_DIR / "documents.json").read_text())["documents"]
        if doc["arrival"] == 0
    ]
    doc_ids = ["KB-0101.v1", "KB-0107.v1", "KB-0112.v1", "KB-0120.v1", "KB-0131.v1"]
    N = 5

    # Analyzer A1 and the index builder come from F06A-L1
    analyzer = tf_idf_weights.analyze
    vocabulary_size = 98
    postings_total = 154

    # IDF variant: Lucene's, used by Tantivy; natural logarithm, always positive
    idf_formula = lambda df, N: math.log(1 + (N - df + 0.5) / (df + 0.5))

    # Settings: Lucene's and Tantivy's defaults
    k1 = 1.2
    b = 0.75

    # Query terms: the distinct terms of the analyzed query, in sorted order
    query_terms = lambda query: sorted(set(analyzer(query)))

    # Tie-break rule: equal scores ordered by rising doc_id
    tie_break_key = lambda doc_id_score: (-doc_id_score[1], doc_id_score[0])

    # Tantivy's stored lengths: one byte, exact up to 40 terms; KB-0101.v1 41 -> 40; avgdl stays 37.0
    tantivy_dl = {"KB-0101.v1": 40, "KB-0107.v1": 39, "KB-0112.v1": 35, "KB-0120.v1": 44, "KB-0131.v1": 26}
    tantivy_avgdl = 37.0

    # Worked-example queries, from data/F05_v1/queries.json (sha256 d2574d07...)
    wq1 = "retry failed"
    wq2 = "TR-0042"
    wq3 = "wait seconds attempts"
    wq4 = "failed message"

    # Unseen queries: one term outside the vocabulary, only such a term, and a repeated term
    unseen_query = "cache kafka"
    kafka_only = "kafka"
    repeated_query = "retry retry failed"

    return (
        documents, doc_ids, N, analyzer, vocabulary_size, postings_total,
        idf_formula, k1, b, query_terms, tie_break_key,
        tantivy_dl, tantivy_avgdl,
        wq1, wq2, wq3, wq4, unseen_query, kafka_only, repeated_query,
    )


@pytest.fixture
def known_results():
    # Full double-precision values computed on this machine (Python 3.12, tantivy 0.26.2) on
    # 17 October 2026 from the formulas in the spec; each matches the spec's four-decimal value.

    # K1: document length in terms and the average
    known_dl = {"KB-0101.v1": 41, "KB-0107.v1": 39, "KB-0112.v1": 35, "KB-0120.v1": 44, "KB-0131.v1": 26}
    known_avgdl = 37.0

    # K2: Lucene idf by df, with the number of vocabulary terms at each df; all positive
    known_idf_by_df = {1: 1.3862943611198906, 2: 0.8754687373538999, 3: 0.5389965007326871, 4: 0.28768207245178085, 5: 0.0870113769896297}
    known_terms_per_df = {1: 66, 2: 17, 3: 8, 4: 5, 5: 2}
    known_df5_terms = ["the", "to"]

    # K3: at tf = 1 per document: the denominator and the saturation-normalized term
    known_denominator_tf1 = {"KB-0101.v1": 2.2972972972972974, "KB-0107.v1": 2.2486486486486488, "KB-0112.v1": 2.1513513513513516, "KB-0120.v1": 2.3702702702702703, "KB-0131.v1": 1.9324324324324325}
    known_saturation_tf1 = {"KB-0101.v1": 0.9576470588235294, "KB-0107.v1": 0.9783653846153847, "KB-0112.v1": 1.0226130653266332, "KB-0120.v1": 0.9281641961231472, "KB-0131.v1": 1.1384615384615386}

    # K4: the trace of retry in KB-0107.v1, nine values in execution order
    known_trace = {
        "tf": 2, "df": 3, "argument": 1.7142857142857142, "idf": 0.5389965007326871, "dl": 39,
        "denominator": 3.2486486486486488, "saturation": 1.3544093178036607, "contribution": 0.7300218828559191,
    }

    # K5: wq1 contributions per candidate and the ranked list; KB-0112.v1 and KB-0120.v1 are absent
    known_wq1_contributions = {
        "KB-0107.v1": {"failed": 0.8565283079399935, "retry": 0.7300218828559191},
        "KB-0101.v1": {"failed": 0.8383900614189111, "retry": 0.5161684136428322},
        "KB-0131.v1": {"retry": 0.6136267854495209},
    }
    known_wq1 = {"KB-0107.v1": 1.5865501907959125, "KB-0101.v1": 1.3545584750617432, "KB-0131.v1": 0.6136267854495209}
    known_wq1_ranked = ["KB-0107.v1", "KB-0101.v1", "KB-0131.v1"]
    known_wq1_absent = ["KB-0112.v1", "KB-0120.v1"]

    # K6: wq2 contributions and scores; no tie
    known_wq2_contributions = {
        "KB-0131.v1": {"0042": 0.6136267854495209, "tr": 0.6136267854495209},
        "KB-0101.v1": {"0042": 0.5161684136428322, "tr": 0.5161684136428322},
        "KB-0120.v1": {"0042": 0.5002772538157438, "tr": 0.5002772538157438},
    }
    known_wq2 = {"KB-0131.v1": 1.2272535708990417, "KB-0101.v1": 1.0323368272856643, "KB-0120.v1": 1.0005545076314877}
    known_wq2_ranked = ["KB-0131.v1", "KB-0101.v1", "KB-0120.v1"]

    # K7: wq3 and wq4 contributions and scores
    known_wq3_contributions = {
        "KB-0101.v1": {"attempts": 1.3275807175901069, "seconds": 0.8383900614189111, "wait": 1.3275807175901069},
        "KB-0120.v1": {"seconds": 0.8125787368370291},
    }
    known_wq3 = {"KB-0101.v1": 3.4935514965991246, "KB-0120.v1": 0.8125787368370291}
    known_wq3_ranked = ["KB-0101.v1", "KB-0120.v1"]
    known_wq4_contributions = {
        "KB-0107.v1": {"failed": 0.8565283079399935, "message": 1.8776099999194527},
        "KB-0101.v1": {"failed": 0.8383900614189111},
    }
    known_wq4 = {"KB-0107.v1": 2.734138307859446, "KB-0101.v1": 0.8383900614189111}
    known_wq4_ranked = ["KB-0107.v1", "KB-0101.v1"]

    # K8: term-at-a-time on wq1: the accumulator after each posting list; postings read per query
    known_wq1_accumulator = [
        ("failed", {"KB-0101.v1": 0.8383900614189111, "KB-0107.v1": 0.8565283079399935}),
        ("retry", {"KB-0101.v1": 1.3545584750617432, "KB-0107.v1": 1.5865501907959125, "KB-0131.v1": 0.6136267854495209}),
    ]
    known_postings_read = {"wq1": 5, "wq2": 6, "wq3": 4, "wq4": 3}

    # K9: our scores with Tantivy's stored lengths, where KB-0101.v1 differs
    known_tantivy_length_scores = {
        "wq1": {"KB-0107.v1": 1.5865501907959125, "KB-0101.v1": 1.369054344592725, "KB-0131.v1": 0.6136267854495209},
        "wq2": {"KB-0131.v1": 1.2272535708990417, "KB-0101.v1": 1.0433844271020387, "KB-0120.v1": 1.0005545076314877},
        "wq3": {"KB-0101.v1": 3.530937897870697, "KB-0120.v1": 0.8125787368370291},
        "wq4": {"KB-0107.v1": 2.734138307859446, "KB-0101.v1": 0.8473621310417057},
    }
    known_tantivy_df_retry = 3

    # K10: the IDF variants on wq1
    known_okapi_raw = {"retry": -0.3364722366212129, "failed": 0.3364722366212129}
    known_okapi_negative_terms = 15
    known_okapi_mean = 0.6657939713545804
    known_okapi_floor = 0.1664484928386451
    known_fts5_floor = 1e-6
    known_wq1_okapi = {"KB-0107.v1": 0.5546321788293485, "KB-0101.v1": 0.4816205573886169, "KB-0131.v1": 0.1894952072316883}
    known_wq1_fts5 = {"KB-0107.v1": 0.3291941436036295, "KB-0101.v1": 0.32222260542313796, "KB-0131.v1": 1.1384615384615385e-06}

    # K11: unseen cases
    known_wq1_k1_zero = {"KB-0101.v1": 1.414465238086587, "KB-0107.v1": 1.414465238086587, "KB-0131.v1": 0.5389965007326871}
    known_wq1_k1_zero_ranked = ["KB-0101.v1", "KB-0107.v1", "KB-0131.v1"]
    known_unseen = {"KB-0120.v1": 2.271652964170289}
    known_unseen_ranked = ["KB-0120.v1"]
    known_unseen_postings_read = 1
    known_kafka_only_ranked = []
    known_kafka_only_postings_read = 0
    known_repeated_postings_read = 5

    return (
        known_dl, known_avgdl,
        known_idf_by_df, known_terms_per_df, known_df5_terms,
        known_denominator_tf1, known_saturation_tf1,
        known_trace,
        known_wq1_contributions, known_wq1, known_wq1_ranked, known_wq1_absent,
        known_wq2_contributions, known_wq2, known_wq2_ranked,
        known_wq3_contributions, known_wq3, known_wq3_ranked,
        known_wq4_contributions, known_wq4, known_wq4_ranked,
        known_wq1_accumulator, known_postings_read,
        known_tantivy_length_scores, known_tantivy_df_retry,
        known_okapi_raw, known_okapi_negative_terms, known_okapi_mean, known_okapi_floor, known_fts5_floor,
        known_wq1_okapi, known_wq1_fts5,
        known_wq1_k1_zero, known_wq1_k1_zero_ranked,
        known_unseen, known_unseen_ranked, known_unseen_postings_read,
        known_kafka_only_ranked, known_kafka_only_postings_read, known_repeated_postings_read,
    )


TOL = 1e-9
LIBRARY_TOL = 1e-5


def _build(documents):
    """Forward and inverted index from F06A-L1, then the lengths and the Lucene IDF table from the toolkit."""
    forward, inverted = tf_idf_weights.build_index(documents)
    dl, avgdl = bm25.document_lengths(forward)
    idf_table = bm25.build_idf_table(inverted, len(documents))
    return forward, inverted, dl, avgdl, idf_table


def _candidate_by_candidate(query_terms, inverted, idf_table, dl, avgdl, k1, b):
    """The same contributions summed per candidate, one candidate at a time: the sum written in the test."""
    contributions = {}
    for term in query_terms:
        for doc_id, tf in inverted.get(term, []):
            _, _, contribution = bm25.term_contribution(tf, dl[doc_id], avgdl, idf_table[term], k1, b)
            contributions.setdefault(doc_id, {})[term] = contribution
    scores = {doc_id: sum(per_term.values()) for doc_id, per_term in contributions.items()}
    return contributions, scores


def _assert_ranked(actual, known_scores, known_order):
    """Exact on order, 1e-9 on scores, against a dict of known scores and a known order."""
    assert [doc_id for doc_id, _ in actual] == known_order
    for doc_id, score in actual:
        assert abs(score - known_scores[doc_id]) <= TOL, (doc_id, score, known_scores[doc_id])


def _assert_contributions(actual, known):
    assert set(actual) == set(known)
    for doc_id in known:
        assert set(actual[doc_id]) == set(known[doc_id]), doc_id
        for term, value in known[doc_id].items():
            assert abs(actual[doc_id][term] - value) <= TOL, (doc_id, term)


def _tantivy_index(documents):
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
    return index, index.searcher()


def _tantivy_hits(index, searcher, query_terms, tie_break_key):
    """Run the query as an OR query over body; return (doc_id, score) hits sorted by the tie-break rule."""
    query = index.parse_query(" OR ".join(query_terms), ["body"])
    hits = [(searcher.doc(address)["doc_id"][0], float(score)) for score, address in searcher.search(query, 10).hits]
    return sorted(hits, key=tie_break_key)


# 1. Known values

def test_known_values(inputs, known_results):
    (
        documents, doc_ids, N, analyzer, vocabulary_size, postings_total,
        idf_formula, k1, b, query_terms, tie_break_key,
        tantivy_dl, tantivy_avgdl,
        wq1, wq2, wq3, wq4, unseen_query, kafka_only, repeated_query,
    ) = inputs
    (
        known_dl, known_avgdl,
        known_idf_by_df, known_terms_per_df, known_df5_terms,
        known_denominator_tf1, known_saturation_tf1,
        known_trace,
        known_wq1_contributions, known_wq1, known_wq1_ranked, known_wq1_absent,
        known_wq2_contributions, known_wq2, known_wq2_ranked,
        known_wq3_contributions, known_wq3, known_wq3_ranked,
        known_wq4_contributions, known_wq4, known_wq4_ranked,
        known_wq1_accumulator, known_postings_read,
    ) = known_results[:23]

    forward, inverted, dl, avgdl, idf_table = _build(documents)
    assert len(inverted) == vocabulary_size
    assert sum(len(postings) for postings in inverted.values()) == postings_total

    # dl per document (exact), avgdl (1e-9)
    assert dict(dl) == known_dl
    assert list(dl) == doc_ids
    assert abs(avgdl - known_avgdl) <= TOL

    # idf for df 1 to 5 (1e-9), the number of terms at each df (exact); all positive
    df = {term: len(postings) for term, postings in inverted.items()}
    assert set(idf_table) == set(inverted)
    for d in range(1, N + 1):
        terms_at_df = sorted(term for term in inverted if df[term] == d)
        assert len(terms_at_df) == known_terms_per_df[d], d
        assert abs(bm25.lucene_idf(d, N) - known_idf_by_df[d]) <= TOL, d
        assert abs(bm25.lucene_idf(d, N) - idf_formula(d, N)) <= TOL, d
        for term in terms_at_df:
            assert abs(idf_table[term] - known_idf_by_df[d]) <= TOL, (term, d)
    assert sorted(term for term in inverted if df[term] == N) == known_df5_terms
    assert min(idf_table.values()) > 0

    # denominator and saturation-normalized term at tf = 1, per document (1e-9)
    for doc_id in doc_ids:
        denominator, saturation, _ = bm25.term_contribution(1, dl[doc_id], avgdl, 1.0, k1, b)
        assert abs(denominator - known_denominator_tf1[doc_id]) <= TOL, doc_id
        assert abs(saturation - known_saturation_tf1[doc_id]) <= TOL, doc_id

    # the nine values of the retry in KB-0107.v1 trace (exact on tf, df, dl; 1e-9 on the rest)
    tf = forward["KB-0107.v1"]["retry"]
    df_retry = len(inverted["retry"])
    argument = 1 + (N - df_retry + 0.5) / (df_retry + 0.5)
    idf_retry = idf_table["retry"]
    dl_0107 = dl["KB-0107.v1"]
    denominator, saturation, contribution = bm25.term_contribution(tf, dl_0107, avgdl, idf_retry, k1, b)
    assert tf == known_trace["tf"]
    assert df_retry == known_trace["df"]
    assert abs(argument - known_trace["argument"]) <= TOL
    assert abs(idf_retry - known_trace["idf"]) <= TOL
    assert abs(math.log(argument) - idf_retry) <= TOL
    assert dl_0107 == known_trace["dl"]
    assert abs(denominator - known_trace["denominator"]) <= TOL
    assert abs(saturation - known_trace["saturation"]) <= TOL
    assert abs(contribution - known_trace["contribution"]) <= TOL

    # wq1: order, scores and each candidate's contributions (exact order, 1e-9 values)
    trace = []
    wq1_ranked, wq1_read = bm25.score_term_at_a_time(wq1, inverted, idf_table, dl, avgdl, k1, b, trace=trace)
    _assert_ranked(wq1_ranked, known_wq1, known_wq1_ranked)
    wq1_contributions, wq1_sums = _candidate_by_candidate(query_terms(wq1), inverted, idf_table, dl, avgdl, k1, b)
    _assert_contributions(wq1_contributions, known_wq1_contributions)
    for doc_id in known_wq1_absent:
        assert doc_id not in dict(wq1_ranked)

    # wq2: three distinct scores in the order of K6
    wq2_ranked, wq2_read = bm25.score_term_at_a_time(wq2, inverted, idf_table, dl, avgdl, k1, b)
    _assert_ranked(wq2_ranked, known_wq2, known_wq2_ranked)
    wq2_scores = [score for _, score in wq2_ranked]
    assert len(wq2_scores) == 3
    for a, c in zip(wq2_scores, wq2_scores[1:]):
        assert a - c > TOL
    wq2_contributions, wq2_sums = _candidate_by_candidate(query_terms(wq2), inverted, idf_table, dl, avgdl, k1, b)
    _assert_contributions(wq2_contributions, known_wq2_contributions)

    # wq3, wq4
    wq3_ranked, wq3_read = bm25.score_term_at_a_time(wq3, inverted, idf_table, dl, avgdl, k1, b)
    _assert_ranked(wq3_ranked, known_wq3, known_wq3_ranked)
    wq3_contributions, wq3_sums = _candidate_by_candidate(query_terms(wq3), inverted, idf_table, dl, avgdl, k1, b)
    _assert_contributions(wq3_contributions, known_wq3_contributions)
    wq4_ranked, wq4_read = bm25.score_term_at_a_time(wq4, inverted, idf_table, dl, avgdl, k1, b)
    _assert_ranked(wq4_ranked, known_wq4, known_wq4_ranked)
    wq4_contributions, wq4_sums = _candidate_by_candidate(query_terms(wq4), inverted, idf_table, dl, avgdl, k1, b)
    _assert_contributions(wq4_contributions, known_wq4_contributions)

    # postings read per query (exact); the accumulator after each posting list of wq1 (1e-9)
    assert (wq1_read, wq2_read, wq3_read, wq4_read) == tuple(known_postings_read[name] for name in ("wq1", "wq2", "wq3", "wq4"))
    assert [term for term, _, _ in trace] == [term for term, _ in known_wq1_accumulator]
    assert [n for _, n, _ in trace] == [2, 3]
    for (_, _, accumulator), (_, known_accumulator) in zip(trace, known_wq1_accumulator):
        assert set(accumulator) == set(known_accumulator)
        for doc_id, value in known_accumulator.items():
            assert abs(accumulator[doc_id] - value) <= TOL, doc_id

    # term-at-a-time against the candidate-by-candidate sum of the same contributions (1e-9)
    for ranked, sums in ((wq1_ranked, wq1_sums), (wq2_ranked, wq2_sums), (wq3_ranked, wq3_sums), (wq4_ranked, wq4_sums)):
        assert set(dict(ranked)) == set(sums)
        for doc_id, score in ranked:
            assert abs(score - sums[doc_id]) <= TOL, doc_id


# 2. Against the library: Tantivy's BM25

def test_against_library(inputs, known_results):
    (
        documents, doc_ids, N, analyzer, vocabulary_size, postings_total,
        idf_formula, k1, b, query_terms, tie_break_key,
        tantivy_dl, tantivy_avgdl,
        wq1, wq2, wq3, wq4, unseen_query, kafka_only, repeated_query,
    ) = inputs
    known_tantivy_length_scores, known_tantivy_df_retry = known_results[23:25]

    forward, inverted, dl, avgdl, idf_table = _build(documents)
    index, searcher = _tantivy_index(documents)

    # df of retry from our posting list against the library (exact)
    assert len(inverted["retry"]) == searcher.doc_freq("body", "retry") == known_tantivy_df_retry

    largest_stored, largest_exact = 0.0, (0.0, None, None)
    lists_checked = 0
    for name, query in (("wq1", wq1), ("wq2", wq2), ("wq3", wq3), ("wq4", wq4)):
        terms = query_terms(query)
        hits = _tantivy_hits(index, searcher, terms, tie_break_key)
        exact, _ = bm25.score_term_at_a_time(query, inverted, idf_table, dl, avgdl, k1, b)
        stored, _ = bm25.score_term_at_a_time(query, inverted, idf_table, tantivy_dl, tantivy_avgdl, k1, b)

        # candidate sets (exact)
        assert {doc_id for doc_id, _ in hits} == {doc_id for doc_id, _ in exact} == {doc_id for doc_id, _ in stored}

        # scores with Tantivy's stored lengths, every candidate (1e-5); K9 (1e-9)
        stored_scores = dict(stored)
        for doc_id, hit_score in hits:
            assert abs(hit_score - stored_scores[doc_id]) <= LIBRARY_TOL, (name, doc_id)
            largest_stored = max(largest_stored, abs(hit_score - stored_scores[doc_id]))
        _assert_ranked(stored, known_tantivy_length_scores[name], [doc_id for doc_id, _ in hits])

        # ranked lists with the tie-break rule applied to both (exact)
        assert [doc_id for doc_id, _ in hits] == [doc_id for doc_id, _ in stored] == [doc_id for doc_id, _ in exact]
        lists_checked += 1

        # reported, not asserted: the exact-length scores against the hits
        exact_scores = dict(exact)
        for doc_id, hit_score in hits:
            gap = abs(hit_score - exact_scores[doc_id])
            if gap > largest_exact[0]:
                largest_exact = (gap, name, doc_id)
    assert lists_checked == 4
    print(f"\nlargest |tantivy - ours| with Tantivy's stored lengths (KB-0101.v1 41 -> 40): {largest_stored:.1e}")
    print(f"largest |tantivy - ours| with the exact lengths: {largest_exact[0]:.1e} ({largest_exact[1]}, {largest_exact[2]})")


# 3. The hazard: the IDF variant

def test_hazard(inputs, known_results):
    (
        documents, doc_ids, N, analyzer, vocabulary_size, postings_total,
        idf_formula, k1, b, query_terms, tie_break_key,
        tantivy_dl, tantivy_avgdl,
        wq1, wq2, wq3, wq4, unseen_query, kafka_only, repeated_query,
    ) = inputs
    known_wq1, known_wq1_ranked = known_results[9], known_results[10]
    (
        known_okapi_raw, known_okapi_negative_terms, known_okapi_mean, known_okapi_floor, known_fts5_floor,
        known_wq1_okapi, known_wq1_fts5,
    ) = known_results[25:32]

    forward, inverted, dl, avgdl, idf_table = _build(documents)

    # raw Okapi idf over the vocabulary: retry and failed (1e-9); negative terms (exact)
    okapi_raw = {term: math.log((N - len(postings) + 0.5) / (len(postings) + 0.5)) for term, postings in inverted.items()}
    assert len(okapi_raw) == vocabulary_size
    for term, value in known_okapi_raw.items():
        assert abs(okapi_raw[term] - value) <= TOL, term
    assert sum(value < 0 for value in okapi_raw.values()) == known_okapi_negative_terms

    # the mean of the raw values and BM25Okapi's floor (1e-9)
    okapi_mean = sum(okapi_raw.values()) / len(okapi_raw)
    okapi_floor = 0.25 * okapi_mean
    assert abs(okapi_mean - known_okapi_mean) <= TOL
    assert abs(okapi_floor - known_okapi_floor) <= TOL

    # the two floored tables: BM25Okapi replaces negatives, FTS5 replaces values <= 0
    okapi_table = {term: value if value >= 0 else okapi_floor for term, value in okapi_raw.items()}
    fts5_table = {term: value if value > 0 else known_fts5_floor for term, value in okapi_raw.items()}

    lucene, _ = bm25.score_term_at_a_time(wq1, inverted, idf_table, dl, avgdl, k1, b)
    okapi, _ = bm25.score_term_at_a_time(wq1, inverted, okapi_table, dl, avgdl, k1, b)
    fts5, _ = bm25.score_term_at_a_time(wq1, inverted, fts5_table, dl, avgdl, k1, b)

    # wq1 scores under each table (1e-9); the same order under all three (exact)
    _assert_ranked(lucene, known_wq1, known_wq1_ranked)
    _assert_ranked(okapi, known_wq1_okapi, known_wq1_ranked)
    _assert_ranked(fts5, known_wq1_fts5, known_wq1_ranked)

    # reported, not asserted: the Lucene score over the FTS5-variant score for KB-0107.v1
    ratio = dict(lucene)["KB-0107.v1"] / dict(fts5)["KB-0107.v1"]
    print(f"\nwq1 Lucene:        {[(d, round(s, 4)) for d, s in lucene]}")
    print(f"wq1 Okapi, BM25Okapi floor {okapi_floor:.4f}: {[(d, round(s, 4)) for d, s in okapi]}")
    print(f"wq1 Okapi, FTS5 floor 1e-6:  {[(d, round(s, 7)) for d, s in fts5]}")
    print(f"KB-0107.v1 Lucene / FTS5 ratio: {ratio:.4f}")


# 4. Unseen cases: a setting at its end and terms outside the index

def test_unseen_cases(inputs, known_results):
    (
        documents, doc_ids, N, analyzer, vocabulary_size, postings_total,
        idf_formula, k1, b, query_terms, tie_break_key,
        tantivy_dl, tantivy_avgdl,
        wq1, wq2, wq3, wq4, unseen_query, kafka_only, repeated_query,
    ) = inputs
    known_wq1, known_wq1_ranked, known_postings_read = known_results[9], known_results[10], known_results[22]
    (
        known_wq1_k1_zero, known_wq1_k1_zero_ranked,
        known_unseen, known_unseen_ranked, known_unseen_postings_read,
        known_kafka_only_ranked, known_kafka_only_postings_read, known_repeated_postings_read,
    ) = known_results[32:40]

    forward, inverted, dl, avgdl, idf_table = _build(documents)

    # k1 = 0: every contribution equals its term's idf; the two-term documents tie and doc_id orders them
    for doc_id in doc_ids:
        for tf in (1, 2, 4):
            _, saturation, contribution = bm25.term_contribution(tf, dl[doc_id], avgdl, idf_table["retry"], 0.0, b)
            assert abs(saturation - 1.0) <= TOL
            assert abs(contribution - idf_table["retry"]) <= TOL
    k1_zero, _ = bm25.score_term_at_a_time(wq1, inverted, idf_table, dl, avgdl, 0.0, b)
    _assert_ranked(k1_zero, known_wq1_k1_zero, known_wq1_k1_zero_ranked)
    assert abs(dict(k1_zero)["KB-0101.v1"] - dict(k1_zero)["KB-0107.v1"]) <= TOL
    assert abs(dict(k1_zero)["KB-0101.v1"] - (idf_table["failed"] + idf_table["retry"])) <= TOL

    # cache kafka: kafka has no posting list; one candidate, 1 posting read
    assert "cache" in inverted
    assert "kafka" not in inverted
    unseen, unseen_read = bm25.score_term_at_a_time(unseen_query, inverted, idf_table, dl, avgdl, k1, b)
    _assert_ranked(unseen, known_unseen, known_unseen_ranked)
    assert unseen_read == known_unseen_postings_read

    # kafka alone: an empty list, 0 postings read, no exception
    kafka, kafka_read = bm25.score_term_at_a_time(kafka_only, inverted, idf_table, dl, avgdl, k1, b)
    assert kafka == known_kafka_only_ranked
    assert kafka_read == known_kafka_only_postings_read

    # retry retry failed: one distinct term; equal to wq1's list, 5 postings read
    repeated, repeated_read = bm25.score_term_at_a_time(repeated_query, inverted, idf_table, dl, avgdl, k1, b)
    _assert_ranked(repeated, known_wq1, known_wq1_ranked)
    assert repeated_read == known_repeated_postings_read == known_postings_read["wq1"]
    assert query_terms(repeated_query) == query_terms(wq1)
