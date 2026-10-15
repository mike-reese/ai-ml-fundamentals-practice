"""Assemble the FEVER v1.0 data for experiment X1.

Written agentically by the tutor at Mike's request (owner decision 2026-10-13),
with a visual walkthrough in the course repo (notebooks/tests/X01_data_assembly.ipynb).
Every function here is importable by that notebook; the stages run in order:

    uv run python scripts/get_data.py            # all stages
    uv run python scripts/get_data.py download   # one stage

Stages
    download  the three claim files and wiki-pages.zip from fever.ai, each pinned by SHA-256
    claims    one row per claim: label, text, primary page (the group), gold evidence
    wiki      the Wikipedia pages cited by any claim, sentences parsed, from the zip
    split     training / validation / test by primary page
    evidence  for every claim, the top-k sentences of its primary page by TF-IDF cosine, fitted on the training set
    card      counts for the data card and the check tables

Everything lands in data/fever/, which git ignores. Nothing here reads a label of
the test set; the test set is only cut and counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- pins

# Claim files: the original FEVER v1.0 JSON-lines files from fever.ai, pinned by SHA-256.
# The Hugging Face conversion (fever/fever, parquet branch 5f577157) was abandoned on 2026-10-15:
# it maps evidence sentence 0 to -1 (95,529 training evidence lines lost, the most common gold
# sentence) and its Wikipedia pages are truncated at 3,741,000 of 5,416,537.
CLAIM_BASE_URL = "https://fever.ai/download/fever/"
CLAIM_FILES = {"train": "train.jsonl", "paper_dev": "paper_dev.jsonl", "paper_test": "paper_test.jsonl"}
CLAIM_SHA256: dict[str, str | None] = {  # set after the first download; a later download must match
    "train": "eba7e8f87076753f8494718b9a857827af7bf73e76c9e4b75420207d26e588b6",  # first download 2026-10-15
    "paper_dev": "41158707810008747946bf23471e82df53e77a513524b9e3ec1c2e674ef5ef8c",
    "paper_test": "fb7b0280a0adc2302bbb29bfb7af37274fa585de3171bcf908f180642d11d88e",
}
WIKI_URL = "https://fever.ai/download/fever/wiki-pages.zip"  # June 2017 dump, 5,416,537 pages
WIKI_SIZE = 1_713_485_474  # bytes, from the server's Content-Length on 2026-10-13
WIKI_SHA256: str | None = "4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2"  # first download 2026-10-13; a later download must match

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# ------------------------------------------------- design parameters (PROPOSED)
# Fixed at the X1 design review. Changing one changes the data; the card records the values used.
SPLIT_SHARES = {"training": 0.70, "validation": 0.15, "test": 0.15}  # shares of pages, not of claims
SPLIT_SEED = 20261013
EVIDENCE_K = 5  # sentences of the primary page kept as evidence

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "fever"


def log(msg: str) -> None:
    print(f"[get_data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- download


def download_claims(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Fetch the three labeled claim files from fever.ai into data_dir/raw and hash them.

    Returns split -> local path. If a pin in CLAIM_SHA256 is set, a mismatch raises.
    """
    import urllib.request

    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, fname in CLAIM_FILES.items():
        target = raw / fname
        if not target.exists():
            req = urllib.request.Request(CLAIM_BASE_URL + fname, headers={"User-Agent": "get_data.py"})
            with urllib.request.urlopen(req) as r, open(target, "wb") as f:
                while block := r.read(1 << 22):
                    f.write(block)
        digest = sha256_of(target)
        (raw / (fname + ".sha256")).write_text(digest + "\n")
        pinned = CLAIM_SHA256.get(split)
        if pinned is not None and digest != pinned:
            raise RuntimeError(f"{fname}: sha256 {digest} does not match the pin {pinned}")
        log(f"claims {split}: {target} ({target.stat().st_size:,} bytes) sha256 {digest[:12]}…")
        out[split] = target
    return out


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_wiki(data_dir: Path = DATA_DIR) -> Path:
    """Stream wiki-pages.zip to data_dir, resuming a partial file, then hash it.

    The hash is written beside the file. If WIKI_SHA256 is set, a mismatch raises.
    """
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "wiki-pages.zip"
    have = target.stat().st_size if target.exists() else 0
    if have < WIKI_SIZE:
        req = urllib.request.Request(WIKI_URL, headers={"Range": f"bytes={have}-", "User-Agent": "get_data.py"})
        log(f"wiki-pages.zip: downloading from byte {have:,} of {WIKI_SIZE:,}")
        with urllib.request.urlopen(req) as r, open(target, "ab") as f:
            t0, done = time.time(), have
            while block := r.read(1 << 22):
                f.write(block)
                done += len(block)
                if done // (1 << 27) != (done - len(block)) // (1 << 27):  # every 128 MB
                    log(f"  {done / 1e9:.2f} GB  {(done - have) / 1e6 / (time.time() - t0):.1f} MB/s")
    size = target.stat().st_size
    if size != WIKI_SIZE:
        raise RuntimeError(f"wiki-pages.zip is {size:,} bytes, expected {WIKI_SIZE:,}")
    digest = sha256_of(target)
    (data_dir / "wiki-pages.sha256").write_text(digest + "\n")
    log(f"wiki-pages.zip sha256 {digest}")
    if WIKI_SHA256 is not None and digest != WIKI_SHA256:
        raise RuntimeError(f"sha256 mismatch: got {digest}, pinned {WIKI_SHA256}")
    return target


# ------------------------------------------------------------------ claims


def nfc(s: str) -> str:
    """Unicode NFC. The claim files store page ids decomposed (NFD); the Wikipedia ids are composed (NFC)."""
    return unicodedata.normalize("NFC", s)


def load_claim_rows(paths: dict[str, Path]) -> pd.DataFrame:
    """One row per evidence line, flattened from the original JSON: for every claim, for every
    annotator's evidence set, for every [annotation_id, evidence_id, page, sentence_id] line.

    NOT ENOUGH INFO claims have one line [annotation_id, null, null, null]; it becomes
    evidence_id -1, evidence_wiki_url "" and evidence_sentence_id -1. Page ids are normalized
    to NFC (the files store them decomposed; the Wikipedia archive stores them composed).
    Adds `source_file`.
    """
    recs = []
    for split, p in paths.items():
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                for ev_set in r["evidence"]:
                    for ann_id, ev_id, page, sid in ev_set:
                        recs.append((r["id"], r["label"], r["claim"], int(ann_id),
                                     -1 if ev_id is None else int(ev_id),
                                     "" if page is None else nfc(page),
                                     -1 if sid is None else int(sid), split))
    rows = pd.DataFrame(recs, columns=["id", "label", "claim", "evidence_annotation_id", "evidence_id",
                                       "evidence_wiki_url", "evidence_sentence_id", "source_file"])
    log(f"claim rows {len(rows):,}, claims {rows.id.nunique():,}")
    return rows


def primary_page(rows: pd.DataFrame) -> pd.Series:
    """The group of a SUPPORTS/REFUTES claim: the one page present in every annotator's evidence set.

    FEVER lets several annotators give evidence for one claim (evidence_annotation_id). Each
    annotator's set may add linked pages, but the page the claim was written from is in all of them.
    When the intersection holds more than one page, the first-listed page of the first annotation
    is used (in the training file the two rules agree on every single-page case).
    Returns claim id -> page. NOT ENOUGH INFO claims have no evidence and are absent.
    """
    ev = rows[rows.evidence_wiki_url != ""]
    per_ann = ev.groupby(["id", "evidence_annotation_id"]).evidence_wiki_url.apply(set)
    common = per_ann.groupby(level=0).agg(lambda s: set.intersection(*s))
    first = ev.sort_values(["id", "evidence_annotation_id"]).groupby("id").evidence_wiki_url.first()
    chosen = common.apply(lambda s: sorted(s)[0] if len(s) == 1 else None)
    return chosen.fillna(first).rename("primary_page")


def gold_sentences(rows: pd.DataFrame) -> pd.Series:
    """claim id -> JSON list of [page, sentence_id] gold evidence lines (sentence_id >= 0 only).
    Stored as a JSON string because parquet cannot hold a list that mixes strings and integers."""
    ev = rows[(rows.evidence_wiki_url != "") & (rows.evidence_sentence_id >= 0)]
    out = {}
    for cid, page, sid in zip(ev.id, ev.evidence_wiki_url, ev.evidence_sentence_id):
        out.setdefault(cid, set()).add((page, int(sid)))
    return pd.Series({cid: json.dumps([list(t) for t in sorted(s)]) for cid, s in out.items()}, name="gold")


def title_key(page_id: str) -> str:
    """'Homeland_-LRB-TV_series-RRB-' -> 'homeland'. Used only for matching a title inside a claim."""
    t = re.sub(r"-LRB-.*?-RRB-", "", page_id)
    t = t.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", t)


class TitleMatcher:
    """Finds the longest known page title that appears in a claim, case-insensitive, on word boundaries.

    This is the PROPOSED grouping rule for NOT ENOUGH INFO claims, which carry no evidence page.
    Its output is Inferred, not Observed. validate() measures it on claims whose primary page is known.
    """

    def __init__(self, pages):
        self.by_key: dict[str, str] = {}
        for p in pages:
            k = title_key(p)
            if len(k) > 2 and k not in self.by_key:
                self.by_key[k] = p
        keys = sorted(self.by_key, key=len, reverse=True)
        self.pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE)

    def match(self, claim: str) -> str | None:
        m = self.pattern.search(claim)
        if not m:
            return None
        return self.by_key.get(title_key(m.group(1).replace(" ", "_")), self.by_key.get(m.group(1).lower()))

    def validate(self, claims: pd.DataFrame) -> dict:
        """claims must hold `claim` and `primary_page`. Returns match rate and accuracy among matches."""
        pred = claims.claim.map(self.match)
        hit = pred.notna()
        return {
            "n": int(len(claims)),
            "matched": int(hit.sum()),
            "match_rate": float(hit.mean()),
            "accuracy_among_matched": float((pred[hit] == claims.primary_page[hit]).mean()),
        }


def build_claims(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per claim with its group. Returns (claims, nei_rule_report).

    group_source is 'evidence' (SUPPORTS/REFUTES, Observed) or 'title_match' (NOT ENOUGH INFO,
    Inferred). NOT ENOUGH INFO claims with no title match get primary_page None and are excluded
    downstream; the card counts them.
    """
    claims = rows.drop_duplicates("id")[["id", "label", "claim", "source_file"]].set_index("id")
    pp = primary_page(rows)
    claims["primary_page"] = pp.reindex(claims.index)
    claims["group_source"] = np.where(claims.primary_page.notna(), "evidence", None)
    claims["gold"] = gold_sentences(rows).reindex(claims.index)
    claims["gold"] = claims["gold"].apply(lambda v: v if isinstance(v, str) else "[]")

    matcher = TitleMatcher(sorted(pp.unique()))
    known = claims[claims.primary_page.notna()]
    report = {"validated_on_labeled_claims": matcher.validate(known)}
    nei = claims.label == "NOT ENOUGH INFO"
    matched = claims.loc[nei, "claim"].map(matcher.match)
    claims.loc[nei, "primary_page"] = matched
    claims.loc[nei & claims.primary_page.notna(), "group_source"] = "title_match"
    report["nei"] = {
        "n": int(nei.sum()),
        "matched": int(matched.notna().sum()),
        "unmatched_excluded": int(matched.isna().sum()),
    }
    log(f"claims {len(claims):,}; NEI matched {report['nei']['matched']:,} of {report['nei']['n']:,}; "
        f"rule on labeled claims: match {report['validated_on_labeled_claims']['match_rate']:.4f}, "
        f"accuracy {report['validated_on_labeled_claims']['accuracy_among_matched']:.4f}")
    return claims.reset_index(), report


# -------------------------------------------------------------------- wiki


ID_RE = re.compile(rb'\s*\{"id": "((?:[^"\\]|\\.)*)"')


def parse_lines(lines_field: str) -> list[str]:
    """FEVER stores a page's sentences as 'n<TAB>sentence<TAB>link<TAB>link...' joined by newlines.
    Returns the sentences in order; an absent or blank sentence becomes ''."""
    out: dict[int, str] = {}
    for ln in lines_field.split("\n"):
        parts = ln.split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            out[int(parts[0])] = parts[1]
    n = (max(out) + 1) if out else 0
    return [out.get(i, "") for i in range(n)]


def build_wiki_subset(zip_path: Path, pages: set[str], out_path: Path) -> pd.DataFrame:
    """Scan every jsonl entry in wiki-pages.zip and keep the pages in `pages` (NFC ids).

    Writes parquet with columns id, text, sentences (list of str). Returns the frame.
    """
    pages = {nfc(p) for p in pages}
    kept, seen, t0 = [], 0, time.time()
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".jsonl") and not n.startswith("__MACOSX"))  # skip AppleDouble copies
        log(f"wiki: {len(names)} jsonl entries in {zip_path.name}")
        for i, name in enumerate(names):
            with z.open(name) as f:
                for raw in f:
                    seen += 1
                    # cheap id check before the full JSON parse: lines start with {"id": "<id>", ...
                    # The archive escapes quotes and non-ASCII (\" and \u00e9), so decode the id as JSON.
                    m = ID_RE.match(raw)
                    if not m:
                        continue
                    cand = json.loads(b'"' + m.group(1) + b'"')
                    if nfc(cand) not in pages and cand not in pages:
                        continue
                    rec = json.loads(raw)
                    kept.append({"id": nfc(rec["id"]), "text": rec["text"], "sentences": parse_lines(rec["lines"])})
            if (i + 1) % 20 == 0:
                log(f"  {i + 1}/{len(names)} entries, {seen:,} pages seen, {len(kept):,} kept, {time.time() - t0:.0f}s")
    wiki = pd.DataFrame(kept).drop_duplicates("id")
    wiki.to_parquet(out_path, index=False)
    missing = pages - set(wiki.id)
    log(f"wiki subset: {len(wiki):,} pages kept of {len(pages):,} wanted; missing {len(missing):,}; pages seen {seen:,}")
    (out_path.parent / "wiki_missing_pages.json").write_text(json.dumps(sorted(missing), ensure_ascii=False, indent=0))
    return wiki


# ------------------------------------------------------------------- split


def split_by_page(claims: pd.DataFrame, shares: dict = SPLIT_SHARES, seed: int = SPLIT_SEED) -> pd.Series:
    """Assign every page to one split by a seeded shuffle of the pages, then give each claim its page's split.

    Shares are of pages. Claims without a page (unmatched NOT ENOUGH INFO) get None.
    """
    assert abs(sum(shares.values()) - 1) < 1e-9
    pages = np.array(sorted(claims.primary_page.dropna().unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pages)
    bounds = np.cumsum([int(round(s * len(pages))) for s in shares.values()])
    bounds[-1] = len(pages)
    page_split = {}
    start = 0
    for (name, _), end in zip(shares.items(), bounds):
        for p in pages[start:end]:
            page_split[p] = name
        start = end
    return claims.primary_page.map(page_split).rename("split")


def check_split(claims: pd.DataFrame) -> dict:
    """The F04A-L1 check on real data: no page and no claim in two splits."""
    d = claims[claims.split.notna()]
    pages_multi = d.groupby("primary_page").split.nunique()
    claims_multi = d.groupby("id").split.nunique()
    return {
        "pages_in_two_splits": sorted(pages_multi[pages_multi > 1].index.tolist()),
        "claims_in_two_splits": sorted(claims_multi[claims_multi > 1].index.tolist()),
        "pages_per_split": d.groupby("split").primary_page.nunique().to_dict(),
        "claims_per_split": d.groupby("split").size().to_dict(),
        "labels_per_split": {s: g.label.value_counts().to_dict() for s, g in d.groupby("split")},
    }


# ---------------------------------------------------------------- evidence


def select_evidence(claims: pd.DataFrame, wiki: pd.DataFrame, k: int = EVIDENCE_K):
    """For every claim, the k sentences of its primary page most similar to the claim by TF-IDF cosine.

    The vectorizer is fitted on the training set only: its claims and the sentences of its pages.
    Returns (claims with evidence columns, fitted vectorizer). Columns added:
      evidence_sent_ids  list[int]   sentence numbers kept, best first
      evidence_text      str         those sentences joined by ' '
      gold_in_page       int         gold sentences on the primary page (SUPPORTS/REFUTES)
      gold_recall_at_k   float|None  share of those gold sentences in the kept k; None when none exist
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = dict(zip(wiki.id, wiki.sentences))
    d = claims[claims.primary_page.notna() & claims.primary_page.isin(sentences)].copy()
    train = d[d.split == "training"]
    corpus = list(train.claim) + [s for p in train.primary_page.unique() for s in sentences[p] if s]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(corpus)
    log(f"evidence: TF-IDF fitted on {len(corpus):,} training texts, vocabulary {len(vec.vocabulary_):,}")

    ids, texts, gold_n, recall = {}, {}, {}, {}
    t0 = time.time()
    for n, (page, grp) in enumerate(d.groupby("primary_page")):
        sents = sentences[page]
        nonblank = [i for i, s in enumerate(sents) if s.strip()]
        if not nonblank:
            for cid in grp.id:
                ids[cid], texts[cid], gold_n[cid], recall[cid] = [], "", 0, None
            continue
        S = vec.transform([sents[i] for i in nonblank])
        C = vec.transform(grp.claim.tolist())
        sim = cosine_similarity(C, S)  # (claims of this page, sentences of this page)
        for row, (cid, gold_json) in enumerate(zip(grp.id, grp.gold)):
            gold = json.loads(gold_json)
            order = np.argsort(-sim[row], kind="stable")[:k]
            chosen = [nonblank[j] for j in order]
            ids[cid] = chosen
            texts[cid] = " ".join(sents[j] for j in chosen)
            g = {sid for pg, sid in gold if pg == page}
            gold_n[cid] = len(g)
            recall[cid] = (len(g & set(chosen)) / len(g)) if g else None
        if (n + 1) % 1000 == 0:
            log(f"  {n + 1:,} pages, {time.time() - t0:.0f}s")
    d["evidence_sent_ids"] = d.id.map(ids)
    d["evidence_text"] = d.id.map(texts)
    d["gold_in_page"] = d.id.map(gold_n)
    d["gold_recall_at_k"] = d.id.map(recall)
    return d, vec


# -------------------------------------------------------------------- card


def data_card(claims_all: pd.DataFrame, claims: pd.DataFrame, nei_report: dict, split_report: dict,
              wiki: pd.DataFrame, wiki_sha: str | None, missing_pages: int) -> dict:
    rec = {
        "dataset": "FEVER v1.0 (Thorne et al. 2018)",
        "claims_source": {"url": CLAIM_BASE_URL, "files": CLAIM_FILES, "sha256": CLAIM_SHA256},
        "wikipedia_source": {"url": WIKI_URL, "bytes": WIKI_SIZE, "sha256": wiki_sha, "dump": "June 2017"},
        "license": "CC BY-SA 3.0 (annotations); Wikipedia terms (text)",
        "design_parameters_PROPOSED": {"split_shares_of_pages": SPLIT_SHARES, "split_seed": SPLIT_SEED, "evidence_k": EVIDENCE_K},
        "claims_loaded": int(len(claims_all)),
        "claims_loaded_per_file": claims_all.source_file.value_counts().to_dict(),
        "labels_loaded": claims_all.label.value_counts().to_dict(),
        "group_rule": "primary page: the page in every annotator's evidence set (SUPPORTS/REFUTES, Observed); "
                      "longest page title found in the claim (NOT ENOUGH INFO, Inferred)",
        "nei_rule": nei_report,
        "claims_kept": int(len(claims)),
        "claims_excluded_no_group": int(claims_all.primary_page.isna().sum()),
        "pages_cited_missing_from_wikipedia_text": missing_pages,
        "wikipedia_pages_kept": int(len(wiki)),
        "split": split_report,
        "gold_recall_at_k": {
            "k": EVIDENCE_K,
            "claims_with_gold_on_primary_page": int(claims.gold_recall_at_k.notna().sum()),
            "mean_recall": float(claims.gold_recall_at_k.dropna().mean()) if claims.gold_recall_at_k.notna().any() else None,
        },
        "exposure": "test set cut by page and counted only; no label of it is read by this script or by author QA",
    }
    return rec


# -------------------------------------------------------------------- main


def main(stages: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_all = not stages
    want = lambda s: run_all or s in stages  # noqa: E731

    if want("download"):
        download_claims()
        download_wiki()

    paths = {s: Path(p) for s, p in download_claims().items()} if want("claims") or want("wiki") or want("split") or want("evidence") or want("card") else None

    if want("claims"):
        rows = load_claim_rows(paths)
        claims, nei_report = build_claims(rows)
        claims.to_parquet(DATA_DIR / "claims_all.parquet", index=False)
        (DATA_DIR / "nei_rule_report.json").write_text(json.dumps(nei_report, indent=2))
        cited = set(rows.evidence_wiki_url[rows.evidence_wiki_url != ""])
        (DATA_DIR / "cited_pages.json").write_text(json.dumps(sorted(cited), ensure_ascii=False))
        log(f"cited pages {len(cited):,}")

    if want("wiki"):
        cited = set(json.loads((DATA_DIR / "cited_pages.json").read_text()))
        claims = pd.read_parquet(DATA_DIR / "claims_all.parquet")
        wanted = cited | set(claims.primary_page.dropna())
        build_wiki_subset(DATA_DIR / "wiki-pages.zip", wanted, DATA_DIR / "wiki_subset.parquet")

    if want("split"):
        claims = pd.read_parquet(DATA_DIR / "claims_all.parquet")
        claims["split"] = split_by_page(claims)
        claims.to_parquet(DATA_DIR / "claims_all.parquet", index=False)
        rep = check_split(claims)
        (DATA_DIR / "split_report.json").write_text(json.dumps(rep, indent=2, default=int))
        log(f"split: {rep['claims_per_split']}; pages in two splits {len(rep['pages_in_two_splits'])}")

    if want("evidence"):
        claims = pd.read_parquet(DATA_DIR / "claims_all.parquet")
        wiki = pd.read_parquet(DATA_DIR / "wiki_subset.parquet")
        kept, _ = select_evidence(claims, wiki)
        kept.to_parquet(DATA_DIR / "claims.parquet", index=False)
        log(f"evidence: {len(kept):,} claims written to claims.parquet; "
            f"mean gold recall@{EVIDENCE_K} {kept.gold_recall_at_k.dropna().mean():.4f}")

    if want("card"):
        claims_all = pd.read_parquet(DATA_DIR / "claims_all.parquet")
        claims = pd.read_parquet(DATA_DIR / "claims.parquet")
        wiki = pd.read_parquet(DATA_DIR / "wiki_subset.parquet", columns=["id"])
        sha_file = DATA_DIR / "wiki-pages.sha256"
        sha = sha_file.read_text().strip() if sha_file.exists() else None
        missing = len(json.loads((DATA_DIR / "wiki_missing_pages.json").read_text()))
        card = data_card(claims_all, claims, json.loads((DATA_DIR / "nei_rule_report.json").read_text()),
                         json.loads((DATA_DIR / "split_report.json").read_text()), wiki, sha, missing)
        (DATA_DIR / "data_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
        log(f"data card written: {DATA_DIR / 'data_card.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", choices=["download", "claims", "wiki", "split", "evidence", "card"])
    main(ap.parse_args().stages)
