"""Eval harness — runs golden.jsonl through retrieval, scores per category.

What each category measures at this stage (see evals/README.md honest limits):

- answerable / negative: **retention@k** — does the RRF-fused top-k keep at
  least one proposed evidence PMID? NOTE: evidence PMIDs are BM25-proposed,
  so this is NOT true recall — it measures whether fusion demotes them, not
  whether retrieval finds ground truth. True recall waits on the semantic
  verification pass.
- abstain: retrieval-strength signal — top BM25 score + top dense similarity.
  Abstain rows have no corpus coverage by construction, so weak scores here
  vs strong scores on answerable rows is the separation an abstention
  threshold needs. (Vectorize always returns 20 nearest hits — a "hit" is
  not evidence of coverage.)
- advice: naive advice-pattern detection rate — informs the unbuilt gate;
  production decline is a designed contract, not this regex.

Usage: python evals/run.py  (needs data/bm25.json, data/chunks.jsonl,
data/dense-golden.json — from `node scripts/embed-questions.mjs`)
"""

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supplements_rag.bm25 import score as bm25_score
from supplements_rag.query import retrieve

ROOT = Path(__file__).resolve().parents[1]
ADVICE_RE = re.compile(
    r"dose|dosage|how much|should i|safe|safety|precaution|side effect"
    r"|interact|take .* with|pregnan|child",
    re.IGNORECASE,
)


def hits_at_k(fused_ids: list[str], pmids: list[str], k: int) -> bool:
    return bool(set(fused_ids[:k]) & set(pmids))


def median(xs: list[float]) -> float:
    return round(statistics.median(xs), 2) if xs else 0.0


def eval_row(index: dict, chunks_by_id: dict, row: dict, dense: list) -> dict:
    sparse = bm25_score(index, row["question"], top_k=20)
    # the production path — fused + rare-entity injection — is what we
    # measure, not the bare RRF list
    pack = retrieve(index, chunks_by_id, row["question"], dense, top_k=20)
    fused_ids = [c["chunk_id"] for c in pack]
    ev = row["evidence_pmids"]
    return {
        "category": row["category"],
        "retention@5": hits_at_k(fused_ids, ev, 5) if ev else None,
        "retention@10": hits_at_k(fused_ids, ev, 10) if ev else None,
        "bm25_top": sparse[0][1] if sparse else 0.0,
        "dense_top": dense[0][1] if dense else 0.0,
        "advice_pattern": bool(ADVICE_RE.search(row["question"])),
    }


def summarize(results: list[dict]) -> dict:
    by_cat: dict[str, dict] = {}
    for cat in ("answerable", "negative", "abstain", "advice"):
        rs = [r for r in results if r["category"] == cat]
        ret5 = [r["retention@5"] for r in rs if r["retention@5"] is not None]
        ret10 = [r["retention@10"] for r in rs if r["retention@10"] is not None]
        by_cat[cat] = {
            "n": len(rs),
            "retention@5": round(sum(ret5) / len(ret5), 2) if ret5 else None,
            "retention@10": round(sum(ret10) / len(ret10), 2) if ret10 else None,
            "bm25_top_median": median([r["bm25_top"] for r in rs]),
            "bm25_top_range": [
                round(min(r["bm25_top"] for r in rs), 2),
                round(max(r["bm25_top"] for r in rs), 2),
            ]
            if rs
            else None,
            "dense_top_median": median([r["dense_top"] for r in rs]),
            "advice_pattern_rate": round(
                sum(r["advice_pattern"] for r in rs) / len(rs), 2
            )
            if rs
            else None,
        }
    return by_cat


if __name__ == "__main__":
    index = json.loads((ROOT / "data" / "bm25.json").read_text())
    chunks_by_id = {
        json.loads(l)["chunk_id"]: json.loads(l)
        for l in (ROOT / "data" / "chunks.jsonl").open()
    }
    golden = [json.loads(l) for l in (ROOT / "evals" / "golden.jsonl").open()]
    dense_map = json.loads((ROOT / "data" / "dense-golden.json").read_text())
    results = [
        eval_row(index, chunks_by_id, row, [tuple(h) for h in dense_map[str(i)]])
        for i, row in enumerate(golden)
    ]
    report = {"golden": len(golden), "categories": summarize(results)}
    (ROOT / "evals" / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
