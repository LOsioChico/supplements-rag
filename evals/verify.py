"""Semantic verification — the vllm-sr SystemOne judge scores each proposed
evidence PMID against its question. Completes the admission rule.

The judge is a semantic-router scoring endpoint:
POST {base}/v1/systemone  {model, state, questions} → answers.{key}.noul
(a 0–1 support score). Python-native — no JS handoff, stdlib only.

Verdict bands (assumption, tunable — raw scores are stored so bands can be
re-derived): >=0.7 yes / >=0.4 partial / <0.4 no.

Writes data/verdicts.json {row_index: {pmid: score}} then folds verdicts back
into golden.jsonl:
- evidence_verdicts — per-PMID score map, kept for audit
- verification — semantic_verified (>=1 yes) / semantic_partial (>=1 partial)
  / semantic_refuted (all no) / verification_error; retrieval_only rows
  (no evidence_pmids — abstain, advice) stay untouched

Usage: VLLM_SR_API_KEY=... python evals/verify.py [--workers 8]
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("VLLM_SR_BASE", "https://app.vllm-sr.ai")
MODEL = os.environ.get("S1_MODEL", "vllm-sr/d3-flash")
YES, PARTIAL = 0.7, 0.4
VERIFIED, PARTIAL_V, REFUTED = (
    "semantic_verified",
    "semantic_partial",
    "semantic_refuted",
)


def judge(key: str, base: str, model: str, question: str, chunk: dict) -> float:
    """One noul score: does this abstract support the question's answer?"""
    payload = {
        "model": model,
        "state": (
            f"Source:\n{chunk['title']}\n{chunk['text'][:12000]}\n\n"
            f"Question being checked: {question}"
        ),
        "questions": {
            "ev": {
                "type": "noul",
                "instructions": "Does the source provide evidence that answers "
                f"this question: {question}",
            }
        },
    }
    req = urllib.request.Request(
        f"{base}/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["answers"]["ev"]["noul"]


def status_of(scores: dict) -> str:
    vals = [v for v in scores.values() if isinstance(v, float)]
    if not vals:
        return "verification_error"
    if max(vals) >= YES:
        return VERIFIED
    if max(vals) >= PARTIAL:
        return PARTIAL_V
    return REFUTED


def apply_verdicts(rows: list[dict], verdicts: dict) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        v = verdicts.get(str(i))
        if not v or not row["evidence_pmids"]:
            out.append(row)
            continue
        out.append({**row, "evidence_verdicts": v, "verification": status_of(v)})
    return out


def judge_all(
    rows, chunks, key, workers, index=None, dense_map=None, done_path=None
) -> dict:
    done_path = done_path or ROOT / "data" / "verdicts.json"
    verdicts: dict[str, dict] = {}
    if done_path.exists():
        verdicts = json.loads(done_path.read_text())
    pairs = {
        (i, p) for i, r in enumerate(rows) for p in r["evidence_pmids"] if p in chunks
    }
    if index is not None and dense_map is not None:
        # Re-proposal: rows still not verified get a second, deeper
        # candidate pool — BM25 top-10 + dense top-10 — before "no
        # evidence" stands. Costs calls only where the verdict matters.
        from supplements_rag.bm25 import score as bm25_score

        for i, r in enumerate(rows):
            judged = {
                p: v
                for p, v in verdicts.get(str(i), {}).items()
                if isinstance(v, float)
            }
            if judged and status_of(judged) == VERIFIED:
                continue
            deeper = {p for p, _ in bm25_score(index, r["question"], top_k=10)}
            deeper |= {p for p, _ in dense_map.get(str(i), [])[:10]}
            pairs |= {(i, p) for p in deeper if p in chunks}
    todo = [
        (i, p)
        for i, p in sorted(pairs)
        if not isinstance(verdicts.get(str(i), {}).get(p), float)
    ]

    def one(i_p):
        i, p = i_p
        row = rows[i]
        try:
            return i, p, judge(key, BASE, MODEL, row["question"], chunks[p])
        except Exception as e:  # noqa: BLE001 — record, don't silently drop
            return i, p, "error: " + str(e)[:80]

    with cf.ThreadPoolExecutor(workers) as ex:
        for n, (i, p, v) in enumerate(ex.map(one, todo), 1):
            verdicts.setdefault(str(i), {})[p] = v
            if n % 20 == 0:
                done_path.write_text(json.dumps(verdicts))
                print(f"{n}/{len(todo)}", file=sys.stderr)
    done_path.write_text(json.dumps(verdicts))
    return verdicts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--repropose",
        action="store_true",
        help="deeper candidate pool for unverified rows",
    )
    ap.add_argument(
        "--rows",
        default=str(ROOT / "evals" / "golden.jsonl"),
        help="jsonl of rows to verify (default: golden set)",
    )
    ap.add_argument(
        "--verdicts",
        default=str(ROOT / "data" / "verdicts.json"),
        help="checkpoint file for raw scores",
    )
    ap.add_argument(
        "--dense",
        default=str(ROOT / "data" / "dense-golden.json"),
        help="per-row dense lists for --repropose ({row_index: hits})",
    )
    a = ap.parse_args()
    key = os.environ["VLLM_SR_API_KEY"]
    rows_path = Path(a.rows)
    rows = [json.loads(l) for l in rows_path.open()]
    chunks = {
        json.loads(l)["chunk_id"]: json.loads(l)
        for l in (ROOT / "data" / "chunks.jsonl").open()
    }
    index = dense = None
    if a.repropose:
        index = json.loads((ROOT / "data" / "bm25.json").read_text())
        dense = json.loads(Path(a.dense).read_text())
    verdicts = judge_all(rows, chunks, key, a.workers, index, dense, Path(a.verdicts))
    updated = apply_verdicts(rows, verdicts)
    rows_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in updated) + "\n"
    )
    print(
        json.dumps(
            dict(Counter(r.get("verification", "no_evidence") for r in updated)),
            indent=1,
        )
    )
