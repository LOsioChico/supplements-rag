"""Golden-set admission — seeds become eval rows only with evidence receipts.

The rule: a question is admitted only when its evidence PMIDs verify inside
our corpus. This script does the mechanical half — coverage check via BM25
+ top-k evidence PMIDs attached per seed — and emits three auditable files:

- evals/admitted.jsonl — candidate pool, one row per admitted seed
- evals/rejected.jsonl — every dropped seed with its rejection reason
- evals/golden.jsonl   — stratified sample for actual eval runs

Evidence PMIDs are BM25-proposed, marked "verification": "retrieval_only" —
the semantic pass is `evals/verify.py` (the vllm-sr judge), which folds
per-PMID support scores back into the rows. Measured finding: ~52% of
lexically-admitted pool rows have no real evidence — coverage is not
evidence.

Categories → expected_behavior:
- answerable → answer_with_citations (MedQuAD info/effective/how-it-works,
  SuppLLMent positive-effect labels, corpus-covered)
- negative   → answer_with_citations (SuppLLMent ineffective labels — cite
  the no-benefit evidence; different from abstaining)
- abstain    → abstain (insufficient_evidence label, no corpus coverage)
- advice     → answer_or_decline (MedQuAD dose/interactions/precautions are
  safety-information questions — evidence-answerable; a decline is also
  defensible, so the eval records the gate's per-row intent)

Usage: python evals/admit.py [--bm25-min 8] [--seed 42]
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supplements_rag.bm25 import score

ROOT = Path(__file__).resolve().parents[1]
ANSWERABLE_QTYPES = {"information", "how effective is it", "how does it work"}
POSITIVE_LABELS = {"effective_for", "likely_effective_for", "possibly_effective_for"}
NEGATIVE_LABELS = {
    "ineffective_for",
    "likely_ineffective_for",
    "possibly_ineffective_for",
}
EXPECTED = {
    "answerable": "answer_with_citations",
    "negative": "answer_with_citations",
    "abstain": "abstain",
    "advice": "answer_or_decline",
}
# Per-category quotas for golden.jsonl (~120 rows — small enough to review).
GOLDEN_QUOTA = {"answerable": 50, "negative": 30, "advice": 25, "abstain": 15}


def load_index() -> dict:
    return json.loads((ROOT / "data" / "bm25.json").read_text())


def covered(index: dict, query: str, bm25_min: float) -> tuple[bool, list[str]]:
    hits = score(index, query, top_k=3)
    keep = [cid for cid, s in hits if s > bm25_min]
    return bool(keep), keep


def row(
    source: str, category: str, question: str, evidence_pmids: list[str], **extra
) -> dict:
    return {
        "source": source,
        "category": category,
        "question": question,
        "expected_behavior": EXPECTED[category],
        "evidence_pmids": evidence_pmids,
        "verification": "retrieval_only",
        **extra,
    }


def admit_medquad(index: dict, seeds: list[dict], bm25_min: float) -> tuple[list, list]:
    admitted, rejected = [], []
    for s in seeds:
        base = {"supplement": s["supplement"], "qtype": s["qtype"]}
        if s["qtype"] not in ANSWERABLE_QTYPES:
            admitted.append(row("medquad", "advice", s["question"], [], **base))
            continue
        ok, pmids = covered(index, s["supplement"], bm25_min)
        if ok:
            admitted.append(row("medquad", "answerable", s["question"], pmids, **base))
        else:
            rejected.append(
                {
                    "source": "medquad",
                    "reason": "no_corpus_coverage",
                    **base,
                    "question": s["question"],
                }
            )
    return admitted, rejected


def admit_suppllment(
    index: dict, seeds: list[dict], bm25_min: float
) -> tuple[list, list]:
    admitted, rejected = [], []
    for s in seeds:
        base = {
            "supplement": s["substance"],
            "condition": s["condition"],
            "label": s["label"],
        }
        question = f"Is {s['substance']} effective for {s['condition']}?"
        ok, pmids = covered(index, f"{s['substance']} {s['condition']}", bm25_min)
        label = s["label"]
        if label == "insufficient_evidence":
            if ok:
                rejected.append(
                    {
                        "source": "suppllment",
                        "reason": "labeled_insufficient_but_covered",
                        **base,
                        "question": question,
                    }
                )
            else:
                admitted.append(row("suppllment", "abstain", question, [], **base))
        elif label in NEGATIVE_LABELS or label in POSITIVE_LABELS:
            cat = "negative" if label in NEGATIVE_LABELS else "answerable"
            if ok:
                admitted.append(row("suppllment", cat, question, pmids, **base))
            else:
                rejected.append(
                    {
                        "source": "suppllment",
                        "reason": "no_corpus_coverage",
                        **base,
                        "question": question,
                    }
                )
        else:
            rejected.append(
                {
                    "source": "suppllment",
                    "reason": "unknown_label",
                    **base,
                    "question": question,
                }
            )
    return admitted, rejected


def golden_sample(admitted: list[dict], seed: int) -> list[dict]:
    """Stratified sample — per-category quota, split proportionally across
    sources so neither MedQuAD nor SuppLLMent dominates 'answerable'."""
    rng = random.Random(seed)
    by_cat_source = defaultdict(list)
    for r in admitted:
        by_cat_source[(r["category"], r["source"])].append(r)
    out = []
    for cat, quota in GOLDEN_QUOTA.items():
        groups = [v for (c, _), v in by_cat_source.items() if c == cat]
        total = sum(len(g) for g in groups)
        for g in groups:
            take = min(len(g), round(quota * len(g) / max(total, 1)))
            out.extend(rng.sample(g, take))
    rng.shuffle(out)
    return out


def report(admitted: list, rejected: list, golden: list) -> dict:
    return {
        "admitted": len(admitted),
        "by_category": dict(Counter(r["category"] for r in admitted)),
        "rejected": len(rejected),
        "reject_reasons": dict(Counter(r["reason"] for r in rejected)),
        "golden": len(golden),
        "golden_by_category": dict(Counter(r["category"] for r in golden)),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25-min", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    index = load_index()

    def load(p):
        return [json.loads(l) for l in (ROOT / "evals" / "seeds" / p).open()]

    mq_a, mq_r = admit_medquad(index, load("medquad.jsonl"), a.bm25_min)
    sp_a, sp_r = admit_suppllment(index, load("suppllment.jsonl"), a.bm25_min)
    admitted, rejected = mq_a + sp_a, mq_r + sp_r
    golden = golden_sample(admitted, a.seed)
    write_jsonl(ROOT / "evals" / "admitted.jsonl", admitted)
    write_jsonl(ROOT / "evals" / "rejected.jsonl", rejected)
    write_jsonl(ROOT / "evals" / "golden.jsonl", golden)
    print(json.dumps(report(admitted, rejected, golden), indent=1))
