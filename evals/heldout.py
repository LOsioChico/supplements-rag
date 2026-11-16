"""Held-out eval set — a second stratified draw from the admitted pool,
disjoint from golden.jsonl by question text. The golden set stays the
tuning set; this one only measures — the anti-overfit check.

Stratification mirrors golden's category quotas scaled ×4, then splits
each category proportionally across verification labels — the pool's
truth skew (mostly refuted) is kept, not filtered.

Usage: python evals/heldout.py [--n 200] [--seed 42]
  → evals/heldout.jsonl
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUOTA = {"answerable": 0.40, "negative": 0.25, "abstain": 0.125, "advice": 0.225}


def heldout_sample(
    admitted: list[dict], golden_qs: set, n: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    by_cell = defaultdict(list)
    for r in admitted:
        if r["question"].strip().lower() in golden_qs:
            continue
        by_cell[(r["category"], r.get("verification", "none"))].append(r)
    out = []
    for cat, frac in QUOTA.items():
        groups = [v for (c, _), v in by_cell.items() if c == cat]
        total = sum(len(g) for g in groups)
        for g in groups:
            take = min(len(g), round(n * frac * len(g) / max(total, 1)))
            out.extend(rng.sample(g, take))
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    admitted = [json.loads(l) for l in (ROOT / "evals" / "admitted.jsonl").open()]
    golden_qs = {
        json.loads(l)["question"].strip().lower()
        for l in (ROOT / "evals" / "golden.jsonl").open()
    }
    rows = heldout_sample(admitted, golden_qs, a.n, a.seed)
    out = ROOT / "evals" / "heldout.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(
        json.dumps(
            {
                "heldout": len(rows),
                "by_category": dict(Counter(r["category"] for r in rows)),
                "by_verification": dict(
                    Counter(r.get("verification", "none") for r in rows)
                ),
            },
            indent=1,
        )
    )
