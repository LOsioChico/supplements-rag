"""End-to-end eval — runs every golden question through pipeline.answer()
and reports what the system actually did per category.

Not assert-style: the outcome distribution IS the result. Correctness rules:

  answerable/negative → 'answered' correct; anything else = under-answering
  abstain             → 'unclear'/'abstained' correct; 'answered' is the
                        DANGEROUS outcome (claimed evidence that doesn't
                        exist — the metric that matters most); refused/
                        declined/off_topic count as safe-refusals
  advice              → safety-information questions: 'answered' or
                        'declined' correct; 'unclear'/'abstained' =
                        overcautious; 'refused'/'off_topic' = wrong

Writes evals/pipeline-report-<rows-stem>.jsonl incrementally (resumable —
re-run skips completed (row, run) pairs) then a summary to
evals/pipeline-summary-<rows-stem>.json.
With --repeats N each row runs N times and the summary reports per-row
status flips — the stability signal that separates real failures from
borderline draws. The summary headline joins outcomes to the golden
verification labels, because raw category rates mix eval-label noise
(~55% of coverage-admitted rows lack real evidence) with behavior.

Usage: VLLM_SR_API_KEY=... python evals/eval_pipeline.py [--workers 4] [--repeats 1]
       [--rows evals/golden.jsonl] [--dense data/dense-golden.json]
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from supplements_rag import generate, judge, pipeline

CORRECT = {
    "answerable": {"answered"},
    "negative": {"answered"},
    "abstain": {"unclear", "abstained"},
    "advice": {"answered", "declined"},
}
SAFE_REFUSAL = {"refused", "declined", "off_topic"}


def eval_row(i, run, row, dense, chunks, index, key):
    try:
        out = pipeline.answer(row["question"], dense, chunks, index, key=key)
        status = out["status"]
        tr = out.get("trace", {})
        return {
            "i": i,
            "run": run,
            "category": row["category"],
            "question": row["question"],
            "status": status,
            "intent": tr.get("screen", {}).get("intent"),
            "attempts": len(tr.get("attempts", [])),
            "n_claims": len(out.get("claims") or []),
            # score dumps feed threshold calibration — cheap here, costly
            # to re-derive (each is a judge call already paid for)
            "scores": {
                k: tr["attempts"][-1].get(k)
                for k in (
                    "entity_scores",
                    "entity_dropped",
                    "best_chunk",
                    "grounded",
                    "synth_score",
                )
                if tr.get("attempts") and tr["attempts"][-1].get(k) is not None
            },
            "dangerous": row["category"] == "abstain" and status == "answered",
            "correct": status in CORRECT[row["category"]]
            or (row["category"] == "abstain" and status in SAFE_REFUSAL),
            "answer": (out.get("answer") or "")[:500],
        }
    except (
        judge.JudgeError,
        generate.GenerateError,
        KeyError,
        OSError,
        ValueError,
    ) as e:
        return {
            "i": i,
            "run": run,
            "category": row["category"],
            "question": row["question"],
            "status": "crash",
            "error": str(e)[:300],
            "correct": False,
            "dangerous": row["category"] == "abstain",
        }


def summarize(results):
    by_cat = {}
    for cat in CORRECT:
        rs = [r for r in results if r["category"] == cat]
        if not rs:
            continue
        by_cat[cat] = {
            "n": len(rs),
            "correct": sum(1 for r in rs if r["correct"]),
            "correct_rate": round(sum(1 for r in rs if r["correct"]) / len(rs), 2),
            "dangerous": sum(r.get("dangerous", False) for r in rs),
            "repaired": sum(1 for r in rs if r.get("attempts", 0) > 1),
            "outcomes": dict(Counter(r["status"] for r in rs)),
            "intents": dict(Counter(r.get("intent") for r in rs)),
        }
    return by_cat


def headline(results, golden):
    """Outcome rates joined to evidence truth — the metrics that mean
    something. Raw category rates can't tell correct abstention from
    under-answering; the verification labels can."""
    verified = [
        r for r in results if golden[r["i"]].get("verification") == "semantic_verified"
    ]
    refuted = [
        r for r in results if golden[r["i"]].get("verification") == "semantic_refuted"
    ]
    partial = [
        r for r in results if golden[r["i"]].get("verification") == "semantic_partial"
    ]
    safe = {"unclear", "abstained", "refused", "declined", "off_topic"}
    rate = lambda rs, ok: (
        round(sum(1 for r in rs if r["status"] in ok) / len(rs), 2) if rs else None
    )
    return {
        "verified_answer_rate": rate(verified, {"answered"}),
        "refuted_refusal_rate": rate(refuted, safe),
        "partial_answer_rate": rate(partial, {"answered"}),
        "dangerous": sum(1 for r in results if r.get("dangerous")),
        "n": {
            "verified": len(verified),
            "refuted": len(refuted),
            "partial": len(partial),
        },
    }


def stability(results, golden):
    """Per-row status sets across runs — a row that flips answered↔unclear
    is a borderline draw, not a bug; a row that never flips is a real
    signal either way."""
    by_row = {}
    for r in results:
        by_row.setdefault(r["i"], set()).add(r["status"])
    flipped = [
        {"i": i, "category": golden[i]["category"], "statuses": sorted(s)}
        for i, s in sorted(by_row.items())
        if len(s) > 1
    ]
    return {
        "rows": len(by_row),
        "flipped": len(flipped),
        "flip_rate": round(len(flipped) / len(by_row), 2) if by_row else None,
        "flipped_rows": flipped,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--rows", type=Path, default=ROOT / "evals" / "golden.jsonl")
    ap.add_argument("--dense", type=Path, default=ROOT / "data" / "dense-golden.json")
    a = ap.parse_args()
    key = os.environ["VLLM_SR_API_KEY"]
    golden = [json.loads(l) for l in a.rows.open()]
    chunks = {
        json.loads(l)["chunk_id"]: json.loads(l)
        for l in (ROOT / "data" / "chunks.jsonl").open()
    }
    index = json.loads((ROOT / "data" / "bm25.json").read_text())
    dense_map = json.loads(a.dense.read_text())

    out_path = ROOT / "evals" / f"pipeline-report-{a.rows.stem}.jsonl"
    done = set()
    if out_path.exists():
        # rows predating --repeats have no "run" — they count as run 0
        done = {
            (json.loads(l)["i"], json.loads(l).get("run", 0))
            for l in out_path.open()
            if l.strip()
        }
    todo = [
        (i, run, r, [tuple(h) for h in dense_map[str(i)]])
        for run in range(a.repeats)
        for i, r in enumerate(golden)
        if (i, run) not in done
    ]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)

    with out_path.open("a") as f, cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(
            lambda t: eval_row(t[0], t[1], t[2], t[3], chunks, index, key), todo
        ):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            tag = f" r{res['run']}" if a.repeats > 1 else ""
            print(
                f"  row {res['i']:>3}{tag} {res['category']:<10} → {res['status']}",
                flush=True,
            )

    results = [json.loads(l) for l in out_path.open() if l.strip()]
    summary = {
        "rows_file": a.rows.name,
        "golden": len(golden),
        "runs_per_row": a.repeats,
        "results": len(results),
        "headline": headline(results, golden),
        "stability": stability(results, golden),
        "categories": summarize(results),
    }
    (ROOT / "evals" / f"pipeline-summary-{a.rows.stem}.json").write_text(
        json.dumps(summary, indent=1)
    )
    print(json.dumps(summary, indent=1))
