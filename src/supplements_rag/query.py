"""Query stage — question → fused evidence pack.

Question enters, one BM25 ranked list and one vector ranked list come back,
RRF fuses them into a single ranked evidence pack. This module owns the
keyword side + fusion; the vector side arrives pre-computed (the Worker calls
Vectorize, a build-time script calls it through wrangler — either way the
shape is the same [(chunk_id, score), ...]).

Pure functions over the loaded index — no I/O at query time.
"""

from .bm25 import rare_hits
from .bm25 import score as bm25_score
from .rrf import fuse


def retrieve(
    index: dict,
    chunks_by_id: dict[str, dict],
    question: str,
    dense: list[tuple[str, float]],
    top_k: int = 10,
) -> list[dict]:
    """BM25 + dense lists → fused evidence pack of chunk dicts, ranked.

    Rare-entity guarantee: a query token present in only a few docs names
    a substance; the chunks containing it force into the pool because
    fusion can't be trusted to rank entity evidence over topic-phrasing
    matches — measured: 'laxogenin' retrieved breath-holding studies, not
    laxogenin ones.

    Chunks marked superseded_by stay in the pack — the answer layer decides
    whether to prefer or flag them (hiding them here would silently lose
    historical evidence)."""
    sparse = bm25_score(index, question, top_k=20)
    fused = fuse(sparse, dense, top_k=top_k)
    ids = [cid for cid, _ in fused]
    for cid in reversed(rare_hits(index, question)):
        if cid not in ids:
            ids.insert(0, cid)
    ids = ids[:top_k]
    return [
        {"rrf": dict(fused).get(cid, 0.0), **chunks_by_id[cid]}
        for cid in ids
        if cid in chunks_by_id
    ]


if __name__ == "__main__":
    # Dev/eval glue: the JS side embeds whatever text we hand it, so the
    # keyword transform stays single-sourced here.
    #   python -m supplements_rag.query --dense-query "what does magnesium do"
    #       → prints "information about magnesium"  (feed to query.mjs)
    #   python -m supplements_rag.query --dense-map evals/golden.jsonl
    #       → writes data/dense-queries.json {row: "information about …"}
    import argparse
    import json
    from pathlib import Path

    from .bm25 import dense_query

    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-query")
    ap.add_argument("--dense-map", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/dense-queries.json"))
    a = ap.parse_args()
    if a.dense_query:
        print(dense_query(a.dense_query))
    elif a.dense_map:
        rows = [json.loads(l) for l in a.dense_map.open()]
        a.out.write_text(
            json.dumps({str(i): dense_query(r["question"]) for i, r in enumerate(rows)})
        )
        print(f"{len(rows)} dense queries → {a.out}")
