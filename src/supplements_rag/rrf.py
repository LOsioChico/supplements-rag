"""Reciprocal Rank Fusion — merges BM25 and vector result lists.

RRF scores each document as sum over lists of 1/(k + rank). Ranks, not raw
scores, are combined — BM25 scores and cosine similarities live on
incompatible scales, so fusing raw numbers would let whichever list has the
bigger score range dominate. k=60 is the canonical constant from the
original paper (Cormack et al., 2009).

Pure functions: ranked lists in → one fused ranked list out.
"""

K = 60


def fuse(
    *ranked_lists: list[tuple[str, float]], top_k: int = 20
) -> list[tuple[str, float]]:
    """ranked lists [(chunk_id, _ignored_score), ...] → fused [(id, rrf)]."""
    rrf = {}
    for lst in ranked_lists:
        for rank, (cid, _) in enumerate(lst):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K + rank + 1)
    return sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
