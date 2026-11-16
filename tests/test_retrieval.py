"""Chunk + BM25 tests — pure functions, no fixtures or network needed."""

from supplements_rag.bm25 import (
    build_index,
    dense_query,
    rare_hits,
    score,
    tokenize,
)
from supplements_rag.chunk import chunk_record, mark_superseded
from supplements_rag.query import retrieve
from supplements_rag.rrf import fuse


def _rec(**kw):
    base = {
        "pmid": 123,
        "title": "Magnesium and sleep",
        "abstract": "Plain abstract text.",
        "abstract_sections": [],
        "has_abstract": True,
    }
    return {**base, **kw}


def test_chunk_record_structured_sections():
    rec = _rec(
        abstract_sections=[
            {"label": "BACKGROUND", "text": "Why."},
            {"label": "RESULTS", "text": "What."},
        ]
    )
    c = chunk_record(rec)
    assert c["text"] == "BACKGROUND: Why.\nRESULTS: What."
    assert c["chunk_id"] == "123" and c["has_abstract"]


def test_chunk_record_falls_back_to_abstract_and_flags_missing():
    assert "Plain abstract" in chunk_record(_rec())["text"]
    assert not chunk_record(_rec(abstract="", has_abstract=False))["has_abstract"]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Mg, Sleep! 5-HTP") == ["mg", "sleep", "5", "htp"]


def test_tokenize_strips_accents():
    assert tokenize("ácido fólico") == ["acido", "folico"]


def test_dense_query_strips_scaffolding():
    assert (
        dense_query("what does magnesium do for sleep?")
        == "information about magnesium sleep"
    )
    # all-STOP questions fall back to full tokens — never an empty vector
    assert dense_query("what is it") == "information about what is it"


def test_rare_hits_force_entity_chunks_into_pool():
    # Corpus big enough that condition words aren't rare (df > rare_max):
    # only 'laxogenin' is a true entity signal — its doc must surface even
    # when it carries just one rare token vs others' many.
    chunks = [
        {
            "chunk_id": "a",
            "title": "laxogenin study",
            "text": "laxogenin pharmacokinetics",
        },
    ] + [
        {
            "chunk_id": chr(98 + i),
            "title": f"breath study {i}",
            "text": "breath holding spells children supplementation",
        }
        for i in range(4)
    ]
    idx = build_index(chunks)
    hits = rare_hits(idx, "is laxogenin effective for breath holding")
    assert hits == ["a"]


def test_retrieve_injects_rare_entity_chunk():
    chunks = [
        {
            "chunk_id": "a",
            "pmid": "1",
            "title": "laxogenin study",
            "text": "laxogenin pharmacokinetics",
            "sections": [],
            "has_abstract": True,
        },
        {
            "chunk_id": "b",
            "pmid": "2",
            "title": "breath holding",
            "text": "breath holding spells",
            "sections": [],
            "has_abstract": True,
        },
        {
            "chunk_id": "c",
            "pmid": "3",
            "title": "unrelated",
            "text": "completely different words",
            "sections": [],
            "has_abstract": True,
        },
    ]
    idx = build_index(chunks)
    by_id = {str(c["chunk_id"]): c for c in chunks}
    # dense buries the entity chunk — rare-hit coverage still surfaces it
    pack = retrieve(
        idx,
        by_id,
        "is laxogenin effective for breath holding",
        dense=[("b", 0.9), ("c", 0.8)],
        top_k=2,
    )
    assert pack[0]["pmid"] == "1" or any(p["pmid"] == "1" for p in pack)


def test_bm25_ranks_relevant_doc_first():
    chunks = [
        {
            "chunk_id": "a",
            "title": "Magnesium glycinate sleep RCT",
            "text": "magnesium supplementation improved sleep quality",
        },
        {
            "chunk_id": "b",
            "title": "Vitamin C and colds",
            "text": "ascorbic acid did not shorten colds",
        },
    ]
    idx = build_index(chunks)
    ranked = score(idx, "magnesium sleep")
    assert ranked[0][0] == "a" and len(ranked) == 1 or ranked[0][0] == "a"


def test_bm25_empty_query_scores_zero():
    idx = build_index([{"chunk_id": "a", "title": "x", "text": "magnesium"}])
    assert score(idx, "unrelated-term-zzz") == []


def test_rrf_fuses_by_rank_not_score():
    bm25 = [("a", 30.0), ("b", 2.0)]
    dense = [("b", 0.91), ("c", 0.88)]
    fused = fuse(bm25, dense)
    # b is rank-2 in bm25 and rank-1 in dense → beats a (only in one list)
    assert fused[0][0] == "b" and [c for c, _ in fused] == ["b", "a", "c"]


def test_mark_superseded_flags_updates_and_drops_dupes():
    old = {
        "chunk_id": "1",
        "pmid": "1",
        "title": "Zinc for colds",
        "text": "old text",
        "pub_date": "2015",
        "sections": [],
        "has_abstract": True,
    }
    new = {
        **old,
        "chunk_id": "2",
        "pmid": "2",
        "text": "updated text",
        "pub_date": "2020",
    }
    dupe = {**old, "chunk_id": "3", "pmid": "3", "pub_date": "2015"}
    kept, stats = mark_superseded([old, new, dupe])
    assert len(kept) == 2 and stats == {"dropped_dupes": 1, "marked_superseded": 1}
    assert kept[0]["pmid"] == "2"  # newest first
    assert kept[1]["superseded_by"] == "2"


def test_retrieve_fuses_sparse_and_dense_and_keeps_superseded():
    chunks = [
        {
            "chunk_id": "a",
            "pmid": "1",
            "title": "magnesium sleep RCT",
            "text": "magnesium supplementation improved sleep",
            "sections": [],
            "has_abstract": True,
        },
        {
            "chunk_id": "b",
            "pmid": "2",
            "title": "old review",
            "text": "unrelated words",
            "sections": [],
            "has_abstract": True,
            "superseded_by": "9",
        },
    ]
    idx = build_index(chunks)
    by_id = {str(c["chunk_id"]): c for c in chunks}
    dense = [("b", 0.9), ("a", 0.8)]
    pack = retrieve(idx, by_id, "magnesium sleep", dense, top_k=5)
    ids = [p["pmid"] for p in pack]
    assert "1" in ids and "2" in ids  # superseded stays, flagged not hidden
