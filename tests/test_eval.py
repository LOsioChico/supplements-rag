"""Eval-harness scoring tests — pure functions, no fixtures."""

import run


def test_hits_at_k_boundary():
    assert run.hits_at_k(["a", "b", "c"], ["b"], 2)
    assert not run.hits_at_k(["a", "b", "c"], ["c"], 2)
    assert run.hits_at_k(["a", "b", "c"], ["c"], 3)


def test_summarize_aggregates_per_category():
    results = [
        {
            "category": "answerable",
            "retention@5": True,
            "retention@10": True,
            "bm25_top": 20.0,
            "dense_top": 0.5,
            "advice_pattern": False,
        },
        {
            "category": "answerable",
            "retention@5": False,
            "retention@10": True,
            "bm25_top": 10.0,
            "dense_top": 0.4,
            "advice_pattern": False,
        },
        {
            "category": "abstain",
            "retention@5": None,
            "retention@10": None,
            "bm25_top": 5.0,
            "dense_top": 0.3,
            "advice_pattern": False,
        },
    ]
    s = run.summarize(results)
    assert s["answerable"]["n"] == 2
    assert s["answerable"]["retention@5"] == 0.5
    assert s["answerable"]["retention@10"] == 1.0
    assert s["abstain"]["retention@5"] is None
    assert s["abstain"]["bm25_top_range"] == [5.0, 5.0]


def test_advice_pattern_catches_gate_vocabulary():
    assert run.ADVICE_RE.search("What is the dose of Chondroitin?")
    assert run.ADVICE_RE.search("does it interact with medications")
    assert not run.ADVICE_RE.search("How does magnesium work in the body")
