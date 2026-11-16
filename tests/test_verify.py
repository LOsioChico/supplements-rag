"""Semantic-verification tests — band logic + verdict application, no network."""

import verify


def test_status_of_bands():
    assert verify.status_of({"1": 0.71}) == "semantic_verified"
    assert verify.status_of({"1": 0.2, "2": 0.5}) == "semantic_partial"
    assert verify.status_of({"1": 0.1, "2": 0.39}) == "semantic_refuted"
    assert verify.status_of({"1": "error: timeout"}) == "verification_error"


def test_apply_verdicts_updates_rows_and_skips_evidenceless():
    rows = [
        {
            "category": "answerable",
            "evidence_pmids": ["1", "2"],
            "verification": "retrieval_only",
        },
        {"category": "abstain", "evidence_pmids": [], "verification": "retrieval_only"},
    ]
    out = verify.apply_verdicts(rows, {"0": {"1": 0.9, "2": 0.1}})
    assert out[0]["verification"] == "semantic_verified"
    assert out[0]["evidence_verdicts"] == {"1": 0.9, "2": 0.1}
    assert out[1]["verification"] == "retrieval_only"  # untouched
