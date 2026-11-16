"""Golden-set admission tests — synthetic index + inline seeds, no files."""

import admit

from supplements_rag.bm25 import build_index

INDEX = build_index(
    [
        {
            "chunk_id": "1",
            "title": "Magnesium insomnia RCT",
            "text": "magnesium supplementation improved insomnia sleep quality",
        },
        {
            "chunk_id": "2",
            "title": "Arnica bruising trial",
            "text": "arnica gel did not reduce bruising versus placebo",
        },
    ]
)


def test_advice_admitted_without_coverage_check():
    seeds = [
        {
            "supplement": "Unobtanium",
            "qtype": "dose",
            "question": "What is the dose of Unobtanium?",
        }
    ]
    admitted, rejected = admit.admit_medquad(INDEX, seeds, 0.5)
    assert admitted[0]["expected_behavior"] == "answer_or_decline"
    assert admitted[0]["evidence_pmids"] == [] and not rejected


def test_medquad_answerable_requires_coverage():
    seeds = [
        {
            "supplement": "Magnesium",
            "qtype": "information",
            "question": "What is Magnesium?",
        },
        {
            "supplement": "Unobtanium-zzz",
            "qtype": "information",
            "question": "What is Unobtanium-zzz?",
        },
    ]
    admitted, rejected = admit.admit_medquad(INDEX, seeds, 0.5)
    assert len(admitted) == 1 and admitted[0]["evidence_pmids"] == ["1"]
    assert rejected[0]["reason"] == "no_corpus_coverage"


def test_ineffective_labels_classify_as_negative_not_answerable():
    """Regression: 'effective_for' substring-matches 'ineffective_for' —
    label checks must be exact set membership."""
    seeds = [
        {"substance": "arnica", "condition": "bruising", "label": lab}
        for lab in (
            "ineffective_for",
            "likely_ineffective_for",
            "possibly_ineffective_for",
        )
    ]
    admitted, _ = admit.admit_suppllment(INDEX, seeds, 0.5)
    assert all(r["category"] == "negative" for r in admitted)
    assert all(r["expected_behavior"] == "answer_with_citations" for r in admitted)


def test_insufficient_evidence_abstains_only_when_uncovered():
    seeds = [
        {
            "substance": "magnesium",
            "condition": "insomnia",
            "label": "insufficient_evidence",
        },
        {
            "substance": "unobtanium-zzz",
            "condition": "zilch",
            "label": "insufficient_evidence",
        },
    ]
    admitted, rejected = admit.admit_suppllment(INDEX, seeds, 0.5)
    assert len(admitted) == 1 and admitted[0]["category"] == "abstain"
    assert admitted[0]["evidence_pmids"] == []
    assert rejected[0]["reason"] == "labeled_insufficient_but_covered"


def test_golden_sample_respects_quotas_and_is_deterministic():
    admitted = [
        {"category": "answerable", "source": "suppllment"} for _ in range(100)
    ] + [{"category": "advice", "source": "medquad"} for _ in range(100)]
    g1 = admit.golden_sample(admitted, 42)
    g2 = admit.golden_sample(admitted, 42)
    cats = {}
    for r in g1:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    assert g1 == g2 and cats == {"answerable": 50, "advice": 25}
