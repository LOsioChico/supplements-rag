"""Ingest contract tests — run against a real efetch fixture, no network."""

import xml.etree.ElementTree as ET
from pathlib import Path

from supplements_rag.ingest import (
    EVIDENCE_PUBTYPES,
    load_ledger,
    month_windows,
    normalize,
)

FIXTURE = Path(__file__).parent / "fixtures" / "efetch_sample.xml"


def test_month_windows_covers_range_and_format():
    wins = month_windows(2015, 2016, end_month=3)
    assert wins[0] == (
        "2015/01/01",
        "2015/02/01",
    )  # only slash-format pdat works on PubMed
    assert wins[-1] == ("2016/03/01", "2016/04/01")
    assert len(wins) == 12 + 3
    # contiguous — no gaps between windows
    assert all(wins[i][1] == wins[i + 1][0] for i in range(len(wins) - 1))


def test_normalize_extracts_real_record():
    recs = list(ET.parse(FIXTURE).getroot().iter("PubmedArticle"))
    assert len(recs) == 3
    r = normalize(recs[0])
    assert r["pmid"] == "39872318"
    assert r["source"] == "pubmed"
    assert r["title"] and r["has_abstract"] and r["abstract"]
    assert set(r["pubtypes"]) & set(EVIDENCE_PUBTYPES)
    assert r["mesh"] and all("descriptor" in m for m in r["mesh"])
    assert r["pub_date"].startswith("20")
    assert r["fetched_at"]


def test_normalize_every_fixture_row():
    for rec in ET.parse(FIXTURE).getroot().iter("PubmedArticle"):
        r = normalize(rec)
        assert r["pmid"] and r["title"], f"empty core field in {r['pmid']}"


def test_ledger_roundtrip(tmp_path):
    p = tmp_path / "ledger.json"
    led = load_ledger(p)
    assert led["pmids"] == [] and led["windows"] == {}
    led["pmids"] = ["1", "2"]
    p.write_text(__import__("json").dumps(led))
    assert load_ledger(p)["pmids"] == ["1", "2"]
