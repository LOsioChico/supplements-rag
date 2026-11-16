"""PubMed ingest: date-sliced esearch → efetch XML → normalized JSONL + checkpointed ledger.

Verified constraints (2026-11-15, live):
- esearch/efetch history paging dies at retstart 9,999 — so the query is sliced
  into per-month windows, each far under the cap.
- Evidence-grade query (measured): ("dietary supplements"[mesh] OR
  "dietary supplement*"[tiab]) AND YYYYMMDD:YYYYMMDD[pdat] AND
  (RCT | meta-analysis | systematic review | controlled clinical trial)[pt]
  → 17,698 records over 2015-01→2026-12.
- Rate limits: 3 req/s anonymous, 10 req/s with NCBI_API_KEY.
- Freshness: the ledger tracks fetched PMIDs per window; re-running a window
  fetches only new articles (MeSH lag re-catches automatically via [tiab]).

efetch XML shape (verified on live records):
  PubmedArticle → MedlineCitation{PMID, Article{ArticleTitle,
  Abstract/AbstractText[@Label], AuthorList, Journal{Title, JournalIssue/PubDate}},
  MeshHeadingList/MeshHeading{DescriptorName, QualifierName}},
  PubmedData/ArticleIdList/ArticleId[@IdType]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EVIDENCE_PUBTYPES = (
    "Randomized Controlled Trial",
    "Meta-Analysis",
    "Systematic Review",
    "Controlled Clinical Trial",
)
QUERY_TEMPLATE = (
    '("dietary supplements"[mesh] OR "dietary supplement*"[tiab])'
    " AND {start} : {end}[pdat]"
    " AND ({pubtypes})"
)
_PUBTYPE_Q = " OR ".join(f'"{p}"[pt]' for p in EVIDENCE_PUBTYPES)


def month_windows(
    start_year: int, end_year: int, end_month: int = 12
) -> list[tuple[str, str]]:
    """[(start,end)] pdat range pairs, one per month — keeps every esearch under the 9,999 cap.
    Verified: PubMed accepts only YYYY/MM/DD:YYYY/MM/DD[pdat]; YYYYMMDD silently returns 0."""
    out = []
    for y in range(start_year, end_year + 1):
        last_m = end_month if y == end_year else 12
        for m in range(1, last_m + 1):
            ny, nm = (y, m + 1) if m < 12 else (y + 1, 1)
            out.append((f"{y}/{m:02d}/01", f"{ny}/{nm:02d}/01"))
    return out


class Eutils:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.delay = 0.11 if api_key else 0.35

    def get(self, endpoint: str, **params) -> bytes:
        params.setdefault("retmode", "json" if endpoint == "esearch" else "xml")
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{EUTILS}/{endpoint}.fcgi?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    time.sleep(self.delay)
                    return r.read()
            except urllib.error.HTTPError as e:
                wait = 2**attempt + (int(e.headers.get("Retry-After") or 0))
                if attempt == 4 or e.code not in (429, 500, 502, 503):
                    raise
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def esearch(self, start: str, end: str) -> tuple[int, list[str]]:
        d = json.loads(
            self.get(
                "esearch",
                db="pubmed",
                term=QUERY_TEMPLATE.format(start=start, end=end, pubtypes=_PUBTYPE_Q),
                retmax=9999,
            )
        )["esearchresult"]
        return int(d["count"]), d["idlist"]

    def efetch(self, pmids: list[str]) -> ET.Element:
        return ET.fromstring(
            self.get("efetch", db="pubmed", id=",".join(pmids), rettype="xml")
        )


def _abstract_sections(article: ET.Element) -> list[dict]:
    """Structured abstracts keep their labels (Objective/Methods/Results/…) in metadata — not split."""
    secs = []
    for t in article.iter("AbstractText"):
        secs.append(
            {"label": t.get("Label") or "", "text": "".join(t.itertext()).strip()}
        )
    return secs


def _pubdate(article: ET.Element) -> str:
    pd = article.find(".//JournalIssue/PubDate")
    if pd is None:
        return ""
    y, m, d = (pd.findtext(x) or "" for x in ("Year", "Month", "Day"))
    if not y:
        return pd.findtext("MedlineDate") or ""
    months = {
        n: f"{i:02d}"
        for i, n in enumerate(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            1,
        )
    }
    mm = months.get(m[:3], m if m.isdigit() else "01")
    return f"{y}-{mm}-{d or '01'}"


def normalize(rec: ET.Element) -> dict:
    """Source-agnostic record shape: flat fields + provenance."""
    med = rec.find("MedlineCitation")
    art = med.find("Article") if med is not None else None
    if med is None or art is None:
        raise ValueError("malformed PubmedArticle: missing citation/article")
    sections = _abstract_sections(art)
    doi = ""
    for aid in rec.iter("ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text or ""
    title_el = art.find("ArticleTitle")
    return {
        "pmid": med.findtext("PMID"),
        "title": "".join(title_el.itertext()).strip() if title_el is not None else "",
        "abstract": " ".join(s["text"] for s in sections),
        "abstract_sections": sections,
        "has_abstract": bool(sections),
        "authors": [
            " ".join(filter(None, [a.findtext("ForeName"), a.findtext("LastName")]))
            for a in art.iter("Author")
            if a.findtext("LastName")
        ],
        "journal": art.findtext(".//Journal/Title") or "",
        "pub_date": _pubdate(art),
        "pubtypes": [p.text for p in art.iter("PublicationType") if p.text],
        "mesh": [
            {
                "descriptor": h.findtext("DescriptorName") or "",
                "qualifiers": [q.text for q in h.iter("QualifierName") if q.text],
            }
            for h in med.iter("MeshHeading")
        ],
        "doi": doi,
        "source": "pubmed",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def load_ledger(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "query": "evidence-grade supplements slice",
        "windows": {},
        "pmids": [],
        "updated_at": "",
    }


def run(
    start_year: int,
    end_year: int,
    end_month: int,
    out_path: Path,
    ledger_path: Path,
    batch: int = 200,
    max_windows: int = 0,
) -> dict:
    api_key = os.environ.get("NCBI_API_KEY") or _env_file()
    eu = Eutils(api_key)
    ledger = load_ledger(ledger_path)
    seen = set(ledger["pmids"])
    fetched = skipped_windows = 0

    with out_path.open("a") as out:
        for w_start, w_end in month_windows(start_year, end_year, end_month):
            name = w_start[:7].replace("/", "-")
            if name in ledger["windows"]:
                skipped_windows += 1
                continue
            count, pmids = eu.esearch(w_start, w_end)
            new = [p for p in pmids if p not in seen]
            for i in range(0, len(new), batch):
                for rec in eu.efetch(new[i : i + batch]).iter("PubmedArticle"):
                    out.write(json.dumps(normalize(rec), ensure_ascii=False) + "\n")
                    fetched += 1
            seen.update(pmids)
            ledger["windows"][name] = {"count": count, "new": len(new)}
            ledger["pmids"] = sorted(seen, key=int)
            ledger["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ledger_path.write_text(json.dumps(ledger, indent=1))
            print(f"{name}: {count} in slice, {len(new)} new → fetched")
            if max_windows and len(ledger["windows"]) >= max_windows:
                break
    return {
        "fetched": fetched,
        "windows_done": len(ledger["windows"]),
        "windows_skipped": skipped_windows,
        "total_pmids": len(ledger["pmids"]),
    }


def _env_file() -> str:
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("NCBI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ingest the evidence-grade supplements slice from PubMed."
    )
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--end-month", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--ledger", type=Path, default=Path("data/ledger.json"))
    ap.add_argument(
        "--batch", type=int, default=200, help="efetch batch size (max ~200)"
    )
    ap.add_argument(
        "--max-windows", type=int, default=0, help="0 = all (smoke-test with e.g. 1)"
    )
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            run(
                a.start_year,
                a.end_year,
                a.end_month,
                a.out,
                a.ledger,
                a.batch,
                a.max_windows,
            ),
            indent=1,
        )
    )
