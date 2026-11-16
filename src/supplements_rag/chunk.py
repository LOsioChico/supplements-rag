"""Chunk normalized PubMed records into retrievable evidence units.

Decision (see docs/decisions.md): the abstract is the atomic chunk — PubMed
only exposes abstracts, so there is nothing larger to split. Structured-abstract
section labels (BACKGROUND, METHODS, ...) are kept as metadata for display and
prefixed into the embedding text so the retriever sees the same shape a reader
sees.

Pure functions, functional style — corpus line in, chunk dict out.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def chunk_record(rec: dict) -> dict:
    """One normalized record → one chunk. `text` is what gets embedded."""
    sections = rec.get("abstract_sections") or []
    if sections:
        text = "\n".join(f"{s['label']}: {s['text']}" for s in sections)
    else:
        text = rec.get("abstract") or ""
    return {
        "chunk_id": str(rec["pmid"]),
        "pmid": rec["pmid"],
        "title": rec.get("title") or "",
        "text": text,
        "sections": sections,
        "has_abstract": rec.get("has_abstract", bool(text)),
        "journal": rec.get("journal") or "",
        "pub_date": rec.get("pub_date") or "",
        "pubtypes": rec.get("pubtypes") or [],
        "mesh": rec.get("mesh") or [],
        "doi": rec.get("doi") or "",
    }


def mark_superseded(chunks: list[dict]) -> tuple[list[dict], dict]:
    """Same normalized title + different text = an updated living review
    (Cochrane reviews republish over years). The older versions stay
    retrievable but get `superseded_by` — the answer layer must prefer and
    flag the newest evidence. Identical text = true duplicate → dropped.

    Returns (kept chunks, counters)."""
    by_title = defaultdict(list)
    for c in chunks:
        by_title[c["title"].strip().lower()].append(c)
    kept, dropped, superseded = [], 0, 0
    for group in by_title.values():
        if len(group) == 1:
            kept.extend(group)
            continue
        group.sort(key=lambda c: c["pub_date"], reverse=True)  # newest first
        survivors = [group[0]]
        seen = {hashlib.md5(group[0]["text"].encode()).hexdigest()}
        for g in group[1:]:
            h = hashlib.md5(g["text"].encode()).hexdigest()
            if h in seen:
                dropped += 1  # identical text — the second PMID adds nothing
                continue
            seen.add(h)
            g["superseded_by"] = group[0]["pmid"]
            superseded += 1
            survivors.append(g)
        kept.extend(survivors)
    return kept, {"dropped_dupes": dropped, "marked_superseded": superseded}


def run(corpus_path: Path, out_path: Path) -> dict:
    """Stream corpus.jsonl → chunks.jsonl. Returns counters for the ledger."""
    chunks, no_abstract = [], 0
    with corpus_path.open() as src:
        for line in src:
            if not line.strip():
                continue
            chunk = chunk_record(json.loads(line))
            chunks.append(chunk)
            no_abstract += not chunk["has_abstract"]
    chunks, stats = mark_superseded(chunks)
    with out_path.open("w") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    return {"chunks": len(chunks), "no_abstract": no_abstract, **stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunk corpus.jsonl into evidence units.")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/chunks.jsonl"))
    a = ap.parse_args()
    print(json.dumps(run(a.corpus, a.out), indent=1))
