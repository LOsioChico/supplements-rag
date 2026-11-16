# data/ — build artifacts (gitignored, regenerable)

Everything here is produced by the pipeline — nothing is committed, nothing is
hand-edited. **None of these files ship to production as-is:** they are build
intermediates. At deploy time chunks load into D1 (pmid-keyed rows), the BM25
postings into a D1 table (term-indexed, per-term lookups), and vectors into
Vectorize — the worker queries those, never these files.

| File | Produced by | Contents |
| --- | --- | --- |
| `corpus.jsonl` | `ingest.py` | One normalized PubMed record per line: pmid, title, abstract + `abstract_sections`, journal, pub_date, pubtypes, mesh, doi, `fetched_at` |
| `ledger.json` | `ingest.py` | Checkpoint ledger — which monthly windows completed + every fetched PMID. This is what makes re-runs incremental (freshness mechanism) |
| `chunks.jsonl` | `chunk.py` | One retrievable unit per line (currently 1 chunk = 1 abstract, per ADR-003) |
| `bm25.json` | `bm25.py` | Serialized keyword index: per-doc term frequencies, document frequencies, average doc length |
| `embed-ledger.json` | `embed.mjs` | Checkpoint ledger — chunk ids already embedded + upserted to Vectorize. Re-runs skip them (same idempotency idea as `ledger.json`) |
| `dense.json` | `query.mjs` | One query's Vectorize response — local dev-test glue for `query.py`. In production the worker fetches the dense list inline; no file |
| `dense-golden.json` | `embed-questions.mjs` | Per-golden-row dense lists (120 × top-20) — eval + re-proposal input |
| `verdicts.json` | `verify.py` | Per-(row, pmid) judge scores 0–1 on the golden set — checkpointed, resumable, re-judges non-float values |
| `verdicts-pool.json` | `verify.py --rows evals/admitted.jsonl` | Same, on the full 4,026-row admitted pool (~10k judge calls) |
| `dense-queries.json` | `query.py --dense-map` | Keyword-form embedding text per eval row (`"information about {kw}"`) — keeps the dense transform single-sourced in Python |
| `pool-verify*.log` | `verify.py` (stderr) | Pool-pass progress logs — regenerate, ignore |

Measured on the full corpus (17,696 chunks): corpus ≈ 86MB, chunks ≈ 84MB,
bm25 ≈ 35MB (~2KB/doc — term frequencies dominate), embed-ledger ≈ 190KB.
