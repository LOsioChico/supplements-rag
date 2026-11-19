# data/ — build artifacts (gitignored, regenerable)

Everything here is produced by the pipeline — nothing is committed, nothing is
hand-edited. **None of these files ship to production as-is:** they are build
intermediates. `worker/scripts/load_d1.mjs` loads them into D1 — `chunks`
(chunk_id-keyed rows), `bm25_postings` (term-indexed, sharded because stopword
postings exceed D1's statement cap), `bm25_meta` (corpus stats) — and vectors
live in Vectorize; the Worker queries those, never these files. A fourth D1
table, `runs`, is written at runtime not loaded — one row per ask holding the
full event stream, the prod equivalent of `runs.jsonl` (stub-inserted when the
stream opens, updated when it ends; abandoned runs keep the stub). The
public-facing id is `rid` (random hex) — the autoincrement `id` stays
internal so stored questions can't be enumerated via `/v1/run/{rid}`.

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
| `runs.jsonl` | `serve.py` (runtime, not the pipeline) | One line per visitor ask — the full event stream for completed, crashed, and abandoned runs alike; question text masked |

Measured on the full corpus (17,696 chunks): corpus ≈ 86MB, chunks ≈ 84MB,
bm25 ≈ 35MB (~2KB/doc — term frequencies dominate), embed-ledger ≈ 190KB.
