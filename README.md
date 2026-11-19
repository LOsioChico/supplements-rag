# supplements-rag

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![python: 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Ask "does creatine help build muscle?" → get a cited answer built only from real PubMed abstracts, a designed abstention when the evidence isn't there, or a decline when it's personal medical advice. Grounded Q&A over a bounded dietary-supplement corpus — built to be inspectable, not just impressive.

**Live demo:** [luisosio.dev](https://luisosio.dev) — the supplements box runs this exact pipeline (`GET /sr/v1/ask?q=`); every run's trace is readable back at `/sr/v1/run/{rid}`.

## What a question looks like

```
$ python -m supplements_rag.pipeline "does creatine help build muscle"
{
  "status": "answered",
  "direction": "mixed",
  "verdict": "Creatine helps increase muscle strength … but the evidence
    for actual muscle growth is less direct and mixed.",
  "claims": [
    {"text": "A meta-analysis found creatine significantly improves muscle
      strength; untrained individuals improved more than trained ones",
      "ids": ["41328071"]},
    {"text": "20 g/d loading did NOT prevent muscle loss during leg
      immobilization", "ids": ["28054322"]},
    "…7 claims total"
  ],
  "window": {"min": 2018, "max": 2024},
  "trace": {"usage": {"calls": 61, "cost_usd": 0.000735}, "ms": 20054}
}
```

Every number must literally appear in its cited source (doses, p-values — checked deterministically), every claim gets a judge support score ≥0.7, and the verdict is checked against the combined evidence — a failed check marks its sentence `unverified`; only unparseable output abstains. The `trace` carries the full reasoning (gate intent, timings, judge scores, token/cost ledger) — it is the contract the demo streams; `window` bounds answers to the cited studies' publication years.

## The finding that shaped the design

**Coverage ≠ evidence.** BM25 admits eval questions by lexical overlap — and on 4,026 admitted rows, the judge confirmed real evidence for only 24%. **52% of everything that looked covered has no supporting abstract at all.** A system that answers whatever it retrieves fabricates on half its input — so this one verifies every claim before the answer is trusted, and abstains by design.

Measured on the golden set (126 rows — the tuning set; the v8 numbers below ran at 120, before six authored first-person rows) **and a 199-row held-out set it never saw** (`evals/heldout.jsonl`, disjoint questions — the anti-overfit check):

- **91%** of verified-evidence rows answered with citations (golden, 360 runs across `--repeats 3` — **0 status flips**); held-out: **92%**
- **100%** of refuted rows safely refused; **0** dangerous answers in 360 runs
- Rows the corpus can't support are refused or marked `unclear` with an explicit `missing` list

Full eval methodology and per-run history: [evals/README.md](evals/README.md).

## The pipeline

```
corpus build (offline, done):
  PubMed xml → ingest → chunk → bm25 index
                     chunk → embed → Vectorize

question path (per ask, done):
  question → gate → retrieve → generate → verify → outcome
             │       (BM25 + dense, RRF-fused + rare-entity)
             ├─ injection → refuse          outcomes:
             ├─ advice → decline            answered | unclear | abstained |
             └─ unrelated → off_topic       declined | refused | off_topic |
                                          unavailable (fail-closed)
```

Every stage is one small stdlib-Python module with its own CLI — run any stage alone, inspect its output, re-run it idempotently. `dependencies = []`: no framework, no vector DB client, no ORM — the retrieval math is ~50 lines you can read.

## Corpus

The PubMed query that built the corpus:

```
("dietary supplements"[mesh] OR "dietary supplement*"[tiab])
  AND 2015:2026[pdat]
  AND ("Randomized Controlled Trial"[pt] OR "Meta-Analysis"[pt]
       OR "Systematic Review"[pt] OR "Controlled Clinical Trial"[pt])
```

`[mesh]` = PubMed's subject-index terms · `[tiab]` = title/abstract words · `[pdat]` = publication date · `[pt]` = publication type. Result: **17,698 abstracts → 17,696 chunks** after dedupe (measured, not estimated).

The `tiab` clause exists because MeSH indexing lags ~15–17% in recent years — recent papers would be invisible without it. MeSH associations also admit ~0.1% complementary-intervention studies — supplement-*centered*, not all alternative medicine.

## Run it

```bash
pip install -e .[dev]                  # package + pytest/ruff/pyrefly

# verify the code — no data or network needed
pytest                                 # 60 tests
ruff check . && ruff format --check . && pyrefly check

# ask a question — needs built data/ + VLLM_SR_API_KEY
# (embed + vector search run on Cloudflare; the node script writes the dense matches)
node scripts/query.mjs "does magnesium help sleep"   # → data/dense.json
python -m supplements_rag.pipeline "does magnesium help sleep"   # answer + trace
python -m supplements_rag.gate "does magnesium help sleep"       # or one stage alone

# rebuild the corpus — the long path, PubMed rate limits apply
python -m supplements_rag.ingest    # monthly windows → data/corpus.jsonl
python -m supplements_rag.chunk     # → data/chunks.jsonl
python -m supplements_rag.bm25      # → data/bm25.json
node scripts/embed.mjs              # chunk vectors → Vectorize (needs wrangler auth)

# measure it — the eval track
python evals/admit.py               # seed questions → {admitted,rejected,golden}.jsonl
node scripts/embed-questions.mjs    # golden questions → dense lists
python evals/run.py                 # retrieval metrics → evals/report.json
python evals/eval_pipeline.py       # golden × pipeline → outcome matrix
```

Ingest is resumable via `data/ledger.json` (checkpoints per window + PMID — re-runs fetch only new articles). Keys in `.env` (see `.env.example`): `NCBI_API_KEY` raises ingest to 10 req/s, `VLLM_SR_API_KEY` is required by the pipeline.

## Honest status

- **Built + verified live:** the answering path — gate (injection + intent), hybrid retrieval (BM25 + dense + rare-entity rescue), strict contract, deterministic + semantic verification, abstain/decline/refuse. `answer()` buffers (nothing leaves before verify — evals use it); `answer_events()` streams the same checks — extractive emits extraction progress + per-claim judge stamps, the opt-in writer lane streams tokens (ADR-013). `status` carries `direction`, `verdict_kind`, checked `points` (ADR-015)
- **Built + verified live:** the same SSE contract in two deploys — local `python -m supplements_rag.serve` (:8778, `GET /v1/ask?q=` + `scripts/dense-server.mjs` sidecar) and the Worker (`worker/src/worker.py`, FastAPI/`workers.asgi`, `GET /sr/v1/ask?q=` on `luisosio.dev`: D1 chunks+BM25, Vectorize, `httpx2` judge/writer, per-ask `runs` table — the `run` event's public id readable at `/sr/v1/run/{rid}`; ADR-019)
- **Built + measured:** the extractive lane (`extract.py`, `EXTRACTIVE=1`) — verbatim quotes instead of generated claims (chunk scores → sentence scores → quotes ≥0.6, same checks after) plus an entity check for rare/absent substances (ADR-016). Same 30-row golden slice: 19/30 vs the writer's 9/30, median 3.5s vs 8.6s (ADR-014)
- **Not built yet:** freshness cron (incremental re-ingest on a schedule — the ledger mechanism exists, the trigger doesn't)
- **Known gaps, stated plainly:** [KNOWN_GAPS.md](KNOWN_GAPS.md) — what's unverified, deferred, or an assumption

## What this does NOT do

- Not medical advice — "should I take X?" gets a designed decline, not an answer.
- Abstracts, not full text (PMC open-access full text is the documented v2 path).
- The system searches this bounded corpus — not "all medical knowledge".

## Layout

- `src/supplements_rag/` — one module per stage, flat by design: `ingest`/`chunk`/`bm25` (acquire), `query`/`rrf` (retrieve), `gate`/`generate`/`extract`/`verify`/`contract`/`judge` (answer), `pipeline` (orchestrate), `serve` (local SSE)
- `evals/` — golden set + admission pipeline ([sources, counts, limits](evals/README.md)) · `tests/` — no-network unit tests · `data/` — gitignored artifacts ([what each file is](data/README.md))
- `docs/decisions.md` — ADRs with measured numbers · `KNOWN_GAPS.md` — the honest register · `AGENTS.md` — repo conventions

## Contributing

Solo project, not set up for external PRs — issues with evidence (a question that misbehaves + its run id) are welcome. Conventions live in `AGENTS.md`.

## License

MIT — see [LICENSE](LICENSE).
