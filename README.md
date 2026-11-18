# supplements-rag

Ask "does creatine help build muscle?" → get a cited answer built only from real PubMed abstracts, a designed abstention when the evidence isn't there, or a decline when it's personal medical advice. Grounded Q&A over a bounded dietary-supplement corpus — built to be inspectable, not just impressive.

## What a question looks like

```json
$ python -m supplements_rag.pipeline "does creatine help build muscle"
{
  "status": "answered",
  "direction": "mixed",
  "verdict": "Creatine helps increase muscle strength … but the evidence
    for actual muscle growth is less direct and mixed.",
  "points": [
    {"text": "Meta-analyses consistently find strength gains with
      creatine, strongest in untrained people", "ids": ["41328071"]}
  ],
  "claims": [
    {"text": "A meta-analysis found creatine significantly improves muscle
      strength; untrained individuals improved more than trained ones",
      "ids": ["41328071"]},
    {"text": "20 g/d loading did NOT prevent muscle loss during leg
      immobilization", "ids": ["28054322"]},
    "...7 claims total"
  ],
  "citations": {"41328071": {"title": "Effects of creatine supplementation
    on muscle strength gains — a meta-analysis", "url": "https://pubmed…"}},
  "window": {"min": 2018, "max": 2024},
  "trace": {
    "screen":   {"intent": "evidence", "injection": 0.0003, "ms": 314},
    "retrieve": {"pack": ["41328071", "…10 PMIDs"], "ms": 22},
    "attempts": [{"errors": [], "verdict_score": 0.79, "write_ms": 16745,
                  "verdict_kind": "generated", "ms": 19717,
                  "grounded": {"0:41328071": 0.99, "…each ≥0.7"}}],
    "usage": {"calls": 61, "in": 21257, "out": 964,
              "by_model": {"vllm-sr/d3-flash": {"in": 20473, "out": 0},
                           "deepseek/deepseek-v4.1-flash": {"in": 784, "out": 964}},
              "unpriced": 60, "cost_usd": 0.000735},
    "ms": 20054
  }
}
```

Every number must literally appear in its cited source (doses, p-values — checked deterministically), every claim gets a judge support score ≥0.7 (`attempts[].grounded`), and the verdict is checked against the combined evidence. One pass, no rewrite — a failed check marks its sentence on the answer (`unverified`); only unparseable output still abstains. The `trace` carries the whole reasoning: gate intent + injection score, per-stage timings, each attempt's raw output, errors, judge scores, and the run's token ledger (per-model in/out + estimated USD at the upstream providers' published rates; `vllm-sr/*` calls count tokens only — the free research API publishes no rate) — it is the contract the demo streams. `window` reports the publication-year range of the cited studies, and the writer sees each quote's year — a "recent studies" question gets bounded by the evidence's actual dates instead of echoing the word. (Extractive-lane latency is the judge scoring rounds — chunks, then sentences, then claim stamps; on the opt-in writer lane it's the writer generating the contract, ~1s of checks after.)

## The finding that shaped the design

**Coverage ≠ evidence.** BM25 admits eval questions by lexical overlap — and on 4,026 admitted rows, the judge confirmed real evidence for only 24%. **52% of everything that looked covered has no supporting abstract at all.** A system that answers whatever it retrieves fabricates on half its input — so this one verifies every claim before the answer is trusted (streamed deltas render provisional — stamped `checking` — and only the settled answer is gated), and abstains by design.

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

`[mesh]` = PubMed's subject-index terms · `[tiab]` = title/abstract words · `[pdat]` = publication date · `[pt]` = publication type. Result: **17,698 abstracts → 17,696 chunks** after dedupe (measured, not estimated). The `tiab` clause exists because MeSH indexing lags publication ~15–17% in recent years — recent papers would be invisible without it. The corpus is supplement-*centered* — MeSH associations admit a thin slice of complementary-intervention studies (acupuncture, massage… ~0.1% of chunks), so it is not a general alternative-medicine corpus.

## Run it

```bash
pip install -e .[dev]                  # package + pytest/ruff/pyrefly

# verify the code — no data or network needed
pytest                                 # 60 tests
ruff check . && ruff format --check . && pyrefly check

# ask a question — needs built data/ + VLLM_SR_API_KEY
# embeddings + vector search run on Cloudflare (Workers AI + Vectorize),
# so a wrangler script produces the dense matches; Python reads the file
node scripts/query.mjs "does magnesium help sleep"   # step 1 → writes data/dense.json
python -m supplements_rag.pipeline "does magnesium help sleep"   # step 2 → answer + trace
python -m supplements_rag.gate "does magnesium help sleep"       # or run one stage alone

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

Ingest is resumable: `data/ledger.json` checkpoints every completed window + PMID — re-running only fetches new articles (that's the freshness mechanism). Keys in `.env` (see `.env.example`): `NCBI_API_KEY` raises ingest to 10 req/s (3 anonymous), `VLLM_SR_API_KEY` is required by the answer pipeline (gate, writer, verifier all call the model endpoint).

## Honest status

- **Built + verified live:** the full answering path — gate (injection + intent), hybrid retrieval (BM25 + dense + rare-entity rescue), strict contract, deterministic + semantic verification, abstain/decline/refuse paths. Two flows share the same checks: `answer()` buffered (nothing leaves before verify — evals use it) and `answer_events()` streamed — the extractive lane emits extraction progress (`chunks` → `entity` → `sentences`) then per-claim judge stamps; the opt-in writer lane streams tokens instead (`claim_stamp` can land while later claims still generate; ADR-013). The `status` payload carries the typed `direction`, `verdict_kind` (`generated` | `typed` fallback), and checked `points` findings (ADR-015)
- **Built + verified live:** the local exhibit server — `python -m supplements_rag.serve` (:8778) exposes `GET /v1/ask?q=` as `text/event-stream` of `answer_events()`, spawning `scripts/dense-server.mjs` as a warm sidecar for the embed+Vectorize leg. Same event contract the deployed Worker serves
- **Built + verified live:** the Worker deploy — `worker/src/worker.py` (Python Worker, FastAPI over `workers.asgi`) serves `GET /sr/v1/ask?q=` on `luisosio.dev` with the identical `answer_events()` SSE contract: chunks + BM25 postings in D1, bge-m3 + Vectorize via bindings, judge/writer over `httpx2`, per-ask traces in a D1 `runs` table. Verified end-to-end on prod (ADR-019)
- **Built + measured:** the extractive lane (`extract.py`, `EXTRACTIVE=1`) — answers with VERBATIM quotes instead of generated claims: chunk scores → sentence scores → quotes ≥0.6, same checks after, plus a source-level entity check when the question names a rare/absent substance — evidence must be about IT, not a related compound (ADR-016). On the same 30-row golden slice: 19/30 correct vs the writer's 9/30, median 3.5s vs 8.6s — grounded by construction, ~2.5× faster (ADR-014)
- **Not built yet:** freshness cron (incremental re-ingest on a schedule — the ledger mechanism exists, the trigger doesn't)
- **Known gaps, stated plainly:** [KNOWN_GAPS.md](KNOWN_GAPS.md) — what's unverified, deferred, or an assumption

## What this does NOT do

- Not medical advice — "should I take X?" gets a designed decline, not an answer.
- Abstracts, not full text (PMC open-access full text is the documented v2 path).
- The system searches this bounded corpus — not "all medical knowledge".

## Layout

- `src/supplements_rag/` — one module per pipeline stage, flat by design (nesting earns itself when flat hurts):
  - acquire/index — `ingest`, `chunk`, `bm25`
  - retrieve — `query`, `rrf` (rank fusion)
  - answer — `gate`, `generate`, `extract` (verbatim-quote lane), `verify`, `contract` (output shape), `judge` (model client shared by gate+verify)
  - orchestrate — `pipeline`
  - serve — `serve` (stdlib SSE server for the local exhibit)
- `evals/` — golden set + admission pipeline ([sources, counts, limits](evals/README.md))
- `tests/` — fixture + pure-function tests, no network
- `data/` — gitignored build artifacts ([what each file is](data/README.md))
- `docs/decisions.md` — ADRs with measured numbers (why, not just what)
- `KNOWN_GAPS.md` — the honest register: unbuilt, unverified, deferred
- `AGENTS.md` — repo conventions: functional style, stdlib-first, runtime-verified

## License

MIT — see [LICENSE](LICENSE).
