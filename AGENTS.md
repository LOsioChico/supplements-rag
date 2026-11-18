# Conventions

This is a **public repository** — code and docs must read well to a stranger.

## Style

- **Functional style preferred:** pure functions (data in → data out), no classes
  for logic, no mutable shared state. Small composable functions over methods.
  `Eutils` is the sole exception — it is a thin transport boundary (rate limit +
  API key), not domain logic.
- **Stdlib first.** No dependencies unless a measured need exists (G6). If a dep
  lands, it gets a `docs/decisions.md` entry.
- **Plain English everywhere.** Public artifacts target non-experts: expand
  acronyms at first use (BM25, RRF, PMID), docstrings say *why*, not *what*.
- Every pipeline stage is a small module with a CLI (`python -m ...`) so each
  step is inspectable and rerunnable alone — `rrf` is the exception (pure
  rank math, exercised through `query`).
- **Flat layout is deliberate** — one module per stage: acquire/index
  (ingest, chunk, bm25), retrieve (query, rrf), answer (gate, generate,
  verify, contract, judge), orchestrate (pipeline). No sub-packages until
  the file count earns them.

## Rules

- **The repo stands alone** — never name other projects in code, docs, or
  commits; a stranger should need no sibling context.
- **Never commit `.env`** (holds `NCBI_API_KEY`) — gitignored, `.env.example`
  documents the shape.
- **Never commit `data/`** — corpus + ledger are regenerable; the code is the
  contract. Record counts in `docs/decisions.md` instead.
- Runtime-verify before claiming: `pytest` + `ruff check` +
  `ruff format --check` + `pyrefly check` all clean, plus a live run on a
  small window, evidence pasted into `docs/decisions.md`.
- No speculative stages — each module lands only when the pipeline needs it.
- **Docs are the live reference, updated in the same change as the code.**
  Any change to behavior, counts, files, or stage status lands together with
  its doc update — README pipeline diagram + status line, `data/README.md`
  artifact table, `evals/README.md` counts, `docs/decisions.md` ADR, and
  `KNOWN_GAPS.md`. A doc that lags the code is a bug, not a follow-up task.
- **Commits use conventional prefixes** — `feat:`/`fix:`/`docs:`/
  `refactor:`/`test:`/`chore:`, one-line subject, atomic diffs. The repo is
  public: the log is part of the evidence.

## Verified facts (measured on the live corpus)

- PubMed ESearch cannot page past 9,999 records → ingest is date-sliced monthly.
- Slice `dietary supplements` (MeSH or tiab) + evidence-grade pubtypes,
  2015–2026 → 17,698 articles (measured 2026-11).
- `YYYY/MM/DD:YYYY/MM/DD[pdat]` is the only working date-range syntax.
- SystemOne typed questions: `noul` → `{noul: 0-1}`; `choice` needs
  `criteria: {option: description}` — `options`/`choices` fields are
  silently malformed (422). Judge = `vllm-sr/d3-flash`; writer =
  `/v1/chat/completions` model `vllm-sr/auto`.
- Full answering path verified live: gate → retrieve → contract →
  verify (deterministic + judge) → failed checks mark sentences
  `unverified`; only unparseable output abstains. One pass, no rewrite.
  Extractive lane is the default (`EXTRACTIVE=0` → writer lane). 60 tests.
  The `status` event also carries typed `direction`, `verdict_kind`
  (`generated`|`typed`), and checked `points` findings (ADR-015).
