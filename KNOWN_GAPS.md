# Known gaps

The honest register — everything that is unbuilt, unverified, or deliberately
deferred. If it isn't listed here or marked `todo` in the README pipeline
diagram, it should work; if something here gets fixed, delete the line.

## Unverified / needs a pass

- **Evidence semantically verified** (vllm-sr judge, `--repropose`):
  golden 33 verified / 14 partial / 33 refuted; **full admitted pool
  (4,026) now judged**: 968 verified (24%) / 342 partial / 2,080 refuted
  (52%) / 633 no-evidence / 3 error. Over half of lexical admissions lack
  real evidence — coverage is not evidence.
- **33 refuted rows kept flagged** — corpus may still hold evidence
  retrieval missed; they're hard negatives, not proven unanswerable.
- **Abstention threshold uncalibrated** — golden set shows a thin overlap
  band (abstain max BM25 11.97 vs answerable min 10.99); needs a gray-zone
  design calibrated on the full admitted pool, not 120 rows. (The judge's
  question↔source bands ARE now pool-calibrated — 9,268 pairs, bimodal
  with a 0.20–0.60 valley; the remaining gap is the BM25 side.)
- **Judge-call thresholds remain tuned on the golden slice** —
  `INJECTION_MIN`, `GROUNDED_MIN`, `SCOPE_MIN`, `SENT_MIN` have no
  pool-wide score dumps; each is set at a demonstrated gap from live
  checks (`docs/decisions.md` records the observations). `ENTITY_MIN`
  escaped this class — calibrated on 1,375 real calls from the v8 run
  (dropped p50 0.02 vs kept min 0.60). Eval rows persist `scores`
  (`entity_scores`, `grounded`, `synth_score`, `best_chunk`) so the
  remaining thresholds accumulate a distribution with every run —
  recalibrate once each sample is large enough.
- **1,445 SuppLLMent items rejected as `labeled_insufficient_but_covered`** —
  corpus mentions the pair but the label says insufficient evidence. Review
  queue, not silently admitted.

- **Entity check covers the extractive lane only** — `entity_score` runs
  in `_extractive_answer`/`_extractive_events`; the writer lane
  (`EXTRACTIVE=0`) can still ship a right-format claim about a
  different substance (claim↔source checks can't see question↔claim
  mismatch). Opt-in path, unexercised by evals — fix lands if the writer
  lane becomes the default again (ADR-016).
- **The eval's own semantic labels carry the entity hole** —
  `evals/verify.py` asks "does this abstract answer the question" without
  anchoring the substance, so `semantic_verified` partly means "verified
  for the condition" (marine-oil studies verified as wallflower-for-
  arthritis evidence). Re-running verification with entity-anchored
  instructions would sharpen the golden's truth column. The held-out
  run showed the sibling: `semantic_refuted` flags the *proposed* PMIDs,
  not the achievable answer — 6 rows "refused" by label were answered
  correctly on other correct-entity evidence (entity scores 0.66–0.97).

## Deliberately deferred (not bugs)

- **2 orphan vectors** in Vectorize (`26755330`, `40441699`) — embedded before
  the dedupe pass dropped them. Harmless: `retrieve()` filters fused ids not
  in the chunk map. Clean on next full re-embed.
- **87 no-abstract chunks (0.5%)** — title-only records kept for coverage;
  thin evidence by design.
- **MeSH absent from embedding text** — indexing lag makes it inconsistent on
  recent records; kept as filter metadata only (ADR-003).
- **89 SuppLLMent rows** dropped at seed build — didn't match the
  `Is 'X' effective at treating 'Y'` input pattern.
- **11 of 99 MedQuAD supplements** have no corpus coverage — their answerable
  questions rejected, advice questions kept (gate fires pre-retrieval).

## Not built yet

- Worker deploy layer, D1 schema/loaders, freshness cron

## Built, verified live (single-question CLI runs — not load-tested)

- Gate (injection noul + intent choice, `d3-flash`): evidence/advice/
  unrelated/injection all classified correctly on live calls — but the
  golden `advice` rows are safety-information questions (evidence-
  answerable), so "advice → declined" is NOT the eval expectation;
  per-row intent gets recorded and judged
- Contract + verify + marks: a magnesium question's first attempt failed
  deterministic checks (2 numbers not in cited sources) — caught and
  marked `unverified` on the answer, not silently rewritten (repair was
  removed for this: one pass, marks over fixes) — answered with `[pmid]`
  citations and per-claim grounded ≥0.7
- Paths exercised live: answered / declined / refused / unclear (explicit
  `missing` list) / off_topic. `unavailable` (fail-closed on judge outage)
  is unit-tested only.
- Caveats: writer is `vllm-sr/auto` (routed to qwen3.8-27b today — router
  may swap models); generation is non-deterministic in claim *shape* even
  at temperature 0 (same question can answer or abstain across runs —
  eval-measured); gate `uncertain` → pipeline refuses rather than
  proceeds (fail-closed); thresholds (INJECTION_MIN 0.7, GROUNDED_MIN 0.7,
  SCOPE_MIN 0.4) are assumptions — the pool pass data exists to calibrate
  them, the calibration itself is still owed.
- Verified-answerable eval rows: ~70% answered; remainder are
  verify-strictness abstains, unclear-on-verified rows
  (rescue window vs judge strictness), and rare gate edge cases.

## Housekeeping

- `dense.json` is dev glue — production queries Vectorize inline (see
  `data/README.md`).
