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
- **Condition axis is a veto, not a gate** — `cond >= 0.1` catches
  clearly-wrong conditions (the acupuncture-fibromyalgia class:
  same substance, wrong population) and only fires when the question
  names a condition — `has_cond < 0.5` skips it (definitional asks
  otherwise ate undefined-axis noise: measured cond 0.12–0.58 on
  'What is Capsicum?'). Borderline mismatches in the 0.1–0.6 band
  still pass the axis — but measured post-has_cond (golden rerun):
  all 20 rows with mid-band refuted chunks produced zero kept
  sentences (best_chunk ≤0.44 vs verified answerers ≥0.77) —
  `question_score`/`sentence_score` are the de-facto
  question↔source gate; a wrong-condition chunk can't quote a
  sentence that answers the question. Residual: generic boilerplate
  (safety/tolerability text) answering the letter of a question from
  a wrong-condition study — verdict layer, not extraction.
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

- Freshness cron — `data/ledger.json` makes re-ingest incremental but nothing
  schedules it on prod yet

## Built, verified live (single-question CLI runs — not load-tested)

- Worker deploy (`worker/src/worker.py`, prod `luisosio.dev/sr/v1/ask`):
  full `answer_events()` contract served over FastAPI/`workers.asgi` —
  gate → retrieve (D1 BM25 postings + Vectorize dense) → extract →
  verify → answered, streamed verbatim on prod. Pyodide constraints hit
  and handled: no sockets (judge/writer over sync `httpx2`), no threads
  (`_pool.py` serial executor on `emscripten`), lazy app construction
  (startup memory). Per-ask event log lands in the D1 `runs` table —
  completed and abandoned runs alike (stub insert at stream start +
  update on completion; disconnect-tested on prod: a killed stream left
  the `prepare_begin` stub row, `cold` flag included) and is publicly
  readable — the stream's `run` event carries an opaque public id,
  `GET /sr/v1/run/{rid}` returns the stored events (same route on local
  `serve.py` over `runs.jsonl`). Verified live,
  not load-tested; serial extract/verify makes prod runs slower than
  local's thread pool.
- Gate (injection noul + intent choice, `d3-flash`): the advice/evidence
  boundary is person-hood — impersonal safety/interaction asks are
  evidence, first-person framing ("should I take X", "with my meds")
  takes precedence into advice. Earlier criteria let 'interactions'
  pull personal asks into evidence — caught live ("should I take
  ashwagandha with my blood-pressure pills?" screened `evidence`),
  fixed and battery-verified; six authored golden rows now cover the
  first-person class the MedQuAD-derived rows never had. Golden
  `advice` rows tolerate either outcome (`answer_or_decline`) so the
  route isn't asserted — per-row intent gets recorded and judged
- Contract + verify + marks: a magnesium question's first attempt failed
  deterministic checks (2 numbers not in cited sources) — caught and
  marked `unverified` on the answer, not silently rewritten (repair was
  removed for this: one pass, marks over fixes) — answered with `[pmid]`
  citations and per-claim grounded ≥0.7
- Paths exercised live: answered / declined / refused / unclear (explicit
  `missing` list) / off_topic. `unavailable` (fail-closed on judge outage)
  is unit-tested only.
- Caveats: the opt-in writer lane's model is `vllm-sr/auto` (routed to
  qwen3.8-27b today — router may swap models) and samples claim *shape*
  non-deterministically even at temperature 0; the extractive lane
  measured 0 status flips across 360 runs. Gate `uncertain` → pipeline
  refuses rather than proceeds (fail-closed). Threshold accumulation:
  `ENTITY_MIN` is calibrated (1,375 real calls); the rest carry per-row
  score dumps until their samples are large enough (see above).
- Verified-answerable eval rows (v8, extractive lane): **91%** answered
  on golden (120 rows × 3 runs at the time, 0 flips), **92%** on the
  199-row held-out set; refuted-refusal 100% / 92%, **0 dangerous answers**.
  Remainder are verify-strictness abstains and unclear-on-verified rows
  (rescue window vs judge strictness). Post-worker rerun (126 golden
  rows incl. the six authored advice rows, ×1): **94%** verified-answered,
  **97%** refuted-refusal (one refuted row answered — traced to the
  condition-mismatch hole: real quote, wrong population), **0 dangerous**,
  0 flips. Post condition-axis rerun (ADR-016 second axis, COND_MIN=0.1):
  **88%** verified-answered (3 of the losses were transient judge
  `unavailable`s, not the axis), **100%** refuted-refusal, 0 dangerous.
  Category scoring now joins the verification labels (`score_row` in
  eval_pipeline.py — a `semantic_refuted` row that answers citing the
  refuted PMIDs is DANGEROUS; citing different evidence is FLAGGED for
  review, since the label judges the proposed PMIDs not the achievable
  answer; category + label both derive at summary time so relabeling
  re-scores history): abstain **29/29**, advice **17/17**, answerable
  **41/50**, negative **26/30**, 0 dangerous. 14 MedQuAD interaction
  questions moved advice→abstain — the corpus carries almost no
  herb-drug interaction evidence (~2 real review papers), so `unclear`
  was honest abstention, not over-caution (nearly every "interaction"
  corpus match was a statistical time×treatment artifact). Remaining
  `semantic_partial` weakness measured
  and settled: sentence-score distributions on the 14 unclear rows that
  reach the sentence stage overlap — partial-evidence maxima 0.44–0.67 vs
  abstain noise max 0.58 (an olive-interaction row's top sentence is a
  generic antioxidant quote, not interaction data — a real false
  positive). No threshold separates them, so `unclear` on partial coverage
  is the honest floor of the extractive lane, not a tuning hole; lifting
  it needs a different evidence class (safety/interaction studies aren't
  in the efficacy-grade corpus slice), not a lower gate. Plus 3 transient
  `unavailable`s under burst. Post-has_cond held-out rerun (199 rows,
  fresh): verified 0.92, refuted-refusal 0.87, answerable 0.86, negative
  0.88 — thresholds generalize; and it exposed the class golden never
  had: **6 dangerous** — 5 MedQuAD "interactions between X and Y" rows
  answered on non-interaction evidence (statistical P<0.05 noise,
  efficacy papers — `sentence_score` can't tell a clinical interaction
  from a statistical one) + 1 refuted-PMID leak; 9 `flagged` answers on
  different unjudged evidence. 30 heldout rows relabeled advice→abstain
  on measured corpus absence (kept biotin/oats safety rows — adjacent
  content exists). Post-rel-axis rerun: **dangerous 6→4** — the
  fabricated class is dead (statistical-noise and efficacy-paper
  interaction answers abstain); residual is 3 *scope-mismatch* answers
  on real combination studies ('herbs' answered with a medications
  review — rel correctly counts combination evidence, the pair scope
  is looser) + 1 refuted-PMID leak. Note: the entity/rel gate only
  fires on corpus-rare question tokens — common substances never see
  the axis (sentence layer is their only net).
- Near-miss eval (`evals/nearmiss.jsonl`, 50 rejected-admission questions,
  all `no_corpus_coverage`): 48 refused cleanly; 1 dangerous —
  phytase-for-infant-development answered on an *animal* weight-gain
  study (species mismatch — hedged, but wrong-species evidence); 1 row
  mislabeled by the admission rule itself ('how effective is creatine'
  answered — corpus covers it; dense leg found what BM25's check missed).
  Species class is real but inconsistent across sets: 822 animal-only
  studies in-corpus (PubMed tags animal RCTs with the same
  `Randomized Controlled Trial` pubtype the ingest filter trusts); 4
  answered rows cite them — 3 present animal evidence as human-relevant
  (phytase, L-arginine, strontium), 1 hedges correctly (alfalfa dosage
  answer explicitly says the studies are farm-animal feeding trials).
- Usage/cost ledger: a creatine run logged 61 calls (60 judge + 1 writer),
  `cost_usd` at upstream published rates — judge calls count tokens only
  (`vllm-sr/*` is free research API, no rate published). Pricing is a
  static table of list rates, not the deployment's real bill ($0/quota).

## Housekeeping

- `dense.json` is dev glue — production queries Vectorize inline (see
  `data/README.md`).
