# Decisions

## ADR-001: Date-sliced PubMed acquisition (2026-11-15)

**Context:** the evidence-grade query returns 17,698 records, but PubMed's `retstart` hard-caps at 9,998 — no history paging can enumerate the whole slice (verified live).

**Decision:** slice the query into per-month `[pdat]` windows (`YYYY/MM/DD:YYYY/MM/DD[pdat]` — the only date format esearch accepts; `YYYYMMDD` silently returns 0, verified). Union results into a PMID ledger → `efetch` batches of 200.

**Consequences:** every window is far under the cap; the ledger makes ingest resumable/idempotent and gives incremental freshness for free (re-run recent windows → fetch only new PMIDs). Alternative rejected: EDirect (external tool, same logic we'd reimplement).

## ADR-002: Evidence-grade pubtype filter at acquisition

**Measured:** 93,572 raw slice → 17,698 with `("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR "controlled clinical trial"[pt])` = 18.9%. Filter lives in the query, not post-hoc — we never download noise.

## ADR-003: Abstract = atomic record, sections as metadata — measured

Structured-abstract labels (BACKGROUND/METHODS/RESULTS/CONCLUSION — verified on real records) are preserved in `abstract_sections` metadata, not split into separate chunks. Alternatives (sentence/fixed-window splitting) fragment evidence and lose verdict context; section-split is reconsidered only if retrieval eval says so.

**Measured on the full corpus (17,698 records, pre-dedupe):** p50 = 255 words, p99 = 573, max = 4,774 — everything fits bge-m3's 8,192-token window, so no length-driven split exists. Section-level chunking rejected: it would 4× the index (≈68k docs) and each fragment loses verdict context; citations resolve to PMID regardless and UI evidence-highlighting substring-matches the preserved sections instead.

**Embedding input = `title + labeled abstract`** (not abstract alone): titles carry the intervention+outcome terms queries contain, and it keeps the dense channel symmetric with BM25 (which already indexes title+text) — cleaner RRF fusion. MeSH terms are excluded from embedded text (absent on recent articles from indexing lag → inconsistent signal) but kept as filterable metadata.

## ADR-004: Index = Vectorize + in-process BM25 + D1

On Workers Paid: 13.6M stored dims ≈ $0.002/mo; queried dims under 50M included. sqlite-vec rejected (C extension can't load in Pyodide); full-fp32 in-memory rejected (>128MB isolate); int8 in-worker artifact kept as zero-dependency fallback. Query-time embedding = bge-m3 via Workers AI (custom HF models can't be uploaded to Workers AI; containers are the escape hatch only if eval forces it).

## ADR-005: BioASQ rejected as eval benchmark — measured

0.1% of its 40,181 evidence PMIDs overlap the supplements slice; ~0 of 4,719 questions are supplement questions (keyword hits were molecular-biology false positives). Golden set is sourced instead: SuppLLMent effectiveness items + MedQuAD `MPlusHerbsSupplements` questions, each admitted only after its evidence PMIDs verify in the corpus.

## ADR-006: BM25 hand-rolled, not rank_bm25

BM25 (Best Match 25 — the standard keyword-ranking formula: term frequency
saturated by document length, weighted by inverse document frequency) is ~30
lines of stdlib math. A dependency adds nothing; hand-rolled also means the
index dict is exactly the JSON we ship to the Worker — no serialization
adapter. If a measured gap appears (e.g. phrase matching), we revisit.

## ADR-007: RRF (k=60) for hybrid fusion

Reciprocal Rank Fusion merges BM25 and vector lists by *rank*, not raw score —
BM25 scores and cosine similarities are on incompatible scales, and fusing raw
numbers would let the bigger range dominate. k=60 is the canonical constant
(Cormack et al., 2009). Alternatives rejected: score normalization (fragile —
min/max drifts with every index rebuild), weighted fusion (adds a tuning knob
with no evidence it beats rank fusion).

## ADR-008: Embedding pipeline — Workers AI bge-m3 → Vectorize

`scripts/embed.mjs` (Node, not Python — the Workers bindings only exist on the
wrangler/JS side) embeds `title + abstract` via `getPlatformProxy` and upserts
to Vectorize. Measured constraints: `AI.run` caps each batch call at **60k
tokens total** — the batcher packs by estimated tokens (chars/3.5 proved ~1.8×
optimistic; effective limit set to 25k est ≈ 45k real). Vectorize has no local
mode → binding needs `remote = true` in wrangler.toml. Checkpointed in
`data/embed-ledger.json` — re-runs skip upserted chunks.

## ADR-009: Living-review versioning at chunk time (measured)

30 same-title groups found in the corpus (65 chunks): Cochrane systematic
reviews republish updated versions over years under new PMIDs. Chunk stage
now (a) drops true duplicates — identical text under a second PMID, 2 dropped;
(b) marks older versions `superseded_by: <newest pmid>` — 33 flagged — instead
of deleting them (historical questions are legitimate). The answer layer must
prefer and surface the newest version; citing a 2015 review when a 2020 update
exists is a stale-evidence bug, not a style choice. Verified on the live
corpus: 17,698 → 17,696 chunks.

## ADR-010: Golden set — SuppLLMent + MedQuAD, admitted on corpus verification

Seed sources: MedQuAD `12_MPlusHerbsSupplements_QA` (792 questions, 99
supplements — questions only; answer text was removed upstream) and
SuppLLMent effectiveness items (4,777 parsed of 4,866; 89 dropped on the
input-pattern filter). **Admission rule:** a seed enters the pool only when
BM25 retrieves a corpus chunk above threshold (measured `--bm25-min 8.0`)
and its top-3 PMIDs attach as candidate evidence. Result: 4,026 admitted —
1,832 answerable / 1,561 negative / 495 advice / 138 abstain — and 1,543
rejected with reasons kept in `evals/rejected.jsonl` (98 no coverage;
1,445 labeled-insufficient-but-covered → review queue). `golden.jsonl`
pins a seeded 120-row stratified sample (50/30/25/15) as the eval contract.

Key design points: (a) `insufficient_evidence` + no coverage = clean
abstention cases — free, no authoring; (b) MedQuAD dose/interactions/
precautions questions are *safety-information* probes — evidence-answerable
("studies report GI effects") not "should I take it": a cited evidence
answer is correct and a decline is defensible, so the eval records the
gate's per-row intent instead of asserting decline; (c) ineffective_for
labels are *negative-verdict* cases (cite the no-benefit evidence), not
abstentions — a substring label check misrouted 596 of these to answerable
until exact set membership fixed it; (d) `evidence_pmids` are
BM25-*proposed* (`verification: retrieval_only`) — a semantic pass is still
owed before evals treat them as ground truth.

## ADR-011: Semantic verification via vllm-sr SystemOne judge — coverage ≠ evidence

Golden `evidence_pmids` are BM25-proposed; a semantic pass checks each
(question, abstract) pair. **Judge:** the vllm-sr SystemOne endpoint
(`POST /v1/systemone`, `noul` score 0–1, model `d3-flash`) — a semantic-
router judge that scores support directly, so verdicts get real numbers
for banding instead of one-word labels. Rejected alternative: a chat model
one-word verdict (Workers AI llama-3.3-70b) — cruder output, no score for
banding, and its strictness is unprompted-tunable. Bands: ≥0.7 yes /
≥0.4 partial (assumption, tunable); scores stored so bands re-derive.

**Measured justification:** on the golden set's 80 evidence rows, only 30%
verified at top-3 (24 verified / 12 partial / 44 refuted — refuted median
score 0.05, confidently absent). Inspection showed lexical false positives:
"X for Y" retrieves about-X or about-Y, not the pair. A re-proposal pass
(BM25 top-10 + dense top-10, +1,575 judgments) recovered 9 rows → final
33 verified / 14 partial / 33 refuted. Consequence: admission requires the
judge, and the runtime answer contract needs the same verification before
text streams — retrieval alone cannot certify evidence.

## ADR-012: Pipeline architecture — evidence-verified, never stream unverified

The query path is `gate → retrieve → generate → verify → done`. Every design
choice below exists because of a measured failure or a stated assumption —
each is re-derivable from this repo alone.

- **Retrieval mechanisms (built, verified):** NFKD tokenizer
  (`ácido`→`acido`), bilingual STOP list, `dense_query()` keyword form
  (`"information about {kw}"` — interrogative scaffolding drags query
  vectors away from declarative document space), rare-entity coverage
  (query tokens with df ≤ ~0.5% of corpus force their chunks into the
  pool — the direct fix for the measured 'laxogenin → breath-holding
  studies' failure: entity evidence can't be out-ranked by topic
  phrasing). RRF k=60 fusing BM25 top-20 + dense top-20.
- **Gate (built, verified live):** typed judge questions on the raw
  question — `choice` intent (evidence/advice/unrelated via `criteria`
  option map) + `noul` injection score — before retrieval. The `evidence`
  criterion covers definitional "what is X" questions AND complementary/
  alternative interventions (acupuncture, massage, tai chi — they're in
  the corpus even though they're not ingestible supplements; bare
  substance names and CAM questions were misclassified `unrelated` —
  false-off-topic bugs the eval's verified rows exposed). The scope check
  itself scores top-3 pack chunks with the question-evidence instruction
  (the title-only 'could this answer' phrasing under-scored covered
  questions — same instruction-shape bug as the rescue). An 'unrelated'
  intent is not an instant veto: the post-retrieval scope judge sees the
  evidence titles before deciding (the gray-zone the BM25 distribution
  demanded). The advice/evidence boundary is person-hood, not topic:
  impersonal safety/interaction asks are evidence (a cited summary is
  not a recommendation — the eval's advice category tolerates either),
  first-person framing ("should I take X", "for me", "with my
  medication") is advice and takes precedence even when it names a
  substance or interaction — the earlier criteria let 'interactions'
  pull personal asks into evidence (live-verified: "should I take
  ashwagandha with my blood-pressure pills?" screened `evidence`, six
  authored golden rows now cover the class). Advice declines
  pre-retrieval; gate outage fails closed.
- **Contract + verify (built, verified live):** strict JSON
  `{status, verdict, claims: [{text, ids: [pmid]}], missing}` — `verdict`
  is the 1-2 sentence bottom line (a bare claims list read as a study
  dump, not an answer); it is judged against the *combined* cited evidence
  (per-source judging is structurally wrong for synthesis claims).
  Deterministic checks
  first (every claim cites, cited ids ∈ retrieved, every number ⊆ source
  text — with spelled-out-number expansion ('two' = 2), rounding
  acceptance ('4.4' matches '4.43'), and writer instructions to quote
  exact reported p-values — doses and p-values are where silent
  corruption lives), then per-claim `d3-flash` groundedness (≥0.7).
  Writer: `temperature: 0`, `max_tokens: 8000` (4000 truncated reasoning-
  heavy contracts mid-JSON — reasoning counts against the budget).
  ONE pass, no rewrite — the earlier single repair attempt was removed:
  a failed check marks its sentence (and the bottom line) `unverified`
  on the answer instead of buying a hidden retry; only unparseable
  output still abstains. Per-request canary (CNRY token)
  catches prompt echo. Writer = `/v1/chat/completions` model `auto`
  (router picks the model — qwen3.8-27b at measure time; not pinned).
  Verified: a magnesium/sleep question's first attempt failed the numbers
  check (2 fabricated figures — caught, marked, not silently fixed);
  laxogenin returned `unclear` with an explicit `missing` list; dosing →
  decline; jailbreak → refuse (injection 0.9998); world cup → off_topic
  via scope check.
- **Unclear-rescue (built after eval surfaced under-answering):** when the
  writer emits `unclear`, the judge scores the top-8 pack chunks against
  the *question* — ≥0.7 records an "evidence was there; `unclear` was the
  wrong call" error on the abstained answer's trace (the abstain stands,
  the disagreement is visible — with repair gone there is nothing to
  convert it into). Symmetric to
  verify(): rejects unsupported *refusals*, not just unsupported answers.
  The instruction asks "provides evidence that answers this question",
  not "supports this claim" — a question isn't a declarative claim and
  the claim-shape instruction under-scores it (measured: verified rows
  with evidence at pack positions 0-1 still rescued below threshold
  until the question-shaped instruction fixed it; window 5→8 covers
  evidence at positions 5-7).
  Eval-measured: golden-run showed abstain=100% correct / 0 dangerous,
  while verified-answerable rows abstained ~45% on verify strictness +
  non-deterministic generation (rescue moved advice 20%→36%; answerable
  stayed ~22% because `retrieval_only` rows legitimately lack evidence —
  see evals/README.md).
- **Deliberate scope choices:** hand-rolled BM25 (ADR-006); no
  intent-family votes or canonical pins (a 17.7k uniform corpus has no
  families to vote on); `superseded_by` versioning (living reviews,
  ADR-009); verify fails CLOSED when the judge is unreachable — medical
  claims must not stream unverified; no `url` field on chunks (pmid →
  pubmed.ncbi.nlm.nih.gov/{pmid} derives at hydration, zero storage).
- **Measured honestly:** the keyword-form dense embedding did NOT lift
  category separation in the golden eval (scores dropped uniformly ~0.03).
  Kept because the mechanism is sound for entity questions; not claimed
  as a measured improvement here.

## ADR-013: Two answer paths — buffered `answer()` + streamed `answer_events()`

- **Decision:** the pipeline exposes two flows over the same check
  functions. `answer()` is the strict buffered path (unverified text never
  leaves — evals + CLI use it). `answer_events()` is the exhibit path:
  writer tokens stream live and each claim is judge-stamped the moment
  its JSON object closes — a stamp can land while later claims are still
  generating — with the SAME deterministic → grounded → verdict → rescue
  checks deciding the final `status` event. Optimistic, not weaker: a
  live claim can still end in abstain.
- **Why:** the demo's thesis is "verification is visible process, not
  hidden gate" — claim stamps are rendered live (colors/animations are a
  UI concern; the event contract carries `claim`, `claim_stamp`,
  `verdict_score`, `attempt`, `status`). Buffered `answer()` remains the
  semantic reference so evals still measure the strict path.
- **Event contract:** `screen`, `retrieve`, `scope`?, `token`, `claim`,
  `claim_stamp` (per claim×pmid), `verdict_score`?, `rescue`?, `attempt`,
  `status` (terminal — carries the same payload shape `answer()` returns).
- **Measured live (creatine run):** 69 token events, claim 0 stamped
  0.995 while claims 1–5 still generated; verdict 0.79; `status:
  answered`. Judge outage mid-stream → `unavailable` (fail closed).
- **Shared pieces:** `generate._write_msgs`/`_sum_msgs` build the prompts
  for both buffered and streamed calls (can't drift — the streamed forms
  just swap `chat` for `chat_stream`); `verify.grounded_errors`
  scores the stamp map with the same ANY-cited-source ≥0.7 rule.

## ADR-014: Extractive answer lane — quotes, not generated claims

- **Decision:** `extract.py` + an `EXTRACTIVE` env branch in the pipeline
  answer the question with VERBATIM sentences from the retrieved pack.
  `extract.claims` scores each pack chunk against the question
  (question-shaped noul), mines the top-3 chunks' sentences, and keeps
  sentences scoring >= `SENT_MIN` (0.6) as claims citing their own PMID.
  `extract.compose` emits a contract-shaped object so the SAME
  deterministic + per-claim judge checks run unchanged; the verdict is a
  fixed label ("the strongest evidence in the pack, verbatim:"), not a
  claim, so verdict-level grounding does not apply. No claims over the
  bar → `unclear`. Judge outages still fail closed → `unavailable`.
- **Why:** the chat router forces chain-of-thought on any request whose
  payload is clinical abstracts — measured across ~15 probes: content
  type, not size/prompt/instructions, drives `reasoning=on`, and no
  request-side override exists (direct pool model 403s; effort params are
  ignored or rejected). That thinking phase was most of the wait. A quote
  is also grounded BY CONSTRUCTION — it cannot cite what it isn't.
- **Measured (30-row stratified golden slice, same rows both modes):**
  extractive@0.6 `19/30 correct, 0 dangerous, median 3.5s, p90 5.2s` vs
  writer `9/30, 0 dangerous, median 8.6s, p90 32.1s`. The sentence split
  requires a capital/digit after the boundary so `L. acidophilus`,
  `et al.` stay inside their quote — without it quotes shatter mid-name
  and under-score. Remaining `unclear` misses split ~50/50 between
  retrieval gaps (pack holds nothing — honest) and borderline scores.
- **Reliability:** `judge.ask` retries 429s with backoff (the lane issues
  ~30 judge calls/question; bursts rate-limited before the retry —
  measured 10/30 unavailable dropping to 0).
- **Status:** the default lane since the exhibit shipped — `EXTRACTIVE`
  defaults to `1` and `EXTRACTIVE=0` opts back into the writer lane.
  `answer()`/`answer_events()` share it; the event contract adds
  `extract` (chunks|sentences) in place of `token`/`think`.

## ADR-015: Answer contract — `direction` as data, `verdict_kind`, `points`

- **Decision:** three fields joined the `status` payload so the client can
  render meaning as UI chrome instead of trusting prose:
  - `direction` — a typed judge call (`extract.direction_for`) over the
    kept claims → `supports | contradicts | mixed | speaks`. It travels as
    data and renders as a stamp; the synthesis prompt no longer asks the
    writer to prefix a label, because the model literally wrote "Mixed:"
    into answers when it did.
  - `verdict_kind` — `"generated"` when the synthesis' bottom line scores
    ≥ `GROUNDED_MIN` under `_synth_grounded` (summary-shaped check vs the
    kept quotes, not claim-shaped); `"typed"` when it fails and the
    `DIR_PHRASES[direction]` fallback is all that stands. A rejected
    generated line is preserved on the attempt as `synth_rejected`
    ({text, score}) — the failure is inspectable, not silent.
  - `points` — up to 4 generated findings ({text, ids}); every point must
    pass the same `_synth_grounded` bar before it ships
    (`_check_points`); `points_emitted`/`points_kept` on the attempt show
    what the check dropped.
- **Why:** direction in prose was unreliable (the label leaked into the
  answer text) and unstyleable; the exhibit needed a typed signal it can
  render as neutral chrome without editorializing (no green=yes — "does X
  cause harm" inverts the desirability). Generated summaries that flunk
  grounding used to leave the box claiming a backed answer it didn't
  have; `verdict_kind` lets the header say the honest thing.
- **Live scans:** `_scan_verdict`/`_scan_claims` surface the verdict event
  and per-claim events mid-write from the partial JSON buffer;
  `_scan_points` rescues the points array when the wrapper is chewed —
  the `verdict` event stays authoritative, scans only shape the timeline.
- **Measured:** 57 tests green. Live exhibit runs this session — zinc →
  `status: answered, direction: mixed, verdict_kind: generated`, verdict
  "Zinc might shorten a cold a little, but it probably won't prevent one
  and side effects are possible."; magnesium → `answered`, 8/8 sentences
  backed, verdict_score 0.89; a stress run where the generated line
  failed at 0.55 correctly surfaced `verdict_kind: typed` with the
  direction stamp still on the checked sentences.
- **Status:** shipped on `answer()` and `answer_events()` — the streamed
  `status` event carries the same payload shape `answer()` returns.

## ADR-016: Per-claim entity check — the question's substance must BE the quote's substance

- **Decision:** when the question names a corpus-rare or absent entity
  (`bm25.entity_tokens` — df ≤ ~0.5% of docs, including df=0), the
  top-ranked chunks face `extract.entity_score` — a dedicated judge call
  asking only "does this SOURCE study the same substance or intervention
  the question names? Exact identity or an established name for the
  identical thing counts — including codenames the source itself defines;
  constituents, derivatives, and related compounds do not." Only chunks
  ≥ `ENTITY_MIN` (0.6) can produce quote candidates (`ENTITY_POOL` = top
  5, so a dropped leader leaves replacement material). One implementation
  serves both lanes: `extract.claims_iter` yields (phase, done, n) tuples —
  `claims()` drives it silently, the streamed lane wraps the same yields
  into `extract` events (`entity` phase included); the writer lane
  (`EXTRACTIVE=0`) doesn't run it — recorded in KNOWN_GAPS. Scoring at chunk level, not quote level: the abstract body
  resolves what a bare sentence can't, and a wrong-entity chunk can't
  produce evidence for the named substance at all.
- **Why a separate call, not a criterion inside `sentence_score`:** asked
  inline ("evidence about the named substance that answers it"), the
  judge trades entity identity off against topical relevance — a
  myricetin-for-myrcia quote scored **0.81** folded in, **0.31** as a
  dedicated identity check. One criterion, nowhere to hide the mismatch.
- **Why not deterministic:** literal-containment rules can't separate
  substances from conditions — 'myopathies', 'procedure', 'baby' are all
  corpus-rare question tokens that are conditions, not the asked
  substance; and name-variants invert it ('wallflower' is df=0 while real
  wallflower evidence exists under another name — semantic_verified).
  Identity needs semantics; containment doesn't prove identity anyway.
- **Measured (golden, v5→v7):** dangerous abstain-answers 2→0 (myrcia→
  myricetin, scopolia→griffonia); refuted-row leaks 2→0 (limonene→
  spirulina, tin→OraCMU — plus unflagged wrong-entity catches:
  safflower→nigella, capsicum→avocado-soy, and the adversarial one —
  arnica→`ART`, which expands to *Artemisia annua* (Arthrem), an
  art-prefix look-alike the check still rejects at ~0.03). The apparent
  verified-answer losses were the same class — the eval's own
  `semantic_verified` labels verify the condition, not the substance
  (same unanchored-prompt hole; recorded in KNOWN_GAPS).
- **ENTITY_MIN = 0.6, now distribution-backed:** set from the
  demonstrated spot-check gap (same-entity ≥0.65, wrong-entity ≤0.55),
  then confirmed on the v8 3-run golden pass — 1,375 real entity calls:
  dropped chunks p50 0.02 / max 0.59, kept min 0.60 / p50 0.86. The
  threshold sits in a measured valley (0.40–0.60 is near-empty), not an
  assumed one.
- **Cost:** the gate fires on ~90% of golden questions (108/120) — real
  questions nearly always carry a rare token, so "rare" can't mean
  "rarely". Bounded by `ENTITY_POOL` = 5: ≤5 extra judge calls per
  extraction, chunk-level instead of per-claim.
- **Second axis, zero extra calls (post-worker):** the same ask carries a
  `cond` field — "does this source study the same health condition,
  symptom, or outcome the question asks about? An established name for
  the same condition counts; an adjacent or parent condition does NOT."
  The substance axis alone can't catch the sibling mismatch: a golden
  row asked acupuncture for chronic pelvic pain and was answered verbatim
  from a real RCT — on acupuncture for **fibromyalgia** (adjacent chronic
  pain, wrong population; grounded check 0.99 — the claim IS in the
  source — while `best_chunk` flagged the question↔source gap at 0.34
  and the eval label said semantic_refuted). Both axes share the gate:
  `ev >= ENTITY_MIN and cond >= COND_MIN`, dropped chunks recorded as
  `cond_scores` in the report dump for calibration.
- **COND_MIN = 0.1 — a veto at the bottom edge, not a gate:** at
  ENTITY_MIN (0.6) the axis over-fired — verified chunks on golden kept
  only 77% (verified-answer 94→79%, refuted-refusal 100%) because
  legit-but-loosely-matching conditions score mid-band. The measured
  distributions (530 cond calls): verified chunks mass 0.55–0.98,
  refuted chunks mass ≤0.10 — COND_MIN sits at the bottom edge of the
  verified tail (0.1: keeps 93% of verified chunks, drops 59% of
  refuted; the fibromyalgia offenders scored ≤0.05). Re-run: 88%
  verified-answer (3 transient judge `unavailable`s included),
  **100%** refuted-refusal, 0 dangerous.
- **Third field `has_cond` — the veto only exists when a condition is
  named:** 'Score 1 when the question names no specific condition'
  doesn't hold — definitional asks ('What is Capsicum?', 'How does
  Iodine work?') score cond 0.12–0.58 anyway: the judge anchors on the
  study's own population. The same ask now carries `has_cond` ("does
  the question name a specific health condition, symptom, or
  outcome?"). Measured separation: definitional asks 0.04–0.39,
  conditional asks 0.85–0.89 → `HAS_COND_MIN = 0.5`, and the gate
  becomes `ev >= ENTITY_MIN and (has_cond < 0.5 or cond >= COND_MIN)` —
  below 0.5 the cond axis is undefined and cannot veto. Golden rerun:
  zero has_cond-driven status changes (the definitional rows' chunks
  already scored cond ≥0.1 — the fix removes a wrong-condition veto
  path, it doesn't rescue golden rows); verified-answer 94%, refuted-
  refusal 100%, 0 dangerous. `has_cond_scores` dumps beside
  `cond_scores` for the report.
- **Fourth+fifth fields `rel`/`has_rel` — the relationship axis
  (post-heldout):** the heldout set exposed a class golden never had —
  5 MedQuAD "interactions between X and Y" rows answered on
  non-interaction evidence (statistical P<0.05 noise, efficacy papers).
  Substance and condition axes don't see it: right substance, no
  condition named — the relationship itself IS the question. `has_rel`
  gates it ("does the question ask a specific relationship —
  interactions, combinations, comparisons?"): measured 0.93–0.97
  relationship questions vs 0.03–0.09 single-substance. `rel` checks the
  source studies that relationship in its intended sense — first
  wording excluded combination trials (a real caffeine+coconut-oil
  combined-intake RCT scored 0.36 → false veto); widened to count
  combined/co-administration evidence: wrong-evidence ≤0.53 vs real
  interaction/combination studies ≥0.78 → `REL_MIN=0.6`. `question_score`
  and `sentence_score` carry the same intended-sense clause so the
  chunk/sentence scorers share the disambiguation. Result: heldout
  dangerous 6→4 (fabricated class dead — the statistical-interaction
  and efficacy-paper answers now abstain; residual is 3 scope-mismatch
  answers on *real* combination studies — 'herbs' answered with a
  medications review — plus 1 refuted-PMID leak).
  an actual question — "only if the checked input is itself a question;
  evidence cannot answer something that is not one". The invariant lives
  in the answerability score itself — one place, no input-shape list.
- **Why:** a pasted PubMed URL (`…/42943002/`, the melatonin-fertility
  paper — in the corpus) ran the whole pipeline: gate said `unrelated`,
  retrieval BM25-matched the exact paper on the PMID token, and the scope
  rescue scored topicality max — "does this source answer this question"
  never asked whether there WAS a question. Sentence scores then ranked
  noise (NMN sentences) against a URL string and the run answered
  confidently (verdict 0.97). Every check verified claim↔study; nothing
  verified input↔question.
- **Why one score, not per-stage clauses:** `question_score` is the
  only gate every answering path crosses — the main lane's chunk phase
  and the unrelated→scope rescue both call it. A non-question input
  fails there and nothing downstream runs; duplicating the check into
  `sentence_score` would guard a stage that unreachable input never
  reaches. And no URL regex: the shape varies (raw PMIDs, pasted
  abstracts), the invariant doesn't.
- **Measured live:** URL → scope 0.197, bare `PMID 42943002` → 0.158
  (both under SCOPE_MIN 0.4 → `off_topic`); control question on the
  same chunk 0.93 — real questions unaffected. 60 tests, ruff,
  pyrefly clean.

## ADR-018: Run usage ledger — every model call counted, priced only at published rates

- **Decision:** both entry points (`answer()`, `answer_events()`) open a
  `usage_log` bound once onto the injected `ask_fn`/`chat_fn`/`stream_fn` —
  every gate, scope, verify, extract, synth, and writer call appends
  `{fn, model, in, out, cached?, reasoning?}`. `trace.usage` (buffered) and
  the terminal `status` event (streamed) carry the totals.
- **Pricing:** `MODEL_PRICES` defaults to the router's upstream pool at
  published rates (deepseek-v4.1-flash → DeepInfra, glm-5.3-flash →
  docs.z.ai, qwen3.8-27b → Pearl Research, 2026-11); cached prompt tokens
  bill at the cached rate, not full input. `vllm-sr/*` models — the judge
  and the router itself — run on the free research API, publish no rate,
  and count as `unpriced` rather than get an invented number.
- **Why defaults, not a service:** the deployment's real cost today is
  $0 (free research/eval tier, quota-limited not dollar-billed). The
  table exists so cost evaluators can compare what the same tokens would
  spend upstream — env override swaps rates when the deployment changes.

## ADR-019: Worker deploy — Pyodide constraints mapped, not worked around

- **Decision:** `worker/src/worker.py` serves the identical
  `answer_events()` contract as a Python Worker — FastAPI over
  `workers.asgi`, lazily built (startup memory), same SSE frames the
  local `serve.py` emits so the exhibit UI needed zero changes.
- **Storage:** `chunks.jsonl` → D1 `chunks` (one row per chunk),
  `bm25.json` → D1 `bm25_postings` + `bm25_meta`, vectors already in
  Vectorize. The 35MB index never ships — a query reads only its own
  terms' postings. Stopword postings (~17k docs) exceed D1's statement
  size (`SQLITE_TOOBIG`), so postings shard per term
  (`PRIMARY KEY (term, shard)`, df duplicated per shard, ~3k pairs a
  row) and the loader folds each doc's length into its posting pair —
  no separate giant lens row. `len(DocView)` returns corpus N (idf
  needs the true document count) while `.items()` yields only docs
  containing the query's terms — absent docs score 0, so the math is
  identical to the in-memory index.
- **Pyodide constraints:** no sockets → judge/writer twins in
  worker.py call vllm-sr through sync `httpx2` (JS fetch under JS
  Promise Integration), injected through the same `ask_fn`/`stream_fn`
  parameters the pipeline already took (`stream_fn` was hardcoded in
  two places — now injected like `chat_fn`); no threads →
  `src/supplements_rag/_pool.py` returns a serial executor on
  `sys.platform == "emscripten"` (ThreadPoolExecutor constructs fine,
  then `submit()` dies — detection is at the platform, not the
  exception). Serial keeps `submit`/`as_completed` semantics identical;
  local keeps the pool.
- **Traces:** each ask opens with an *awaited* stub insert into the D1
  `runs` table (the `prepare_begin` event, `cold` flag when the isolate
  just booted `_META`) — written while the request context is certainly
  alive, because `wait_until` scheduled after a client disconnect never
  runs: the runtime cancels the request as hung and drops the scheduled
  coroutine. Completion `UPDATE`s the same row with the event log minus
  `token`/`think` events (replay cosmetics — ~90% of a writer-lane
  stream; the terminal `status` event already carries the assembled answer)
  (`INSERT` fallback if no stub exists); a killed stream keeps the stub
  — an abandoned run is an honest partial row, not silence. Same honesty
  as `data/runs.jsonl` locally. Each run gets an opaque public id
  (`uuid4().hex[:12]`, stored in `runs.rid`) — a sequential rowid would
  let anyone walk `/run/1…N` and read every visitor's stored question.
  The id rides the SSE stream as a `run` event right after
  `prepare_begin`, and `GET /sr/v1/run/{rid}` returns the stored record
  (`{"id", "ts", "q", "events"}`, JSON 404 when absent) — a visitor can
  pull their own run back, which is what makes the exhibit's
  inspectability claim real instead of wrangler-only. Local dev serves
  the same contract: `serve.py` exposes `/v1/run/{rid}` over
  `data/runs.jsonl`. `GET /sr/v1/health` is a real warm-up, not a shell —
  it pays the isolate's cold init (`_meta` D1 stats + one judge ping) so
  the exhibit's scroll-into-view health hit lands before the visitor's
  ask does; measured failure mode without it: the platform cancels
  cold-boot asks as hung before the run stub can even write.
- **Verified live:** prod ask "does magnesium help sleep?" → gate
  (evidence, injection 0.0002) → retrieve 10 PMIDs → extract → verdict
  0.785 → `status: answered` with 8 cited claims, 29 model calls,
  ~9.8s end-to-end, run persisted as `runs.id=6` (24.7KB events).

## ADR-020: `/v1/ping` — shallow liveness for probes, deep health for warm-up

`/sr/v1/health` is deliberately deep (meta + upstream ping, see the traces
ADR-019 bullet) — that depth is what makes it slow on a cold isolate
(~3.2s measured), slow enough for shields-style uptime badges to read
`down`. Rather than shallowing health (it's the visitor warm-up by
design), `/v1/ping` answers `{"ok": true}` with no meta load and no model
call — badges probe the cheap route, the deep one keeps its job.
