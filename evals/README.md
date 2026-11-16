# Evaluation set

The golden set answers one question: **does the system retrieve and cite real
evidence — and does it refuse or abstain when it should?** Nothing here is
hand-picked because it looks good; every row was admitted by a mechanical
rule over the actual corpus.

## Layout

```
evals/
├── seeds/               # question sources (committed — the inputs)
│   ├── medquad.jsonl        # 792 questions · 99 supplements · NIH MedQuAD
│   └── suppllment.jsonl     # 4,777 effectiveness items · SuppLLMent labels
├── admit.py             # seeds → admission pipeline (the mechanical rule)
├── run.py               # golden → retrieval metrics → report.json
├── verify.py            # evidence PMIDs → vllm-sr judge scores → rows file
├── eval_pipeline.py     # golden questions → pipeline.answer() → per-row outcomes
├── golden.jsonl         # 120-row stratified eval set (committed — the contract)
├── report.json          # committed eval numbers
├── pipeline-report*.jsonl  # per-run end-to-end outcomes (vN = named run)
├── pipeline-summary*.json  # per-run outcome matrix
├── admitted.jsonl       # full candidate pool (generated, gitignored)
└── rejected.jsonl       # every dropped seed + reason (generated, gitignored)
```

Regenerate: `python evals/admit.py` (requires `data/bm25.json`).

## Admission rule

A seed is admitted only when its evidence can be verified inside the local
corpus — BM25 retrieves at least one chunk scoring above threshold
(`--bm25-min`, default 8.0), and the top-3 PMIDs are attached as candidate
evidence. Every rejection lands in `rejected.jsonl` with its reason.

| Category | Expected behavior | Source | Count |
| --- | --- | --- | --- |
| `answerable` | `answer_with_citations` | MedQuAD info/effective/how-it-works + SuppLLMent positive labels, covered | 1,832 |
| `negative` | `answer_with_citations` | SuppLLMent `*ineffective_for` labels, covered — cite the no-benefit evidence | 1,561 |
| `advice` | `answer-or-decline` | MedQuAD dose/interactions/precautions — these are *safety-information* questions (evidence-answerable), not "should I take it": a cited evidence answer is correct, a decline is defensible; the eval records the gate's per-row intent rather than asserting decline | 495 |
| `abstain` | `abstain` | SuppLLMent `insufficient_evidence`, no coverage | 138 |

Rejected (1,543): `no_corpus_coverage` 98 ·
`labeled_insufficient_but_covered` 1,445 (the corpus mentions the pair but
the label says insufficient — needs review, not silent admission).

## First measured results (`python evals/run.py` → `evals/report.json`)

| Category | n | retention@5 | @10 | BM25 top median | range | dense top |
| --- | --- | --- | --- | --- | --- | --- |
| answerable | 50 | 0.92 | 0.98 | 18.99 | 10.99–37.00 | 0.49 |
| negative | 30 | 0.93 | 1.00 | 18.03 | 9.87–34.01 | 0.47 |
| abstain | 15 | — | — | 9.63 | 4.70–11.98 | 0.44 |
| advice | 25 | — | — | 18.74 | 11.13–29.65 | 0.49 |

*(2026-11 rerun: production-path eval — `retrieve()` incl. rare-entity
injection, keyword-form dense queries; negative@5 dipped 1.00→0.93 as
injected entity chunks take top-5 slots — proposed evidence still lands
top-10.)*

What the numbers do and don't say:

- **Retention ≠ recall.** Evidence PMIDs are BM25-proposed, so retention@10
  (0.98–1.00) only proves RRF fusion doesn't demote proposed evidence out of
  the top-k — it is not a ground-truth recall claim.
- **Abstention signal separates, barely.** Abstain max BM25 (11.97) vs
  answerable min (10.99) → a ~1-point overlap band. An abstention threshold
  needs a gray zone (abstain / uncertain / answer), not a single cutoff.
- **Dense similarity barely separates** (0.49 answerable vs 0.44 abstain
  after keyword-form embedding — the port didn't lift separation here;
  ADR-012 records it honestly). Vectorize returns 20 nearest hits
  regardless; its score alone can't drive abstention.
- **Advice questions retrieve strong evidence** (BM25 18.66 median) — the
  corpus *can* answer "what's the dose of X", which is exactly why the
  decline contract must fire before evidence reaches the answer. The naive
  advice-pattern catches 25/25 by construction (qtype vocabulary ⊂ pattern);
  the real gate test is semantic.

## End-to-end eval (`evals/eval_pipeline.py` → `evals/pipeline-{report,summary}.json`)

All 120 golden questions through `pipeline.answer()` — outcome, gate
intent, repair count, correctness per row. Two runs exist (v1 = before the
unclear-rescue, v2 = current); outcomes vary run-to-run because writer
sampling is non-deterministic.

Headline numbers across runs (correct rate by golden label):

| Run | answerable | negative | abstain | advice | What changed |
| --- | --- | --- | --- | --- | --- |
| v1 | 22% | 17% | 100% | 20% | baseline — writer over-`unclear` |
| v2 | 22% | 17% | 100% | 36% | +unclear-rescue, prompt: negative findings are answers |
| v3 | **36%** | **27%** | 100% | 36% | +`verdict` field, number-word expansion, rounding, exact-p-value rule, temp 0 |
| v4 | 38% | 20% | 93% | 24% | +rescue question-shaped instruction + window 8 |
| v5 | 62% | 33% | 87% | 48% | extractive lane as default (2 dangerous: wrong-entity answers) |
| v6 | 52% | 23% | **100%** | 40% | +per-claim entity check (extractive lane only) |
| v7 | 52% | 23% | **100%** | 44% | +`ENTITY_MIN` 0.6 calibrated to the entity task |

v4 detail (dangerous = abstain-category answered): **1** — "Is myrcia
effective for diabetes?" answered with honest limits ("closest evidence is
myricetin — mouse studies only, no human evidence"). Calibrated, not a
fabricated claim, but still answered where the label said abstain → counts
dangerous under the strict rule. v4 also exposed a scope-classification
bug: CAM questions (tai chi, acupuncture, massage) gated `unrelated`
despite in-corpus evidence → gate criteria + scope-check instruction
fixed after v4 (verified live on the failure classes).

**The number that reframes the eval:** split by `verification` — the
golden labels are coverage-based, not evidence-confirmed:

| Row truth | n | v5 answered | v7 answered | Note |
| --- | --- | --- | --- | --- |
| semantic_verified | 33 | 32 (97%) | 29 (88%) | v7 losses: quotes that can't self-identify their substance (codenames like `ART`, unnamed interventions) |
| semantic_partial | 14 | 7 | 4 | partial evidence → unclear is often defensible |
| semantic_refuted | 33 | 2 | **0** | both v5 leaks were wrong-entity evidence (spirulina-for-limonene, OraCMU-for-tin) |
| retrieval_only | 40 | 11 | 8 | includes the advice rows; **dangerous count: v5 = 2, v7 = 0** |

**The v7 trade, stated plainly:** a dedicated per-claim entity check
("does this evidence study the same substance the question names — exact
identity or established names only, not constituents/derivatives/
related compounds") eliminates the wrong-entity class entirely — asked
myrcia, cited myricetin; asked scopolia, cited griffonia; asked safflower,
cited nigella. It costs ~3 answers where the corpus holds the evidence
but only under a name the quote can't show (a trial codenamed `ART`, an
intervention named nowhere in the abstract's quotable span). For a
verbatim-quote product, a quote that can't demonstrate its own entity is
the wrong thing to ship — the trade is principled, recorded here so it
reads as a decision, not a regression.

*The one refuted→answered row is a `negative`-category question ("Is
laminaria effective for diabetes?") answered honestly: "no direct evidence
shows laminaria helps diabetes — the only relevant evidence is for
glucomannan, a related fiber." A correct negative verdict, not a
hallucination — `semantic_refuted` flagged the proposed PMIDs, not the
achievable answer.

- `semantic_refuted` (corpus lacks evidence) → the system refuses almost
  everywhere it should.
- `semantic_verified` (evidence confirmed present) → 70% answered after
  the contract/verify fixes; remaining abstains are verify strictness +
  residual generation flakes (repair recovers most single-fault attempts).
- `retrieval_only` (labels only, no semantic confirmation) → the unclear
  volume here is probably *correct* abstention, not failure.

## Semantic verification (`VLLM_SR_API_KEY=... python evals/verify.py --repropose`)

Every proposed evidence PMID is scored by the vllm-sr SystemOne judge —
a semantic-router endpoint returning a 0–1 support score — answering "does
this abstract provide evidence that answers the question?" Bands (assumption): ≥0.7 yes / ≥0.4 partial / &lt;0.4 no; raw scores
are stored per row so bands can re-derive. `--repropose` gives unverified
rows a deeper pool (BM25 top-10 + dense top-10) before "no evidence" stands.

**Result on golden:** 33 semantic_verified · 14 semantic_partial ·
33 semantic_refuted · 40 retrieval_only (abstain/advice carry no evidence).

**Result on the full admitted pool** (`--rows evals/admitted.jsonl
--verdicts data/verdicts-pool.json`, 9,987 judge calls): of 4,026 rows —
**968 semantic_verified (24%)** · 342 semantic_partial (8%) ·
**2,080 semantic_refuted (52%)** · 633 retrieval_only (no evidence
proposed) · 3 verification_error. The golden set (27.5% verified) is
representative of the pool. The headline: **over half of all questions the
coverage rule admitted have no real evidence in the corpus** — lexical
admission is that noisy, which is why semantic verification and abstention
are load-bearing stages rather than polish.

The big measured finding: **BM25 top-3 proposed evidence is wrong ~55% of
the time** on pair-specific questions — it matches substance-OR-condition,
not substance-AND-condition (e.g. "laxogenin for breath-holding" retrieved
iron-for-breath-holding + laxogenin-adjacent migraine trials, judged
0.03–0.13). Re-proposal recovered ~25% of refuted rows — real evidence sat
below top-3. Conclusion: coverage is not evidence; semantic verification
is a required stage, and refuted rows are kept flagged as hard negatives.

## Honest limits

- `evidence_pmids` on the 40 abstain/advice rows remain `retrieval_only`
  by design (no evidence to check). Verified rows carry real support scores,
  but the judge is one model — `d3-flash` spot errors are possible; bands
  and the prompt are stated above so the pass is re-runnable.
- MedQuAD answers are empty by source design — used as question seeds only.
- SuppLLMent rows that didn't match the `Is 'X' effective at treating 'Y'`
  pattern (89 of 4,866) were dropped at seed build.
- 11 of 99 MedQuAD supplements have no corpus coverage; their answerable
  questions were rejected, their advice questions kept (coverage is
  irrelevant for decline tests).

## Sources

- MedQuAD `12_MPlusHerbsSupplements_QA` — NIH, via
  github.com/abachaa/MedQuAD (questions only; answer text removed upstream)
- SuppLLMent benchmark file (effectiveness labels:
  effective / likely / possibly / ineffective / insufficient_evidence)
