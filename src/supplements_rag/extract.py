"""Extractive answer lane — quotes instead of generated claims.

Why it exists: the router forces chain-of-thought on every request whose
payload is clinical abstracts (measured: medical content always routes
reasoning=on regardless of prompt shape, size, or instructions). That
thinking phase is most of the wait. This lane never calls the writer —
it scores pack chunks against the question, then scores the top chunks'
sentences, and the claims ARE the verbatim sentences. A quote is grounded
by construction: it cannot cite what it isn't.

The returned object is contract-shaped so the pipeline's checks run
unchanged — deterministic + per-claim judge still verify every kept
sentence. The verdict is a fixed label, not a claim, so verdict-level
grounding does not apply (the evidence lives in the claims).

CLI: python -m supplements_rag.extract "does magnesium help sleep" \
    --chunks data/chunks.jsonl --index data/bm25.json
"""

import re
from concurrent.futures import as_completed

from . import judge
from ._pool import executor
from .bm25 import entity_tokens

# sentences are only mined from the best-scoring chunks — the rest can't
# produce a sentence that reaches GROUNDED_MIN anyway
TOP_CHUNKS = 3
# the answer-question distribution sits lower than claim-support checks
# (a borderline quote still literally answers) — tuned on the golden
# slice: 0.6 lifts coverage; abstain rows stayed <=0.35 there
SENT_MIN = 0.6
# entity-identity sits at the poles for clear cases (same ≈0.8+ / wrong
# ≤0.55) but a true substance cited tangentially scores mid-band —
# tuned on the golden slice: 0.6 keeps betaine-in-a-malnutrition-study
# while griffonia-for-scopolia (0.55) still fails
ENTITY_MIN = 0.6
# entity-check runs on the top-ranked chunks plus a couple of
# replacement ranks — a dropped leader still leaves quote material
ENTITY_POOL = 5
# sentence ends only before a capital/digit — 'L. acidophilus', 'et al.',
# 'vs.' stay inside their sentence so quotes don't shatter mid-name
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WS = re.compile(r"\s+")


def question_score(question: str, chunk: dict, *, ask_fn=judge.ask, key=None) -> float:
    """noul score: does this chunk provide evidence answering the question?
    (question-shaped instruction — a question is not a claim). The input
    must be a real question — topical match is not answerability."""
    answers = ask_fn(
        f"Source:\n{chunk['title']}\n{chunk['text'][:12000]}\n\n"
        f"Question being checked: {question}",
        {
            "ev": {
                "type": "noul",
                "instructions": "Does the source provide evidence that "
                f"answers this question: {question} — only if the checked "
                "input is itself a question; evidence cannot answer "
                "something that is not one",
            }
        },
        key=key,
        stage="extract",
    )
    return answers["ev"]["noul"]


def sentence_score(
    question: str, sentence: str, title: str, *, ask_fn=judge.ask, key=None
) -> float:
    """noul score: does this single sentence answer the question? Same
    question-shape as question_score, scoped to one sentence so the quote
    is the unit that passes or fails."""
    answers = ask_fn(
        f"Source: {title}\nSentence: {sentence}\n\nQuestion: {question}",
        {
            "ev": {
                "type": "noul",
                "instructions": "Does this sentence provide evidence that "
                f"answers this question: {question}",
            }
        },
        key=key,
        stage="extract",
    )
    return answers["ev"]["noul"]


def entity_score(question: str, chunk: dict, *, ask_fn=judge.ask, key=None) -> float:
    """noul score: does this SOURCE study the same substance the question
    names? Scored at chunk level, not sentence level — the abstract body
    defines what a bare quote can't (a trial codenamed 'ART' names its
    substance in the full text). A dedicated single-criterion check:
    folded into question/sentence_score the judge trades identity off
    against topical relevance (myricetin-for-myrcia scored 0.81 folded
    into sentence_score, 0.31 as this dedicated check)."""
    answers = ask_fn(
        f"Source:\n{chunk['title']}\n{chunk['text'][:12000]}\n\nQuestion: {question}",
        {
            "ev": {
                "type": "noul",
                "instructions": "Does this source study the same "
                "substance or intervention the question names? Count "
                "exact identity or an established name for the identical "
                "thing — including a codename or abbreviation the source "
                "itself defines. Constituents, derivatives, and related "
                "compounds are NOT the same substance.",
            }
        },
        key=key,
        stage="extract",
    )
    return answers["ev"]["noul"]


def candidates(top: list) -> list[tuple]:
    """(pmid, title, sentence) for every quotable sentence in the scored
    chunks — 40–400 chars, whitespace collapsed."""
    return [
        (c["pmid"], c["title"], _WS.sub(" ", s).strip())
        for _, c in top
        for s in _SENT.split(c["text"])
        if 40 < len(s) < 400
    ]


def keep(cands: list[tuple], sscores: list[float]) -> list[dict]:
    """scores → claims: every sentence >= SENT_MIN, best first, PMID as the
    id — no cap: each kept quote renders as a stamp, so a wide rail is a
    feature (more checked evidence), not a wall of text."""
    kept = sorted(
        ((s, pmid, txt) for s, (pmid, _, txt) in zip(sscores, cands) if s >= SENT_MIN),
        key=lambda x: -x[0],
    )
    return [{"text": txt, "ids": [pmid]} for _, pmid, txt in kept]


def _pool_iter(fn, items, phase):
    """Score items in parallel — yields (phase, done, n) as each result
    lands (progress drives the streamed lane's `extract` events), returns
    the ordered score list."""
    yield (phase, 0, len(items))
    out: dict[int, float] = {}
    with executor(max_workers=6) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            yield (phase, len(out), len(items))
    return [out[i] for i in range(len(items))]


# sentences sharing a source ride one judge call as typed questions
# (was: one call per sentence); cap bounds the per-call question count
_SENT_BATCH = 15


def _sent_iter(question: str, cands: list[tuple], *, ask_fn, key):
    """_pool_iter for the sentences phase, batched per source chunk."""
    groups: dict[str, list[int]] = {}
    for i, (pmid, _, _) in enumerate(cands):
        groups.setdefault(pmid, []).append(i)

    def call(idxs):
        title = cands[idxs[0]][1]
        answers = ask_fn(
            f"Source: {title}\n\nQuestion: {question}",
            {
                f"s{j}": {
                    "type": "noul",
                    "instructions": "Does this sentence provide evidence "
                    f"that answers this question: {question}\n\n"
                    f"Sentence: {cands[i][2]}",
                }
                for j, i in enumerate(idxs)
            },
            key=key,
            stage="extract",
        )
        return {i: answers[f"s{j}"]["noul"] for j, i in enumerate(idxs)}

    yield ("sentences", 0, len(cands))
    out: dict[int, float] = {}
    batches = [
        idxs[s : s + _SENT_BATCH]
        for idxs in groups.values()
        for s in range(0, len(idxs), _SENT_BATCH)
    ]
    with executor(max_workers=6) as ex:
        for f in as_completed([ex.submit(call, b) for b in batches]):
            out |= f.result()
            yield ("sentences", len(out), len(cands))
    return [out[i] for i in range(len(cands))]


def claims_iter(
    question: str,
    pack: list[dict],
    index: dict | None = None,
    *,
    ask_fn=judge.ask,
    key=None,
    meta: dict | None = None,
):
    """claims() as a generator — yields (phase, done, n) progress tuples as
    judge scores land (phases: 'chunks', 'entity' when the rare-entity
    gate fires, 'sentences'). Returns (claims, best) via StopIteration."""
    cscores = yield from _pool_iter(
        lambda c: question_score(question, c, ask_fn=ask_fn, key=key),
        pack,
        "chunks",
    )
    best = max(cscores, default=0.0)
    ranked = sorted(zip(cscores, pack), key=lambda x: -x[0])
    pool = ranked[:TOP_CHUNKS]
    if index is not None and entity_tokens(index, question):
        escores = yield from _pool_iter(
            lambda c: entity_score(question, c, ask_fn=ask_fn, key=key),
            [c for _, c in ranked[:ENTITY_POOL]],
            "entity",
        )
        passed = [x for x, s in zip(ranked[:ENTITY_POOL], escores) if s >= ENTITY_MIN]
        if meta is not None:
            meta["entity_scores"] = escores
            meta["entity_dropped"] = len(ranked[:ENTITY_POOL]) - len(passed)
        pool = passed[:TOP_CHUNKS]
    cands = candidates(pool)
    sscores = yield from _sent_iter(question, cands, ask_fn=ask_fn, key=key)
    return keep(cands, sscores), best


def claims(
    question: str,
    pack: list[dict],
    index: dict | None = None,
    *,
    ask_fn=judge.ask,
    key=None,
    meta: dict | None = None,
) -> tuple[list[dict], float]:
    """(claims, best chunk score) — verbatim sentences >= SENT_MIN,
    best first, each citing its own PMID. When `index` is given and the
    question names a corpus-rare/absent entity, only chunks the entity
    check confirms can produce quotes — a wrong-entity chunk's sentences
    can't answer for the named substance (meta collects the scores).
    JudgeError propagates: the pipeline fails closed on a judge outage."""
    it = claims_iter(question, pack, index, ask_fn=ask_fn, key=key, meta=meta)
    try:
        while True:
            next(it)
    except StopIteration as e:
        return e.value


DIR_PHRASES = {
    "supports": "the studies say yes",
    "contradicts": "the studies say no",
    "mixed": "the studies are split",
    "speaks": "the studies answer in their own words",
}
DIR_FALLBACK = "the studies answer —"


def direction_for(
    question: str, claims: list[dict], *, ask_fn=judge.ask, key=None
) -> str:
    """One typed judge call → the evidence's direction WORD (supports /
    contradicts / mixed / speaks). Typed and mechanical — the verdict
    phrase derives from it, and the client can render the direction as
    chrome instead of leaving the word inside the prose."""
    quotes = "\n".join(f"- {c['text']}" for c in claims)
    answers = ask_fn(
        f"Question: {question}\n\nEvidence quoted verbatim from studies:\n{quotes}",
        {
            "dir": {
                "type": "choice",
                "instructions": "Given ONLY the quoted evidence, which "
                "direction does it point for this question? Choose "
                "'speaks' when the question is not a yes/no question "
                "(e.g. 'what is X?', 'how does X work?') and the quotes "
                "simply state what the studies found.",
                "criteria": {
                    "supports": "the quoted evidence supports a positive "
                    "answer or reports the asked-about effect",
                    "contradicts": "the quoted evidence argues against it "
                    "or reports no/negative effect",
                    "mixed": "the quotes point in different directions or "
                    "are clearly inconclusive",
                    "speaks": "not a yes/no question — the evidence just states facts",
                },
            }
        },
        key=key,
        stage="extract",
    )
    return answers["dir"].get("choice") or ""


def verdict_for(
    question: str, claims: list[dict], *, ask_fn=judge.ask, key=None
) -> str:
    """Direction word → the bottom-line phrase. A direction is a check
    output (typed, mechanical) — not generated prose; the quoted claims
    below it stay the proof."""
    return DIR_PHRASES.get(
        direction_for(question, claims, ask_fn=ask_fn, key=key), DIR_FALLBACK
    )


def compose(claims: list[dict], verdict: str | None = None) -> dict:
    """claims + bottom line → contract-shaped object. Empty claims = the
    pack holds no covering sentence — 'unclear' is the honest status, and
    'missing' names what the checks could not find."""
    if not claims:
        return {
            "status": "unclear",
            "verdict": None,
            "claims": [],
            "missing": [
                (
                    "no sentence in the retrieved evidence answers the "
                    "question directly (nothing scored high enough to quote)"
                )
            ],
        }
    return {
        "status": "answered",
        "verdict": verdict or "the strongest evidence in the pack, verbatim:",
        "claims": claims,
        "missing": [],
    }


if __name__ == "__main__":
    # python -m supplements_rag.extract "does magnesium help sleep"
    #   → the sentences it would quote, with their scores
    import json
    import os
    import sys
    from pathlib import Path

    from .bm25 import score as bm25_score

    q = sys.argv[1]
    root = Path(__file__).resolve().parents[2]
    index = json.loads((root / "data" / "bm25.json").read_text())
    chunks = {
        json.loads(l)["chunk_id"]: json.loads(l)
        for l in (root / "data" / "chunks.jsonl").open()
    }
    pack = [chunks[c] for c, _ in bm25_score(index, q, top_k=10) if c in chunks]
    key = os.environ["VLLM_SR_API_KEY"]
    meta: dict = {}
    got, best = claims(q, pack, index, key=key, meta=meta)
    print(json.dumps({"best_chunk_score": best, **meta, "claims": got}, indent=1))
