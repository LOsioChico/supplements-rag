"""gate → retrieve → generate → verify → done — the answering pipeline.

Order and the reasons for it (ADR-012):
  1. gate screens the RAW question (injection, advice) before retrieval —
     advice questions pull strong evidence, so retrieval can't be the gate
  2. retrieval builds the evidence pack (BM25 + dense + rare-entity)
  3. scope: an 'unrelated' intent is not an instant veto — the judge sees
     the evidence before deciding (gray-zone, not a cutoff)
  4. the writer emits the strict contract — streamed deltas render as
     provisional (stamped 'checking'); the settled answer waits for verify
  5. verify: deterministic (cited ids ∈ retrieved, numbers ⊆ sources)
     then per-claim judge groundedness — FAIL CLOSED on judge outage
  6. ONE pass, no rewrite: a failed check marks its sentence (and the
     bottom line) as unverified on the answer — only unparseable
     output still abstains

`ask_fn` / `chat_fn` are injectable for tests. CLI:
  python -m supplements_rag.pipeline "does magnesium help sleep" \
      --dense data/dense.json
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from . import contract, extract, gate, generate, judge, verify
from .query import retrieve

# USD per 1M tokens for the router's upstream pool (published rates, 2026-11):
#   deepseek-v4.1-flash — DeepInfra standard tier
#   glm-5.3-flash — docs.z.ai official
#   qwen3.8-27b — Pearl Research
# vllm-sr/* models (the judge + router itself) run on the free research API —
# no published rate exists, so they stay unpriced rather than guessed.
# MODEL_PRICES env overrides the whole table, e.g. '{"m":{"in":0.1,"out":0.2}}'.
MODEL_PRICES: dict = json.loads(
    os.environ.get(
        "MODEL_PRICES",
        json.dumps(
            {
                "deepseek/deepseek-v4.1-flash": {
                    "in": 0.20,
                    "cached": 0.006,
                    "out": 0.60,
                },
                "zai/glm-5.3-flash": {"in": 0.15, "cached": 0.03, "out": 0.50},
                "qwen/qwen3.8-27b": {"in": 0.20, "cached": 0.05, "out": 2.00},
            }
        ),
    )
)


def _row_cost(row: dict, price: dict) -> float:
    """One ledger row's USD — cached prompt tokens bill at the cached
    rate, not full input."""
    cached = row.get("cached") or 0
    return (
        ((row.get("in") or 0) - cached) * price.get("in", 0)
        + cached * price.get("cached", price.get("in", 0))
        + (row.get("out") or 0) * price.get("out", 0)
    ) / 1e6


def _usage_totals(calls: list[dict]) -> dict:
    """Run ledger → totals: per-model and per-stage breakdowns + USD cost
    when MODEL_PRICES names the model — unpriced models count, never
    estimated."""
    by_model: dict[str, dict] = {}
    by_stage: dict[str, dict] = {}
    unpriced = 0
    unpriced_tok = 0
    cost = 0.0
    for c in calls:
        m = c.get("model") or "?"
        b = by_model.setdefault(m, {"calls": 0, "in": 0, "out": 0})
        s = by_stage.setdefault(
            c.get("stage") or c.get("fn") or "?",
            {"calls": 0, "in": 0, "out": 0, "model": m},
        )
        b["calls"] += 1
        b["in"] += c.get("in") or 0
        b["out"] += c.get("out") or 0
        s["calls"] += 1
        s["in"] += c.get("in") or 0
        s["out"] += c.get("out") or 0
        if s["model"] != m:
            s["model"] = "mixed"
        p = MODEL_PRICES.get(m)
        if p:
            cst = _row_cost(c, p)
            cost += cst
            b["cost"] = round(b.get("cost", 0) + cst, 6)
            s["cost"] = round(s.get("cost", 0) + cst, 6)
        else:
            unpriced += 1
            # the free tier's real meter is the provider's daily token
            # quota (x-quota-tokens-per-day) — sum it so the UI can show
            # consumption instead of inventing a price
            unpriced_tok += (c.get("in") or 0) + (c.get("out") or 0)
    out = {
        "calls": len(calls),
        "in": sum(b["in"] for b in by_model.values()),
        "out": sum(b["out"] for b in by_model.values()),
        "by_model": by_model,
        "by_stage": by_stage,
    }
    if unpriced:
        out["unpriced"] = unpriced
        out["unpriced_tok"] = unpriced_tok
    if cost:
        out["cost_usd"] = round(cost, 6)
    return out


ADVICE_DECLINE = (
    "I can share what the research says about supplements, but I can't "
    "give personal recommendations or dosing advice — that's a "
    "conversation for a clinician who knows your history."
)
OFF_TOPIC = (
    "This tool answers questions about dietary-supplement research. "
    "That question doesn't look related, and the evidence base has "
    "nothing for it."
)
ABSTAIN = (
    "I found related research but couldn't verify a supported answer from "
    "it — the honest answer is that the evidence here isn't strong "
    "enough to say."
)
UNAVAILABLE = (
    "I can't process this reliably right now (a check the answer depends "
    "on is unavailable). Please try again shortly."
)

SCOPE_MIN = 0.4


def _rescue(question: str, pack: list[dict], *, ask_fn, key) -> float:
    """When the writer says 'unclear', score each pack chunk against the
    QUESTION — ≥0.7 means the evidence was there and 'unclear' was wrong.
    The symmetric check to verify(): it rejects unsupported REFUSALS,
    not just unsupported answers. Window = what the writer saw."""
    return max(
        extract.question_score(question, c, ask_fn=ask_fn, key=key)
        for c in pack[: generate.EVIDENCE_CAP]
    )


def _verdict_numbers(obj: dict, retrieved_ids: set, chunks_by_id: dict) -> list:
    """The verdict text gets the same number check as claims — its numbers
    are verified against the UNION of all cited sources (a synthesis claim
    draws on several abstracts at once)."""
    all_ids = sorted({str(x) for c in obj["claims"] for x in c["ids"]})
    return verify.deterministic(
        {"claims": [{"text": obj["verdict"], "ids": all_ids}]},
        retrieved_ids,
        chunks_by_id,
    )


_ANY_STR = re.compile(r'"((?:[^"\\]|\\.){20,})"')


_POINT = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"ids"\s*:\s*\[([^\]]*)\]')


def _scan_points(raw: str) -> list[dict]:
    """Best-effort points recovery from a chewed buffer — the verdict
    fallback rescues a bare string, so scan the points array for
    COMPLETE {"text": "...", "ids": [...]} objects. Anything partial or
    non-numeric drops; every survivor still faces the grounding check."""
    i = raw.find('"points"')
    if i < 0:
        return []
    out = []
    for m in _POINT.finditer(raw[i:]):
        try:
            text = json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            continue
        out.append({"text": text, "ids": re.findall(r"\d+", m.group(2))})
    return out[:4]


def _synth_obj(raw: str, canary: str) -> dict | None:
    """Parse {"verdict","points"} from a summarize() reply — clean JSON
    first, then the writer stream's field scan, then the longest string
    in the buffer: the reasoning→content boundary can drop the {"ver
    prefix into a reasoning delta, leaving just the verdict text inside
    a chewed wrapper. A canary echo fails it. Malformed points drop
    silently — the verdict stands alone rather than losing the answer."""
    if canary in raw:
        return None
    verdict = points_raw = None
    try:
        obj = json.loads(raw.strip())
        verdict = obj.get("verdict")
        points_raw = obj.get("points")
        if not isinstance(verdict, str):
            verdict = None
    except (json.JSONDecodeError, AttributeError):
        pass
    if verdict is None:
        verdict = _scan_verdict(raw)
    if verdict is None:
        m = _ANY_STR.search(raw)
        if m:
            try:
                verdict = json.loads(f'"{m.group(1)}"')
            except json.JSONDecodeError:
                verdict = m.group(1)
    if not isinstance(verdict, str) or not verdict.strip():
        return None
    points = []
    if isinstance(points_raw, list):
        for p in points_raw[:4]:
            if (
                isinstance(p, dict)
                and isinstance(p.get("text"), str)
                and p["text"].strip()
                and isinstance(p.get("ids"), list)
            ):
                points.append({"text": p["text"], "ids": [str(x) for x in p["ids"]]})
    return {"verdict": verdict, "points": points or _scan_points(raw)}


def _check_points(points: list[dict], claims: list[dict], *, ask_fn, key) -> list[dict]:
    """Judge each bullet against the quotes it cites — a point with no
    cited quote, or one scored under the bar, simply never reaches the
    page. Same fail-safe as the verdict: generated prose only shows when
    the checks back it."""
    todo = [
        (p, [c for c in claims if set(p["ids"]) & {str(x) for x in c["ids"]}])
        for p in points[:4]
    ]
    todo = [(p, cited) for p, cited in todo if cited]
    if not todo:
        return []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(_synth_grounded, p["text"], cited, ask_fn=ask_fn, key=key)
            for p, cited in todo
        ]
        out = []
        for (p, _), f in zip(todo, futs):
            s = f.result()
            if s >= verify.GROUNDED_MIN:
                out.append({**p, "score": round(s, 3)})
        return out


def _verdict_grounded(obj: dict, chunks_by_id: dict, *, ask_fn, key) -> float:
    """Score the verdict against the COMBINED cited evidence — a verdict
    synthesizes across sources, so judging it against any single source is
    structurally wrong."""
    cited = [
        chunks_by_id[str(x)]
        for c in obj["claims"]
        for x in c["ids"]
        if str(x) in chunks_by_id
    ]
    return verify.claim_score(
        obj["verdict"],
        {
            "title": "Combined evidence",
            "text": "\n\n".join(c["text"][:3000] for c in cited)[:12000],
        },
        ask_fn=ask_fn,
        key=key,
    )


def _synth_grounded(verdict: str, claims: list[dict], *, ask_fn, key) -> float:
    """Score the generated bottom line against the QUOTES it summarized —
    the writer only saw the quotes, so a fair summary of them is grounded
    even when the full abstracts hold more than the quotes carry. The
    instruction is summary-shaped, not claim-shaped: the verdict is a
    bottom line over evidence, not a literal claim to match."""
    answers = ask_fn(
        "Evidence quotes:\n\n"
        + "\n\n".join(c["text"] for c in claims)
        + f"\n\nBottom line being checked: {verdict}",
        {
            "sup": {
                "type": "noul",
                "instructions": "Is this bottom line a fair, direct "
                "summary of the quoted evidence?",
            }
        },
        key=key,
        stage="verify",
    )
    return answers["sup"]["noul"]


def _scope(question: str, pack: list[dict], *, ask_fn, key) -> float:
    """Best question-evidence score over the top-3 chunks — the caller
    thresholds it and the trace keeps the raw score. (unrelated intent
    only — second look before refusing; title-only 'could this answer'
    under-scored covered CAM questions.)"""
    return max(
        extract.question_score(question, c, ask_fn=ask_fn, key=key) for c in pack[:3]
    )


def _ms(t: float) -> int:
    return int((time.monotonic() - t) * 1000)


def answer(
    question: str,
    dense: list,
    chunks_by_id: dict,
    index: dict,
    *,
    ask_fn=judge.ask,
    chat_fn=generate.chat,
    key: str | None = None,
) -> dict:
    t0 = time.monotonic()
    # trace: stage name → {raw scores, ms}; attempts[] holds each try's
    # raw output, contract/verify errors, and its judge scores
    trace: dict = {"attempts": []}
    # the token ledger — bound once here, it rides the injected-fn
    # plumbing to every gate/scope/verify/extract/synth call
    usage_log: list[dict] = []
    ask_fn = partial(ask_fn, usage=usage_log)
    chat_fn = partial(chat_fn, usage=usage_log, stage="write")

    def done() -> dict:
        trace["ms"] = _ms(t0)
        trace["usage"] = _usage_totals(usage_log)
        return trace

    t = time.monotonic()
    screen = gate.screen(question, ask_fn=ask_fn, key=key)
    trace["screen"] = {**screen, "ms": _ms(t)}
    if screen["injected"]:
        return {"status": "refused", "answer": None, "trace": done()}
    if screen["intent"] == "advice":
        return {"status": "declined", "answer": ADVICE_DECLINE, "trace": done()}
    if screen["intent"] == "uncertain":
        return {"status": "unavailable", "answer": UNAVAILABLE, "trace": done()}

    t = time.monotonic()
    pack = retrieve(index, chunks_by_id, question, dense, top_k=10)
    trace["retrieve"] = {"pack": [c["pmid"] for c in pack], "ms": _ms(t)}
    if screen["intent"] == "unrelated":
        try:
            t = time.monotonic()
            scope = _scope(question, pack, ask_fn=ask_fn, key=key)
            trace["scope"] = {"score": scope, "ms": _ms(t)}
        except judge.JudgeError:
            return {"status": "unavailable", "answer": UNAVAILABLE, "trace": done()}
        if scope < SCOPE_MIN:
            return {"status": "off_topic", "answer": OFF_TOPIC, "trace": done()}

    canary = generate.new_canary()
    retrieved_ids = {str(c["pmid"]) for c in pack}
    citations = {
        str(c["pmid"]): {
            "title": c["title"],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}/",
            "journal": c.get("journal", ""),
            "year": str(c.get("pub_date", ""))[:4],
            "superseded_by": c.get("superseded_by"),
        }
        for c in pack
    }

    if os.environ.get("EXTRACTIVE", "1") != "0":
        return _extractive_answer(
            question,
            pack,
            retrieved_ids,
            chunks_by_id,
            citations,
            index,
            ask_fn=ask_fn,
            chat_fn=chat_fn,
            key=key,
            trace=trace,
            finish=done,
        )

    # one pass — a failed check marks its sentence on the answer instead
    # of buying a hidden rewrite; only unparseable output still abstains
    t = time.monotonic()
    t_w = t
    raw = generate.write(question, pack, canary, chat_fn=chat_fn)
    att: dict = {"write_ms": _ms(t_w)}
    obj, why = contract.parse(raw)
    if canary in raw:
        why = "canary echo"
        obj = None
    errors: list[str]
    if obj is None:
        errors = [why or "contract failure"]
    else:
        errors = verify.deterministic(obj, retrieved_ids, chunks_by_id)
        if not errors and obj["status"] == "answered" and obj.get("verdict"):
            errors = _verdict_numbers(obj, retrieved_ids, chunks_by_id)
        if not errors:
            try:
                errors, att["grounded"] = verify.grounded(
                    obj, chunks_by_id, ask_fn=ask_fn, key=key
                )
                if not errors and obj["status"] == "answered":
                    att["verdict_score"] = vs = _verdict_grounded(
                        obj, chunks_by_id, ask_fn=ask_fn, key=key
                    )
                    if vs < verify.GROUNDED_MIN:
                        errors = [f"verdict unsupported (score {vs:.2f})"]
            except judge.JudgeError as e:
                att.update({"output": raw[:8000], "errors": errors, "ms": _ms(t)})
                trace["attempts"].append(att)
                done()
                trace["judge_error"] = str(e)
                return {
                    "status": "unavailable",
                    "answer": UNAVAILABLE,
                    "trace": trace,
                }
        if not errors and obj["status"] == "unclear":
            try:
                att["rescue_score"] = s = _rescue(
                    question, pack, ask_fn=ask_fn, key=key
                )
                if s >= verify.GROUNDED_MIN:
                    errors = [
                        (
                            "evidence exists in the pack (support "
                            f"{s:.2f}) — the evidence was there; 'unclear' "
                            "was the wrong call"
                        )
                    ]
            except judge.JudgeError as e:
                att.update({"output": raw[:8000], "errors": errors, "ms": _ms(t)})
                trace["attempts"].append(att)
                done()
                trace["judge_error"] = str(e)
                return {
                    "status": "unavailable",
                    "answer": UNAVAILABLE,
                    "trace": trace,
                }
    att.update({"output": raw[:8000], "errors": errors, "ms": _ms(t)})
    trace["attempts"].append(att)

    if obj is None:
        return {
            "status": "abstained",
            "answer": ABSTAIN,
            "errors": errors,
            "trace": done(),
        }
    if obj["status"] == "refused":
        return {"status": "refused", "answer": None, "trace": done()}
    if obj["status"] == "unclear":
        return {
            "status": "unclear",
            "answer": ABSTAIN,
            "missing": obj["missing"],
            "citations": citations,
            "errors": errors,
            "trace": done(),
        }
    out = _answer_out(obj, citations, done())
    if errors:
        return {
            "status": "unverified",
            **out,
            "errors": errors,
            "failed_claims": _failed_claims(errors),
            "verdict_score": att.get("verdict_score"),
        }
    return {"status": "answered", **out}


def _answer_out(obj: dict, citations: dict, trace: dict) -> dict:
    """Contract object → the answer payload both lanes return."""
    return {
        "answer": (
            obj["verdict"]
            + "\n\n"
            + " ".join(
                f"{c['text']} [{', '.join(str(x) for x in c['ids'])}]"
                for c in obj["claims"]
            )
        ),
        "verdict": obj["verdict"],
        "claims": obj["claims"],
        "citations": citations,
        "trace": trace,
    }


def _extractive_answer(
    question: str,
    pack: list[dict],
    retrieved_ids: set,
    chunks_by_id: dict,
    citations: dict,
    index: dict,
    *,
    ask_fn,
    chat_fn,
    key,
    trace: dict,
    finish,
) -> dict:
    """The extractive lane (EXTRACTIVE=1): claims are verbatim sentences —
    the writer then generates ONLY the bottom line over those quotes
    (a small pre-scored pack), kept if it passes verdict grounding.
    The same checks run; fails closed on judge outage."""
    t = time.monotonic()
    att: dict = {"errors": []}
    try:
        got, best = extract.claims(
            question, pack, index, ask_fn=ask_fn, key=key, meta=att
        )
        direction = (
            extract.direction_for(question, got, ask_fn=ask_fn, key=key) if got else ""
        )
        verdict = (
            extract.DIR_PHRASES.get(direction, extract.DIR_FALLBACK) if got else None
        )
    except judge.JudgeError as e:
        att.update({"output": "extract stage failed", "ms": _ms(t)})
        trace["attempts"].append(att)
        finish()
        trace["judge_error"] = str(e)
        return {"status": "unavailable", "answer": UNAVAILABLE, "trace": trace}
    att["best_chunk"] = best
    obj = extract.compose(got, verdict)
    errors: list[str] = []
    if obj["status"] == "answered":
        errors = verify.deterministic(obj, retrieved_ids, chunks_by_id)
        if not errors:
            try:
                errors, att["grounded"] = verify.grounded(
                    obj, chunks_by_id, ask_fn=ask_fn, key=key
                )
            except judge.JudgeError as e:
                att.update(
                    {"output": json.dumps(obj)[:8000], "errors": errors, "ms": _ms(t)}
                )
                trace["attempts"].append(att)
                finish()
                trace["judge_error"] = str(e)
                return {
                    "status": "unavailable",
                    "answer": UNAVAILABLE,
                    "trace": trace,
                }
    # synthesis pass — generated bottom line + findings bullets over the
    # kept quotes; the same grounding check gates both, the typed verdict
    # is the fallback
    points: list[dict] = []
    if obj["status"] == "answered" and not errors:
        t2 = time.monotonic()
        att["verdict_kind"] = "typed"
        for synth_try in range(2):
            try:
                canary = generate.new_canary()
                raw2 = generate.summarize(
                    question, obj["claims"], canary, chat_fn=chat_fn
                )
                if parsed := _synth_obj(raw2, canary):
                    v = parsed["verdict"]
                    att["synth_score"] = vs2 = _synth_grounded(
                        v, obj["claims"], ask_fn=ask_fn, key=key
                    )
                    if vs2 >= verify.GROUNDED_MIN:
                        att["verdict_kind"] = "generated"
                        att["points_emitted"] = len(parsed["points"])
                        points = _check_points(
                            parsed["points"], obj["claims"], ask_fn=ask_fn, key=key
                        )
                        att["points_kept"] = len(points)
                        obj = extract.compose(obj["claims"], v)
                    else:
                        att["synth_rejected"] = {"text": v[:300], "score": vs2}
                else:
                    # temp 0 → a retry returns the same text — a parse
                    # failure is deterministic, not transient; don't retry
                    att["synth_error"] = (
                        "unparseable verdict: " + raw2[:140].replace("\n", " ")
                        if raw2.strip()
                        else "empty synthesis response"
                    )
                break
            except (generate.GenerateError, judge.JudgeError) as e:
                att["synth_error"] = str(e)[:160]  # transient — retry once
        att["synth_ms"] = _ms(t2)
        if synth_try:
            att["synth_tries"] = synth_try + 1
    att.update({"output": json.dumps(obj)[:8000], "errors": errors, "ms": _ms(t)})
    trace["attempts"].append(att)
    if obj["status"] == "unclear":
        return {
            "status": "unclear",
            "answer": ABSTAIN,
            "missing": obj["missing"],
            "citations": citations,
            "errors": errors,
            "trace": finish(),
        }
    out = _answer_out(obj, citations, finish())
    out["points"] = points
    out["direction"] = direction
    out["verdict_kind"] = att.get("verdict_kind", "typed")
    if errors:
        return {
            "status": "unverified",
            **out,
            "errors": errors,
            "failed_claims": _failed_claims(errors),
        }
    return {"status": "answered", **out}


def _failed_claims(errors: list[str]) -> list[int]:
    """Claim indices the checks named — 'claim 2 cites non-retrieved ids'
    → [2]. The marks a visitor sees as failed stamps."""
    out: set[int] = set()
    for e in errors:
        out.update(int(m.group(1)) for m in re.finditer(r"claim (\d+)", e))
    return sorted(out)


def _scan_verdict(buf: str) -> str | None:
    """The verdict string once its closing quote has streamed — it sits
    before the claims array, so the answer can surface mid-write."""
    m = re.search(r'"verdict"\s*:\s*"((?:[^"\\]|\\.)*)"', buf)
    if not m:
        return None
    try:
        return json.loads(f'"{m.group(1)}"')
    except json.JSONDecodeError:
        return None


def _scan_claims(buf: str) -> list[dict]:
    """Each COMPLETE {…} object inside the claims array so far — brace-matched
    over a partial stream. Best-effort live parse; contract.parse at stream
    end stays authoritative."""
    i = buf.find('"claims"')
    if i < 0:
        return []
    i = buf.find("[", i)
    if i < 0:
        return []
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for j in range(i + 1, len(buf)):
        c = buf[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = j
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    out.append(json.loads(buf[start : j + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1
        elif c == "]" and depth == 0:
            break
    return out


def answer_events(
    question: str,
    dense: list,
    chunks_by_id: dict,
    index: dict,
    *,
    ask_fn=judge.ask,
    write_fn=generate.write_stream,
    key: str | None = None,
):
    """The same pipeline as an event stream — the exhibit path.

    Optimistic: writer tokens stream live, the verdict surfaces as soon as
    its string closes, and each claim is stamped by the judge as soon as
    its object closes (a stamp can land while later claims still
    generate). The same checks run; the final status event is only
    decided after the stream completes. Yielded events: screen, retrieve,
    scope?, token, verdict?, claim, claim_stamp, verdict_score?, rescue?,
    attempt, status (terminal).
    """
    t0 = time.monotonic()
    t = t0
    # the token ledger — bound once, rides the injected-fn plumbing to
    # every gate/scope/verify/extract/synth/writer call
    usage_log: list[dict] = []
    ask_fn = partial(ask_fn, usage=usage_log)
    write_fn = partial(
        write_fn,
        stream_fn=partial(generate.chat_stream, usage=usage_log, stage="write"),
    )
    screen = gate.screen(question, ask_fn=ask_fn, key=key)
    yield {"event": "screen", **screen, "ms": _ms(t)}
    if screen["injected"]:
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "refused",
            "answer": None,
            "ms": _ms(t0),
        }
        return
    if screen["intent"] == "advice":
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "declined",
            "answer": ADVICE_DECLINE,
            "ms": _ms(t0),
        }
        return
    if screen["intent"] == "uncertain":
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "unavailable",
            "answer": UNAVAILABLE,
            "ms": _ms(t0),
        }
        return

    t = time.monotonic()
    pack = retrieve(index, chunks_by_id, question, dense, top_k=10)
    yield {"event": "retrieve", "pack": [c["pmid"] for c in pack], "ms": _ms(t)}
    # built once so every outcome that shows the check trail — answered,
    # unclear, abstained — can resolve study titles for its claims
    citations = {
        str(c["pmid"]): {
            "title": c["title"],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}/",
            "journal": c.get("journal", ""),
            "year": str(c.get("pub_date", ""))[:4],
            "superseded_by": c.get("superseded_by"),
        }
        for c in pack
    }
    if screen["intent"] == "unrelated":
        try:
            t = time.monotonic()
            scope = _scope(question, pack, ask_fn=ask_fn, key=key)
        except judge.JudgeError:
            yield {
                "event": "status",
                "usage": _usage_totals(usage_log),
                "status": "unavailable",
                "answer": UNAVAILABLE,
                "ms": _ms(t0),
            }
            return
        yield {"event": "scope", "score": scope, "ms": _ms(t)}
        if scope < SCOPE_MIN:
            yield {
                "event": "status",
                "usage": _usage_totals(usage_log),
                "status": "off_topic",
                "answer": OFF_TOPIC,
                "ms": _ms(t0),
            }
            return

    retrieved_ids = {str(c["pmid"]) for c in pack}
    if os.environ.get("EXTRACTIVE", "1") != "0":
        yield from _extractive_events(
            question,
            pack,
            retrieved_ids,
            chunks_by_id,
            citations,
            index,
            ask_fn=ask_fn,
            key=key,
            t0=t0,
            usage_log=usage_log,
        )
        return

    canary = generate.new_canary()

    # one pass — failed checks land as marks on the answer, not a hidden
    # rewrite; only unparseable output still abstains
    t = time.monotonic()
    t_w = t
    buf = ""
    stamps: dict[str, float] = {}
    seen = 0
    verdict_done = False
    vs: float | None = None
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        for delta in write_fn(question, pack, canary):
            # (kind, text) pairs from write_stream; a bare string is a
            # token (tests stub write_fn with plain strings)
            kind, text = delta if isinstance(delta, tuple) else ("token", delta)
            if kind == "think":
                # the model's reasoning, surfaced while it composes —
                # canary stripped: the tripwire must never reach a browser
                yield {"event": "think", "text": text.replace(canary, "…")}
                continue
            buf += text
            yield {"event": "token", "attempt": 0, "text": text}
            if not verdict_done:
                v = _scan_verdict(buf)
                if v is not None:
                    verdict_done = True
                    # the bottom line closes before the claims array —
                    # surfacing it here is what lets the answer land
                    # while the checks still run
                    yield {"event": "verdict", "attempt": 0, "text": v}
            if "}" in text or "]" in text:  # only these can close a claim
                claims = _scan_claims(buf)
                for k in range(seen, len(claims)):
                    c = claims[k]
                    yield {
                        "event": "claim",
                        "attempt": 0,
                        "i": k,
                        "claim": c,
                    }
                    for x in c.get("ids", []):
                        if str(x) in chunks_by_id:
                            futs[
                                ex.submit(
                                    verify.claim_score,
                                    c["text"],
                                    chunks_by_id[str(x)],
                                    ask_fn=ask_fn,
                                    key=key,
                                )
                            ] = f"{k}:{x}"
                    seen = k + 1
            for f in [f for f in futs if f.done()]:
                tag = futs.pop(f)
                try:
                    s = f.result()
                except judge.JudgeError as e:
                    yield {
                        "event": "status",
                        "usage": _usage_totals(usage_log),
                        "status": "unavailable",
                        "answer": UNAVAILABLE,
                        "judge_error": str(e),
                        "ms": _ms(t0),
                    }
                    return
                stamps[tag] = s
                i_s, x_s = tag.split(":", 1)
                yield {
                    "event": "claim_stamp",
                    "attempt": 0,
                    "i": int(i_s),
                    "pmid": x_s,
                    "score": s,
                }
        write_ms = _ms(t_w)
        raw = buf
        for f, tag in futs.items():  # stream ended — wait for in-flight stamps
            try:
                s = f.result()
            except judge.JudgeError as e:
                yield {
                    "event": "status",
                    "usage": _usage_totals(usage_log),
                    "status": "unavailable",
                    "answer": UNAVAILABLE,
                    "judge_error": str(e),
                    "ms": _ms(t0),
                }
                return
            stamps[tag] = s
            i_s, x_s = tag.split(":", 1)
            yield {
                "event": "claim_stamp",
                "attempt": 0,
                "i": int(i_s),
                "pmid": x_s,
                "score": s,
            }

    obj, why = contract.parse(raw)
    if canary in raw:
        why = "canary echo"
        obj = None
    if obj is None:
        errors = [why or "contract failure"]
    else:
        errors = verify.deterministic(obj, retrieved_ids, chunks_by_id)
        if not errors and obj["status"] == "answered" and obj.get("verdict"):
            errors = _verdict_numbers(obj, retrieved_ids, chunks_by_id)
        if not errors:
            try:
                for i, c in enumerate(obj["claims"]):  # pairs the stream missed
                    for x in c["ids"]:
                        tag = f"{i}:{x}"
                        if tag not in stamps and str(x) in chunks_by_id:
                            s = verify.claim_score(
                                c["text"],
                                chunks_by_id[str(x)],
                                ask_fn=ask_fn,
                                key=key,
                            )
                            stamps[tag] = s
                            yield {
                                "event": "claim_stamp",
                                "attempt": 0,
                                "i": i,
                                "pmid": str(x),
                                "score": s,
                            }
                errors = verify.grounded_errors(stamps)
                if not errors and obj["status"] == "answered":
                    vs = _verdict_grounded(obj, chunks_by_id, ask_fn=ask_fn, key=key)
                    yield {
                        "event": "verdict_score",
                        "attempt": 0,
                        "score": vs,
                    }
                    if vs < verify.GROUNDED_MIN:
                        errors = [f"verdict unsupported (score {vs:.2f})"]
            except judge.JudgeError as e:
                yield {
                    "event": "status",
                    "usage": _usage_totals(usage_log),
                    "status": "unavailable",
                    "answer": UNAVAILABLE,
                    "judge_error": str(e),
                    "ms": _ms(t0),
                }
                return
        if not errors and obj["status"] == "unclear":
            try:
                s = _rescue(question, pack, ask_fn=ask_fn, key=key)
                yield {"event": "rescue", "attempt": 0, "score": s}
                if s >= verify.GROUNDED_MIN:
                    errors = [
                        (
                            "evidence exists in the pack (support "
                            f"{s:.2f}) — the evidence was there; "
                            "'unclear' was the wrong call"
                        )
                    ]
            except judge.JudgeError as e:
                yield {
                    "event": "status",
                    "usage": _usage_totals(usage_log),
                    "status": "unavailable",
                    "answer": UNAVAILABLE,
                    "judge_error": str(e),
                    "ms": _ms(t0),
                }
                return
    yield {
        "event": "attempt",
        "attempt": 0,
        "errors": errors,
        "write_ms": write_ms,
        "ms": _ms(t),
    }

    if obj is None:
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "abstained",
            "answer": ABSTAIN,
            "errors": errors,
            "citations": citations,
            "ms": _ms(t0),
        }
        return
    if obj["status"] == "refused":
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "refused",
            "answer": None,
            "ms": _ms(t0),
        }
        return
    if obj["status"] == "unclear":
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "unclear",
            "answer": ABSTAIN,
            "missing": obj["missing"],
            "citations": citations,
            "errors": errors,
            "ms": _ms(t0),
        }
        return
    answer_text = (
        obj["verdict"]
        + "\n\n"
        + " ".join(
            f"{c['text']} [{', '.join(str(x) for x in c['ids'])}]"
            for c in obj["claims"]
        )
    )
    if errors:
        # the answer the writer gave, shown with its misses marked —
        # 'unverified', not hidden behind a rewrite
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "unverified",
            "answer": answer_text,
            "verdict": obj["verdict"],
            "claims": obj["claims"],
            "citations": citations,
            "errors": errors,
            "failed_claims": _failed_claims(errors),
            "verdict_score": vs,
            "ms": _ms(t0),
        }
        return
    yield {
        "event": "status",
        "usage": _usage_totals(usage_log),
        "status": "answered",
        "answer": answer_text,
        "verdict": obj["verdict"],
        "claims": obj["claims"],
        "citations": citations,
        "ms": _ms(t0),
    }


def _extractive_events(
    question: str,
    pack: list[dict],
    retrieved_ids: set,
    chunks_by_id: dict,
    citations: dict,
    index: dict,
    *,
    ask_fn,
    key,
    t0: float,
    usage_log: list[dict],
):
    """The extractive lane as events — same contract as the writer path
    minus token/think (no writer runs): extract ×2 → verdict → claim →
    claim_stamp → attempt → status."""
    unavailable = {
        "event": "status",
        "usage": _usage_totals(usage_log),
        "status": "unavailable",
        "answer": UNAVAILABLE,
    }
    t = time.monotonic()
    # chunk scores drain as they land — the progress events ARE the
    # stream for this lane (there are no writer tokens to show)
    sent_n = 0
    try:
        gen = extract.claims_iter(question, pack, index, ask_fn=ask_fn, key=key)
        while True:
            try:
                phase, done, n = next(gen)
            except StopIteration as e:
                got, best = e.value
                break
            if phase == "sentences":
                sent_n = n
            yield {"event": "extract", "phase": phase, "done": done, "n": n}
    except judge.JudgeError as e:
        yield {**unavailable, "judge_error": str(e), "ms": _ms(t0)}
        return
    yield {
        "event": "extract",
        "phase": "sentences",
        "done": sent_n,
        "n": sent_n,
        "kept": len(got),
        "best_chunk": best,
        "ms": _ms(t),
    }
    if not got:
        yield {
            "event": "attempt",
            "attempt": 0,
            "errors": [],
            "write_ms": _ms(t),
            "ms": _ms(t),
        }
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "unclear",
            "answer": ABSTAIN,
            "missing": extract.compose([])["missing"],
            "citations": citations,
            "ms": _ms(t0),
        }
        return
    # the typed direction + the claim checks judge in parallel. the typed
    # verdict stays internal — it feeds compose and the status fallback,
    # but the box never renders a line the synthesis will replace
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            vfut = ex.submit(
                extract.direction_for, question, got, ask_fn=ask_fn, key=key
            )
            sfuts = {
                ex.submit(
                    verify.claim_score,
                    c["text"],
                    chunks_by_id[str(x)],
                    ask_fn=ask_fn,
                    key=key,
                ): f"{i}:{x}"
                for i, c in enumerate(got)
                for x in c["ids"]
                if str(x) in chunks_by_id
            }
            direction = vfut.result()
        verdict = extract.DIR_PHRASES.get(direction, extract.DIR_FALLBACK)
        obj = extract.compose(got, verdict)
    except judge.JudgeError as e:
        yield {**unavailable, "judge_error": str(e), "ms": _ms(t0)}
        return
    errors = verify.deterministic(obj, retrieved_ids, chunks_by_id)
    stamps: dict[str, float] = {}
    for i, c in enumerate(obj["claims"]):
        yield {"event": "claim", "attempt": 0, "i": i, "claim": c}
    if not errors:
        # as_completed → stamps land in true finish order; draining the
        # dict waits on submission order and batches every emission
        # behind the slowest call, which is what made 0/8 → 8/8 look instant
        for f in as_completed(sfuts):
            tag = sfuts[f]
            try:
                s = f.result()
            except judge.JudgeError as e:
                yield {**unavailable, "judge_error": str(e), "ms": _ms(t0)}
                return
            stamps[tag] = s
            i_s, x_s = tag.split(":", 1)
            yield {
                "event": "claim_stamp",
                "attempt": 0,
                "i": int(i_s),
                "pmid": x_s,
                "score": s,
            }
        errors = verify.grounded_errors(stamps)
    yield {
        "event": "attempt",
        "attempt": 0,
        "errors": errors,
        "write_ms": _ms(t),
        "ms": _ms(t),
    }

    # synthesis pass — the writer generates ONLY the bottom line over the
    # kept quotes as its pack (a small, already-scored input). The quote
    # rows stay the claims; the generated verdict is the only verdict the
    # box sees unless synthesis fails — then the status carries the typed
    # direction instead (shown once, never replaced).
    synth: str | None = None
    points: list[dict] = []
    synth_meta: dict = {}
    if not errors:
        t2 = time.monotonic()
        for synth_try in range(2):
            try:
                canary = generate.new_canary()
                buf2 = ""
                for delta in generate.summarize_stream(
                    question,
                    obj["claims"],
                    canary,
                    stream_fn=partial(
                        generate.chat_stream, usage=usage_log, stage="write"
                    ),
                ):
                    kind, text = delta if isinstance(delta, tuple) else ("token", delta)
                    if kind == "think":
                        yield {"event": "think", "text": text.replace(canary, "…")}
                        continue
                    buf2 += text
                    yield {"event": "token", "attempt": 0, "text": text}
                if parsed := _synth_obj(buf2, canary):
                    v = parsed["verdict"]
                    synth_meta["had_points_key"] = '"points"' in buf2
                    vs2 = _synth_grounded(v, obj["claims"], ask_fn=ask_fn, key=key)
                    synth_meta["synth_score"] = vs2
                    yield {
                        "event": "verdict_score",
                        "attempt": 0,
                        "score": vs2,
                        "passed": vs2 >= verify.GROUNDED_MIN,
                    }
                    if vs2 >= verify.GROUNDED_MIN:
                        # points before the verdict event — a judge error
                        # here retries the whole synth, and the verdict
                        # must not have been emitted yet or it'd double
                        synth_meta["points_emitted"] = len(parsed["points"])
                        points = _check_points(
                            parsed["points"], obj["claims"], ask_fn=ask_fn, key=key
                        )
                        synth_meta["points_kept"] = len(points)
                        synth = v
                        yield {"event": "verdict", "attempt": 0, "text": synth}
                    else:
                        synth_meta["synth_rejected"] = {"text": v[:300], "score": vs2}
                else:
                    # temp 0 → retry returns the same text — deterministic
                    # failure, not transient; don't retry
                    synth_meta["synth_error"] = (
                        "unparseable verdict: " + buf2[:140].replace("\n", " ")
                        if buf2.strip()
                        else "empty synthesis response"
                    )
                break
            except (generate.GenerateError, judge.JudgeError) as e:
                # transient — one retry; the typed verdict is the last
                # resort, and the failure still lands in the trace
                synth_meta["synth_error"] = str(e)[:160]
        if synth_try:
            synth_meta["synth_tries"] = synth_try + 1
        synth_meta["synth_ms"] = _ms(t2)

    if synth:
        obj = extract.compose(obj["claims"], synth)
    answer_text = (
        obj["verdict"]
        + "\n\n"
        + " ".join(
            f"{c['text']} [{', '.join(str(x) for x in c['ids'])}]"
            for c in obj["claims"]
        )
    )
    base = {
        "answer": answer_text,
        "verdict": obj["verdict"],
        "verdict_kind": "generated" if synth else "typed",
        "direction": direction,
        "points": points,
        "claims": obj["claims"],
        "citations": citations,
        "ms": _ms(t0),
        **synth_meta,
    }
    if errors:
        yield {
            "event": "status",
            "usage": _usage_totals(usage_log),
            "status": "unverified",
            **base,
            "errors": errors,
            "failed_claims": _failed_claims(errors),
        }
        return
    yield {
        "event": "status",
        "usage": _usage_totals(usage_log),
        "status": "answered",
        **base,
    }


def main() -> None:
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--dense", default="data/dense.json")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    chunks = {
        json.loads(l)["chunk_id"]: json.loads(l)
        for l in (root / "data" / "chunks.jsonl").open()
    }
    index = json.loads((root / "data" / "bm25.json").read_text())
    dense = json.loads((root / a.dense).read_text())
    out = answer(
        a.question,
        dense,
        chunks,
        index,
        key=os.environ["VLLM_SR_API_KEY"],
    )
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
