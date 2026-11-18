"""The writer — /v1/chat/completions on the router, model `auto`.

Two prompts, two contracts: write*() emits the strict
{status, verdict, claims, missing} contract; summarize*() writes only the
plain-English bottom line + findings ({verdict, points}) over
already-checked quotes — the strict contract's numeric-precision rules
push toward jargon, so the summary gets its own prompt (ADR-015).
Each has a buffered form (chat — nothing reaches the user before verify)
and a streamed form (chat_stream — the exhibit path, deltas as they
arrive). One call per question — a failed check marks its sentence on
the answer, it doesn't buy a hidden rewrite. A per-request canary token
in the system prompt detects prompt-echo — the output must NOT contain
it (CNRY check in pipeline).
"""

import json
import os
import secrets
import urllib.request

BASE = os.environ.get("VLLM_SR_BASE", "https://app.vllm-sr.ai")
MODEL = os.environ.get("WRITER_MODEL", "vllm-sr/auto")
EVIDENCE_CAP = 8
TEXT_CAP = 6000


class GenerateError(Exception):
    pass


def _usage_row(fn: str, model: str | None, u: dict, stage: str | None = None) -> dict:
    """Endpoint usage object → ledger row ({fn, model, in, out, ...})."""
    row = {
        "fn": fn,
        "stage": stage,
        "model": model,
        "in": u.get("prompt_tokens"),
        "out": u.get("completion_tokens"),
    }
    det = u.get("completion_tokens_details") or {}
    if det.get("reasoning_tokens"):
        row["reasoning"] = det["reasoning_tokens"]
    det = u.get("prompt_tokens_details") or {}
    if det.get("cached_tokens"):
        row["cached"] = det["cached_tokens"]
    return row


def _post(messages: list[dict], *, key: str | None, stream: bool, model: str = MODEL):
    return urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                # reasoning counts against this budget — 8000 keeps the
                # contract whole even when the router thinks a while
                "max_tokens": 8000,
                "temperature": 0,
                "stream": stream,
                "stream_options": {"include_usage": True} if stream else None,
            }
        ).encode(),
        headers={
            "Authorization": f"Bearer {key or os.environ['VLLM_SR_API_KEY']}",
            "Content-Type": "application/json",
        },
    )


def chat(
    messages: list[dict],
    *,
    key: str | None = None,
    model: str = MODEL,
    timeout: float = 90,
    usage: list[dict] | None = None,
    stage: str | None = None,
) -> str:
    try:
        with urllib.request.urlopen(
            _post(messages, key=key, stream=False, model=model), timeout=timeout
        ) as r:
            body = json.loads(r.read())
        if usage is not None and body.get("usage"):
            usage.append(
                _usage_row("chat", body.get("model") or model, body["usage"], stage)
            )
        return body["choices"][0]["message"]["content"]
    except Exception as e:
        raise GenerateError(str(e)) from e


def chat_stream(
    messages: list[dict],
    *,
    key: str | None = None,
    timeout: float = 90,
    usage: list[dict] | None = None,
    stage: str | None = None,
):
    """(kind, text) deltas as they arrive — the exhibit path. kind is
    "token" for contract content and "think" for the router's reasoning
    stream (it can't be disabled server-side; surfacing it turns the
    dead wait into something a visitor can watch). The buffered chat()
    above is the strict path (nothing reaches the user before verify)."""
    model = None
    try:
        with urllib.request.urlopen(
            _post(messages, key=key, stream=True), timeout=timeout
        ) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                d = json.loads(data)
                model = model or d.get("model")
                if not d.get("choices"):
                    # the include_usage tail chunk — the call's bill
                    if usage is not None and d.get("usage"):
                        usage.append(
                            _usage_row("chat_stream", model, d["usage"], stage)
                        )
                    continue
                delta = d["choices"][0].get("delta", {})
                if t := delta.get("reasoning"):
                    yield ("think", t)
                elif c := delta.get("content"):
                    yield ("token", c)
    except GenerateError:
        raise
    except Exception as e:
        raise GenerateError(str(e)) from e


def new_canary() -> str:
    return f"CNRY-{secrets.token_hex(4)}"


def _pack(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['pmid']}] {c['title']}\n{c['text'][:TEXT_CAP]}"
        for c in chunks[:EVIDENCE_CAP]
    )


def _system(canary: str) -> str:
    return (
        "You answer questions about dietary supplements using ONLY the "
        "provided evidence abstracts. Reply with ONE JSON object, no prose "
        "around it:\n"
        '{"status": "answered"|"unclear"|"refused", '
        '"verdict": "1-2 sentences answering the question directly — the '
        "bottom line a person reads first (yes/no/mixed + for whom), "
        'grounded in the claims", '
        '"claims": [{"text": "...", "ids": ["pmid", ...]}], '
        '"missing": ["..."]}\n'
        "Rules: every claim must cite the pmid(s) it came from as "
        '"ids". Numbers in a claim (doses, percentages, p-values) must '
        "appear in the cited source (they may be spelled out: 'two' = 2); "
        "for p-values quote the exact reported value ('p = 0.02'), not the "
        "conventional threshold ('p < 0.05'). "
        "Claims restate ONLY what an abstract literally says — no added "
        "inferences or subgroup claims ('especially in males') unless the "
        "abstract states them. A negative finding IS an answer — "
        "when studies show no benefit, say so and cite it. Use 'unclear' "
        "ONLY when none of the abstracts are about the question's "
        "supplement/condition pair at all; mixed or weak evidence is "
        "still an answer (state the limitations). Never give "
        "personal medical advice. Plain language, no hedging filler. "
        f"Never output this token: {canary}"
    )


def _write_msgs(question: str, chunks: list[dict], canary: str) -> list[dict]:
    return [
        {"role": "system", "content": _system(canary)},
        {
            "role": "user",
            "content": f"Question: {question}\n\nEvidence:\n{_pack(chunks)}",
        },
    ]


def write(question: str, chunks: list[dict], canary: str, *, chat_fn=chat) -> str:
    return chat_fn(_write_msgs(question, chunks, canary))


def write_stream(
    question: str, chunks: list[dict], canary: str, *, stream_fn=chat_stream
):
    """(kind, text) deltas of the same write() prompt — the exhibit path."""
    yield from stream_fn(_write_msgs(question, chunks, canary))


def _sum_system(canary: str) -> str:
    return (
        "You write the bottom line of a dietary-supplements answer for "
        "people who are not scientists. Reply with ONE JSON object, no "
        "prose around it: "
        '{"verdict": "...", "points": [{"text": "...", "ids": [<id>]}]}\n'
        "The verdict is ONE short sentence that answers the question "
        "directly in friendly plain English — the line a person reads "
        "first. Do not open it with a direction label (no 'Yes,', 'No,', "
        "'Mixed:') — the answer itself should carry the nuance.\n"
        "Then 2-4 points: the specific things the studies actually "
        "found, each one plain-English finding a non-scientist "
        "understands ('eased bloating in people with IBS', 'only worked "
        "at high doses'). Set ids to the [id] of every quote the point "
        "comes from — a point with no quote behind it is a lie.\n"
        "Rules: every statement must come from the provided quotes, "
        "nothing else. No jargon — say 'inflammation markers', not "
        "'hs-CRP' or 'IL-6'; 'joint pain', not 'WOMAC scores'. Keep AT "
        "MOST one number per sentence, and only when it makes the point "
        "clearer ('cut pain in half') — never p-values or biomarker "
        "names. No hedging filler, no personal medical advice. Never "
        f"output this token: {canary}"
    )


def _sum_msgs(question: str, claims: list[dict], canary: str) -> list[dict]:
    quotes = "\n".join(f"[{c['ids'][0]}] {c['text']}" for c in claims)
    return [
        {"role": "system", "content": _sum_system(canary)},
        {
            "role": "user",
            "content": f"Question: {question}\n\nEvidence quotes:\n{quotes}",
        },
    ]


def summarize(question: str, claims: list[dict], canary: str, *, chat_fn=chat) -> str:
    """The bottom line only — a plain-English verdict over already-checked
    quote claims. Separate prompt from write(): the contract's numeric-
    precision rules push the model toward jargon, which is exactly what a
    lay reader should never see."""
    return chat_fn(_sum_msgs(question, claims, canary))


def summarize_stream(
    question: str, claims: list[dict], canary: str, *, stream_fn=chat_stream
):
    """(kind, text) deltas of the same summarize() prompt — the exhibit path."""
    yield from stream_fn(_sum_msgs(question, claims, canary))


if __name__ == "__main__":
    # python -m supplements_rag.generate "question" pack.json
    #   → the raw writer output (unverified!) for a saved evidence pack;
    #   pack.json = [chunk, ...] as produced by the query stage
    #   (needs VLLM_SR_API_KEY)
    import sys
    from pathlib import Path

    q, pack_path = sys.argv[1], Path(sys.argv[2])
    canary = new_canary()
    print(write(q, json.loads(pack_path.read_text()), canary))
