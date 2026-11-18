"""The semantic-router judge — one POST, typed questions back.

POST {base}/v1/systemone {model, state, questions} → answers.{key}
  noul questions   → {noul: 0-1}      (support/attitude scores)
  choice questions → {choice: str}    (one of the given options)

Shared by the gate (intent/injection) and verify (per-claim groundedness).
Raises on any transport/shape failure — callers decide open vs closed.
"""

import json
import os
import random
import time
import urllib.error
import urllib.request

BASE = os.environ.get("VLLM_SR_BASE", "https://app.vllm-sr.ai")
MODEL = os.environ.get("S1_MODEL", "vllm-sr/d3-flash")


class JudgeError(Exception):
    pass


def ask(
    state: str,
    questions: dict,
    *,
    key: str | None = None,
    base: str = BASE,
    model: str = MODEL,
    timeout: float = 30,
    usage: list[dict] | None = None,
    stage: str | None = None,
) -> dict:
    """Returns {question_key: answer_dict}. Raises JudgeError on failure.
    `usage` is the run's token ledger — each successful call appends
    {fn, model, in, out} so a run's cost is auditable, not assumed;
    `stage` names which pipeline step made the call (gate, extract,
    verify…) so the ledger can break a run down per stage."""
    payload = {"model": model, "state": state, "questions": questions}
    # the endpoint rate-limits bursts (verify fans out per claim, the
    # extract lane per sentence) — 429s are transient, so retry with
    # backoff; every other failure still raises immediately
    answers = None
    for attempt in range(4):
        req = urllib.request.Request(
            f"{base}/v1/systemone",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {key or os.environ['VLLM_SR_API_KEY']}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            answers = body.get("answers")
            if usage is not None:
                u = body.get("usage") or {}
                usage.append(
                    {
                        "fn": "systemone",
                        "stage": stage,
                        "model": body.get("model") or model,
                        "in": u.get("input_tokens"),
                        "out": u.get("output_tokens"),
                    }
                )
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(1.5 * (attempt + 1) + random.random())
                continue
            raise JudgeError(str(e)) from e
        except Exception as e:
            raise JudgeError(str(e)) from e
    if not isinstance(answers, dict) or not all(k in answers for k in questions):
        raise JudgeError(f"missing answers: {answers!r}")
    return answers


if __name__ == "__main__":
    # python -m supplements_rag.judge "some text"  → endpoint smoke check
    # (needs VLLM_SR_API_KEY)
    import sys

    state = " ".join(sys.argv[1:]) or "The sky is blue."
    print(
        json.dumps(
            ask(
                state,
                {
                    "ok": {
                        "type": "noul",
                        "instructions": "Is this text coherent English?",
                    }
                },
            ),
            indent=1,
        )
    )
