"""The writer's output contract — strict JSON, no prose around it.

    {"status": "answered"|"unclear"|"refused",
     "verdict": str,          # 1-2 sentence bottom line — the actual answer
     "claims": [{"text": str, "ids": [pmid, ...]}, ...],
     "missing": [str, ...]}

`answered` requires >=1 claim, every claim cites >=1 pmid. `unclear` means
the evidence didn't support an answer (missing lists what lacked). `refused`
is for content the model won't touch. Anything outside this shape is a
contract failure → one pass, no rewrite: unparseable output abstains.
"""

import json

STATUSES = {"answered", "unclear", "refused"}


def parse(raw: str) -> tuple[dict | None, str | None]:
    """(contract, None) on success, (None, reason) on failure."""
    raw = raw.strip()
    if not raw.startswith("{"):
        return None, "not json"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid json: {e.msg}"
    return validate(obj)


def validate(obj: object) -> tuple[dict | None, str | None]:
    if not isinstance(obj, dict):
        return None, "not an object"
    status = obj.get("status")
    if status not in STATUSES:
        return None, f"bad status: {status!r}"
    claims = obj.get("claims")
    missing = obj.get("missing")
    verdict = obj.get("verdict")
    if not isinstance(claims, list) or not isinstance(missing, list):
        return None, "claims/missing must be lists"
    if status == "answered" and not isinstance(verdict, str):
        return None, "answered requires a verdict string"
    for i, c in enumerate(claims):
        if not isinstance(c, dict) or not isinstance(c.get("text"), str):
            return None, f"claim {i}: text must be a string"
        ids = c.get("ids")
        if not isinstance(ids, list) or not all(isinstance(x, (str, int)) for x in ids):
            return None, f"claim {i}: ids must be a list of pmids"
    if status == "answered" and not claims:
        return None, "answered with no claims"
    for i, c in enumerate(claims):
        if not c["ids"]:
            return None, f"claim {i} cites nothing"
    return obj, None


if __name__ == "__main__":
    # python -m supplements_rag.contract out.json
    #   → the rejection reason, or the validated contract object
    import sys
    from pathlib import Path

    obj, err = parse(Path(sys.argv[1]).read_text())
    print(err if err else json.dumps(obj, indent=1))
