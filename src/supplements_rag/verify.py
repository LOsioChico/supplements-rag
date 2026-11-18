"""Two-layer check before the answer settles.

Deterministic (no model):
  - every claim cites >=1 pmid, all cited pmids were in the retrieved pack
  - every number in a claim appears in its cited sources (doses, p-values)

Semantic (judge): one noul score per (claim, cited chunk) — does the
source support this claim? >= GROUNDED_MIN grounded.

FAIL CLOSED: a judge outage marks claims unverifiable → the pipeline
abstains. Streamed deltas reach the client provisional — stamped as
unverified — and the settled answer waits for these checks; a skipped
check can never produce one.
"""

import re

from . import judge
from ._pool import executor

GROUNDED_MIN = 0.7
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


_WORD_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"[- ](one|two|three|four|five|six|seven|eight|nine)\b|"
    + r"\b("
    + "|".join(_WORD_NUM)
    + r")\b"
)


def _expand_words(text: str) -> str:
    """'two weeks' → '2 two weeks' so digit claims match spelled-out sources."""

    def repl(m):
        if m.group(1):  # compound 'twenty-three'
            return f"{_WORD_NUM[m.group(1)] + _WORD_NUM[m.group(2)]} {m.group(0)}"
        return f"{_WORD_NUM[m.group(3)]} {m.group(0)}"

    return _WORD_RE.sub(repl, text.lower())


def _is_rounding(claim_num: str, src_nums: set[str]) -> bool:
    """'4.4' is a fair rounding of source '4.43'; '900' rounds to nothing."""
    try:
        cn = float(claim_num.replace(",", ""))
        places = len(claim_num.partition(".")[2])
        return any(round(float(s.replace(",", "")), places) == cn for s in src_nums)
    except ValueError:
        return False


def deterministic(
    obj: dict, retrieved_ids: set[str], chunks: dict[str, dict]
) -> list[str]:
    """Reasons the contract is unsound; empty list = clean."""
    errors = []
    for i, c in enumerate(obj["claims"]):
        ids = [str(x) for x in c["ids"]]
        bad = [x for x in ids if x not in retrieved_ids]
        if bad:
            errors.append(f"claim {i} cites non-retrieved ids: {bad}")
            continue
        src_nums = set()
        for x in ids:
            src_nums |= set(
                _NUM.findall(
                    _expand_words(chunks[x]["title"] + " " + chunks[x]["text"])
                )
            )
        for n in set(_NUM.findall(c["text"])):
            base = n.removesuffix(".0")
            if n in src_nums or base in src_nums:
                continue
            if not _is_rounding(n, src_nums):
                errors.append(f"claim {i} number {n} not in cited sources")
    return errors


def claim_score(claim: str, chunk: dict, *, ask_fn=judge.ask, key=None) -> float:
    answers = ask_fn(
        f"Source:\n{chunk['title']}\n{chunk['text'][:12000]}\n\n"
        f"Claim being checked: {claim}",
        {
            "sup": {
                "type": "noul",
                "instructions": "Does the source directly support this "
                "claim as written?",
            }
        },
        key=key,
        stage="verify",
    )
    return answers["sup"]["noul"]


def grounded(
    obj: dict, chunks: dict[str, dict], *, ask_fn=judge.ask, key=None
) -> tuple[list[str], dict]:
    """(errors, {claim_index: score}) — a claim is grounded if ANY of its
    cited sources scores >= GROUNDED_MIN. JudgeError propagates (fail closed)."""
    scores, errors = {}, []
    with executor(max_workers=6) as ex:
        futures = {
            (i, x): ex.submit(
                claim_score, c["text"], chunks[str(x)], ask_fn=ask_fn, key=key
            )
            for i, c in enumerate(obj["claims"])
            for x in c["ids"]
            if str(x) in chunks
        }
        pair_scores: dict[int, list[float]] = {}
        for (i, x), f in futures.items():
            s = f.result()
            pair_scores.setdefault(i, []).append(s)
            scores[f"{i}:{x}"] = s
    for i, ss in pair_scores.items():
        if not ss or max(ss) < GROUNDED_MIN:
            errors.append(f"claim {i} unsupported (max score {max(ss or [0]):.2f})")
    return errors, scores


def grounded_errors(scores: dict[str, float]) -> list[str]:
    """(errors) from a {'i:pmid': score} map — ANY cited source >= GROUNDED_MIN
    grounds a claim. Shared by the buffered check and the streamed stamps."""
    errors = []
    seen: dict[int, list[float]] = {}
    for k, s in scores.items():
        seen.setdefault(int(k.split(":")[0]), []).append(s)
    for i, ss in seen.items():
        if max(ss) < GROUNDED_MIN:
            errors.append(f"claim {i} unsupported (max score {max(ss):.2f})")
    return errors


if __name__ == "__main__":
    # python -m supplements_rag.verify out.json --chunks data/chunks.jsonl [--judge]
    #   deterministic reasons a saved writer output is unsound; --judge adds
    #   semantic groundedness (network call, needs VLLM_SR_API_KEY)
    import argparse
    import json
    from pathlib import Path

    from .contract import parse

    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="saved writer output (contract json)")
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    obj, cerr = parse(a.out.read_text())
    if obj is None:
        print(cerr or "parse failed")
        raise SystemExit(1)
    chunks = {json.loads(l)["chunk_id"]: json.loads(l) for l in a.chunks.open()}
    cited = {str(x) for c in obj["claims"] for x in c["ids"]}
    errors = deterministic(obj, cited, chunks)
    if a.judge:
        jerrors, scores = grounded(obj, chunks)
        errors += jerrors
        print(json.dumps(scores, indent=1))
    print(json.dumps(errors, indent=1))
