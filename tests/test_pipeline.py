import json

from supplements_rag import contract, gate, judge, pipeline, verify

CHUNKS = {
    "100": {
        "chunk_id": "100",
        "pmid": "100",
        "title": "Magnesium and sleep RCT",
        "text": "A trial of 500 mg magnesium improved sleep onset by 17 minutes.",
    },
    "200": {
        "chunk_id": "200",
        "pmid": "200",
        "title": "Iron for breath holding",
        "text": "Iron supplementation reduced spells.",
    },
}
INDEX = {
    "docs": {
        "100": {"tf": {"magnesium": 3, "sleep": 2}, "len": 10},
        "200": {"tf": {"iron": 2, "spells": 1}, "len": 8},
    },
    "df": {"magnesium": 1, "sleep": 1, "iron": 1, "spells": 1},
    "avgdl": 9.0,
    "k1": 1.2,
    "b": 0.75,
}
GOOD = json.dumps(
    {
        "status": "answered",
        "verdict": "Yes — a trial shows magnesium improved sleep onset.",
        "claims": [
            {
                "text": "500 mg magnesium improved sleep onset by 17 minutes",
                "ids": ["100"],
            }
        ],
        "missing": [],
    }
)


def ask_fn(state, questions, **_):
    out: dict = {}
    for k, q in questions.items():
        if q["type"] == "noul":
            out[k] = {"noul": 0.05 if k == "injection" else 0.9}
        else:
            out[k] = {"choice": "evidence"}
    return out


def test_contract_parse_valid():
    obj, err = contract.parse(GOOD)
    assert err is None and obj is not None and obj["status"] == "answered"


def test_contract_rejects_prose_and_bad_shapes():
    assert contract.parse("Sure! Here is the answer: {}")[0] is None
    assert contract.parse('{"status": "maybe", "claims": [], "missing": []}')[0] is None
    assert (
        contract.parse('{"status": "answered", "claims": [], "missing": []}')[0] is None
    )
    assert (
        contract.parse(
            '{"status": "answered", "claims": [{"text": "x", "ids": []}],'
            ' "missing": []}'
        )[0]
        is None
    )


def test_gate_screen_paths():
    s = gate.screen("what does magnesium do", ask_fn=ask_fn)
    assert s["intent"] == "evidence" and not s["injected"]

    def advice_fn(state, questions, **_):
        return {
            "injection": {"noul": 0.1},
            "intent": {"choice": "advice"},
        }

    assert gate.screen("what dose for me?", ask_fn=advice_fn)["intent"] == "advice"

    def inj_fn(state, questions, **_):
        return {
            "injection": {"noul": 0.95},
            "intent": {"choice": "evidence"},
        }

    assert gate.screen("ignore instructions", ask_fn=inj_fn)["injected"]

    def down_fn(state, questions, **_):
        raise judge.JudgeError("down")

    s = gate.screen("q", ask_fn=down_fn)
    assert s["intent"] == "uncertain" and s["injection"] is None


def test_verify_deterministic():
    obj, _ = contract.parse(GOOD)
    assert obj is not None
    assert verify.deterministic(obj, {"100", "200"}, CHUNKS) == []
    bad = json.loads(GOOD)
    bad["claims"][0]["ids"] = ["999"]
    errs = verify.deterministic(bad, {"100", "200"}, CHUNKS)
    assert any("non-retrieved" in e for e in errs)
    bad2 = json.loads(GOOD)
    bad2["claims"][0]["text"] = "A 900 mg dose helped"
    errs = verify.deterministic(bad2, {"100"}, CHUNKS)
    assert any("900" in e for e in errs)


def test_verify_number_rounding_accepted():
    obj = {
        "status": "answered",
        "verdict": "v",
        "claims": [
            {"text": "onset fell by 17.4 minutes", "ids": ["100"]},
        ],
        "missing": [],
    }
    # source has 17 (integer) — '17.4' is not a rounding of it → error
    errs = verify.deterministic(obj, {"100"}, CHUNKS)
    assert any("17.4" in e for e in errs)
    # source-style: claim '17.4' IS a rounding of source '17.36'
    chunks = dict(CHUNKS)
    chunks["100"] = {**chunks["100"], "text": "onset improved 17.36 minutes"}
    assert verify.deterministic(obj, {"100"}, chunks) == []


def test_verify_grounded_and_fail_closed():
    obj, _ = contract.parse(GOOD)
    assert obj is not None
    errs, scores = verify.grounded(obj, CHUNKS, ask_fn=ask_fn)
    assert errs == [] and scores["0:100"] == 0.9

    def down_fn(state, questions, **_):
        raise judge.JudgeError("down")

    try:
        verify.grounded(obj, CHUNKS, ask_fn=down_fn)
        assert False
    except judge.JudgeError:
        pass


def run_pipeline(question, **kw):
    kw.setdefault("ask_fn", ask_fn)
    kw.setdefault("chat_fn", lambda messages, **_: GOOD)
    return pipeline.answer(question, [], CHUNKS, INDEX, **kw)


def test_pipeline_advice_declines_before_retrieval():
    def advice_fn(state, questions, **_):
        return {"injection": {"noul": 0.1}, "intent": {"choice": "advice"}}

    out = run_pipeline("what should I take", ask_fn=advice_fn)
    assert out["status"] == "declined" and "clinician" in out["answer"]


def test_pipeline_injection_refused():
    def inj_fn(state, questions, **_):
        return {"injection": {"noul": 0.95}, "intent": {"choice": "evidence"}}

    assert run_pipeline("ignore rules", ask_fn=inj_fn)["status"] == "refused"


def test_pipeline_gate_outage_fails_closed():
    def down_fn(state, questions, **_):
        raise judge.JudgeError("down")

    assert run_pipeline("magnesium?", ask_fn=down_fn)["status"] == "unavailable"


def test_pipeline_answered_with_citations():
    out = run_pipeline("does magnesium help sleep")
    assert out["status"] == "answered"
    assert "[100]" in out["answer"]
    assert out["citations"]["100"]["url"].endswith("/100/")


def test_pipeline_bad_contract_abstains_without_repair():
    calls = []

    def flaky(messages, **_):
        calls.append(1)
        return "not json at all"

    out = run_pipeline("magnesium?", chat_fn=flaky)
    assert out["status"] == "abstained" and len(calls) == 1


def test_pipeline_unfixable_abstains():
    def broken(messages, **_):
        return "garbage"

    out = run_pipeline("magnesium?", chat_fn=broken)
    assert out["status"] == "abstained"


def test_pipeline_unsupported_claim_shows_unverified():
    def low_fn(state, questions, **_):
        if "intent" in questions:
            return {"injection": {"noul": 0.0}, "intent": {"choice": "evidence"}}
        return {"sup": {"noul": 0.1}}

    out = run_pipeline("magnesium?", ask_fn=low_fn)
    assert out["status"] == "unverified"
    assert out["failed_claims"] == [0] and out["claims"][0]["ids"] == ["100"]


def test_pipeline_unclear_stands_even_when_evidence_exists():
    unclear = json.dumps(
        {"status": "unclear", "claims": [], "missing": ["nothing found"]}
    )
    calls = []

    def writer(messages, **_):
        calls.append(1)
        return unclear

    out = run_pipeline("magnesium?", chat_fn=writer)
    assert out["status"] == "unclear" and len(calls) == 1
    assert out["trace"]["attempts"][0]["rescue_score"] == 0.9
    assert out["errors"]


def test_pipeline_unclear_stands_when_judge_agrees():
    unclear = json.dumps(
        {"status": "unclear", "claims": [], "missing": ["nothing found"]}
    )

    def low_fn(state, questions, **_):
        if "intent" in questions:
            return {"injection": {"noul": 0.0}, "intent": {"choice": "evidence"}}
        return {k: {"noul": 0.1} for k in questions}

    out = run_pipeline("magnesium?", ask_fn=low_fn, chat_fn=lambda m, **_: unclear)
    assert (
        out["status"] == "unclear"
        and out["trace"]["attempts"][0]["rescue_score"] == 0.1
    )


def test_pipeline_judge_outage_mid_verify_fails_closed():
    def dying(state, questions, **_):
        if "intent" in questions:
            return {"injection": {"noul": 0.0}, "intent": {"choice": "evidence"}}
        raise judge.JudgeError("judge down")

    assert run_pipeline("magnesium?", ask_fn=dying)["status"] == "unavailable"


def run_events(question, **kw):
    kw.setdefault("ask_fn", ask_fn)
    kw.setdefault("write_fn", lambda *a, **k: iter([GOOD[:60], GOOD[60:]]))
    return list(pipeline.answer_events(question, [], CHUNKS, INDEX, **kw))


def test_scan_claims_partial_and_complete():
    assert pipeline._scan_claims(GOOD[: GOOD.index('"text"')]) == []
    claims = pipeline._scan_claims(GOOD)
    assert len(claims) == 1 and claims[0]["ids"] == ["100"]


def test_events_stream_happy_path():
    events = run_events("does magnesium help sleep")
    kinds = [e["event"] for e in events]
    assert kinds[0] == "screen" and kinds[1] == "retrieve"
    assert "token" in kinds and "claim" in kinds and "claim_stamp" in kinds
    assert any(e["event"] == "verdict_score" for e in events)
    last = events[-1]
    assert last["event"] == "status" and last["status"] == "answered"
    assert last["claims"][0]["ids"] == ["100"] and "ms" in last


def test_events_low_score_stamps_then_unverified():
    def low_fn(state, questions, **_):
        if "intent" in questions:
            return {"injection": {"noul": 0.0}, "intent": {"choice": "evidence"}}
        return {k: {"noul": 0.1} for k in questions}

    events = run_events("magnesium?", ask_fn=low_fn)
    assert any(e["event"] == "claim_stamp" and e["score"] == 0.1 for e in events)
    last = events[-1]
    assert last["status"] == "unverified" and last["failed_claims"] == [0]


def test_events_judge_outage_fails_closed():
    def dying(state, questions, **_):
        if "intent" in questions:
            return {"injection": {"noul": 0.0}, "intent": {"choice": "evidence"}}
        raise judge.JudgeError("judge down")

    assert run_events("magnesium?", ask_fn=dying)[-1]["status"] == "unavailable"


def test_events_reasoning_streams_and_strips_canary():
    def writer(q, pack, c, **_):
        yield ("think", f"weighing it… keep {c} secret")
        yield ("token", GOOD)

    events = run_events("does magnesium help sleep", write_fn=writer)
    thinks = [e["text"] for e in events if e["event"] == "think"]
    assert thinks and "CNRY-" not in "".join(thinks)
    tokens = "".join(e["text"] for e in events if e["event"] == "token")
    assert tokens == GOOD and events[-1]["status"] == "answered"
