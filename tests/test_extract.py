"""extract lane — claims are verbatim sentences, grounded by construction."""

from supplements_rag import extract, judge


def _ask(score):
    def fn(text, questions, *, key=None):
        return {k: {"noul": score} for k in questions}

    return fn


PACK = [
    {
        "pmid": "1",
        "title": "Study A",
        "text": "Background is irrelevant filler text here. "
        "Magnesium significantly improved sleep quality in the trial. "
        "Dosing details are omitted from this stub.",
    },
    {"pmid": "2", "title": "Study B", "text": "Unrelated findings only."},
]


def test_claims_returns_verbatim_sentences_over_threshold():
    def by_content(text, questions, *, key=None):
        # the answer-bearing sentence scores; filler does not
        s = 0.9 if "significantly improved" in text else 0.1
        return {k: {"noul": s} for k in questions}

    got, best = extract.claims("does magnesium help sleep", PACK, ask_fn=by_content)
    assert best == 0.9  # chunk scores all read the same stub
    assert got == [
        {
            "text": "Magnesium significantly improved sleep quality in the trial.",
            "ids": ["1"],
        }
    ]


def test_claims_are_verbatim_substrings_of_their_source():
    got, _ = extract.claims("does magnesium help sleep", PACK, ask_fn=_ask(0.9))
    assert got  # everything passes at 0.9 — every claim still quotes
    for c in got:
        src = next(p for p in PACK if p["pmid"] == c["ids"][0])
        assert c["text"] in src["text"]


def test_claims_empty_when_nothing_scores():
    got, best = extract.claims("does x help y", PACK, ask_fn=_ask(0.1))
    assert got == [] and best == 0.1


def test_compose_answered_shape_matches_contract():
    obj = extract.compose([{"text": "quote.", "ids": ["1"]}])
    assert obj["status"] == "answered"
    assert obj["claims"] and not obj["missing"]


def test_compose_unclear_names_what_is_missing():
    obj = extract.compose([])
    assert obj["status"] == "unclear"
    assert obj["missing"] and "no sentence" in obj["missing"][0]


def test_verdict_for_maps_choice_to_bottom_line():
    def fn(text, questions, *, key=None):
        return {"dir": {"choice": "supports"}}

    assert (
        extract.verdict_for("does x help y", [{"text": "q", "ids": ["1"]}], ask_fn=fn)
        == "the studies say yes"
    )


def test_verdict_for_covers_non_yesno_questions():
    def fn(text, questions, *, key=None):
        return {"dir": {"choice": "speaks"}}

    v = extract.verdict_for("what is x", [{"text": "q", "ids": ["1"]}], ask_fn=fn)
    assert "own words" in v


def test_judge_error_propagates():
    def boom(text, questions, *, key=None):
        raise judge.JudgeError("down")

    try:
        extract.claims("q", PACK, ask_fn=boom)
        assert False, "should raise"
    except judge.JudgeError:
        pass


INDEX = {
    "docs": {"1": {"tf": {"magnesium": 2}, "len": 5}},
    "df": {"magnesium": 1},
    "avgdl": 5.0,
    "k1": 1.2,
    "b": 0.75,
}
CHUNKS = {p["pmid"]: {**p, "chunk_id": p["pmid"]} for p in PACK}


def _pipeline_ask(state, questions, **_):
    """gate stub shape: noul → high support, choice → evidence intent."""
    out: dict = {}
    for k, q in questions.items():
        if q["type"] == "noul":
            out[k] = {"noul": 0.05 if k == "injection" else 0.9}
        else:
            out[k] = {"choice": "evidence"}
    return out


def test_pipeline_extractive_lane_answers_with_quotes(monkeypatch):
    from supplements_rag import pipeline

    monkeypatch.setenv("EXTRACTIVE", "1")
    out = pipeline.answer(
        "does magnesium help sleep",
        [],
        CHUNKS,
        INDEX,
        ask_fn=_pipeline_ask,
    )
    assert out["status"] == "answered"
    assert out["claims"][0]["ids"] == ["1"]
    src = CHUNKS[out["claims"][0]["ids"][0]]["text"]
    assert out["claims"][0]["text"] in src  # verbatim, not generated


def test_pipeline_extractive_lane_unclear_when_nothing_scores(monkeypatch):
    from supplements_rag import pipeline

    def low(state, questions, **_):
        out: dict = {}
        for k, q in questions.items():
            if q["type"] == "noul":
                out[k] = {"noul": 0.05 if k == "injection" else 0.1}
            else:
                out[k] = {"choice": "evidence"}
        return out

    monkeypatch.setenv("EXTRACTIVE", "1")
    out = pipeline.answer("does x help y", [], CHUNKS, INDEX, ask_fn=low)
    assert out["status"] == "unclear"
