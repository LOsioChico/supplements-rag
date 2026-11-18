"""extract lane — claims are verbatim sentences, grounded by construction."""

from supplements_rag import extract, judge


def _ask(score):
    def fn(text, questions, **_):
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
    def by_content(text, questions, **kwargs):
        # the answer-bearing sentence scores; filler does not — sentence
        # text rides in the batch question's instructions, chunk text in
        # the prompt
        return {
            k: {
                "noul": 0.9
                if "significantly improved" in text
                or "significantly improved" in q.get("instructions", "")
                else 0.1
            }
            for k, q in questions.items()
        }

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


def test_sentence_scores_batch_one_call_per_source():
    # a chunk's candidate sentences ride ONE judge call as typed questions —
    # 2 chunks × their sentences → 2 sentence calls (plus 2 chunk calls),
    # not 2 + len(sentences)
    calls = []

    def rec(text, questions, **_):
        calls.append(questions)
        return {k: {"noul": 0.9} for k in questions}

    extract.claims("does magnesium help sleep", PACK, ask_fn=rec)
    # batched sentence calls ask s0..sN; chunk calls ask a single 'ev'
    sent_calls = [
        c for c in calls if all(k[0] == "s" and k[1:].isdigit() for k in c)
    ]
    cands = extract.candidates([(0.9, c) for c in PACK])
    n_groups = len({p for p, _, _ in cands})
    assert len(cands) > n_groups  # the batch actually merges several sentences
    assert len(sent_calls) == n_groups  # one call per source, not per sentence


def _entity_routed(entity_map, default=0.9):
    """Stub that answers every score 0.9 EXCEPT entity_score, whose
    'same substance' instruction routes to the per-chunk map — the
    mechanism under test, everything else held at a passing score."""

    def fn(text, questions, *, key=None, **_):
        out: dict = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                out[k] = {"choice": "evidence"}
            elif k == "injection":
                out[k] = {"noul": 0.05}
            elif "same substance" in q.get("instructions", ""):
                score = next(
                    (s for needle, s in entity_map.items() if needle in text),
                    default,
                )
                out[k] = {"noul": score}
            else:
                out[k] = {"noul": 0.9}
        return out

    return fn


def test_claims_entity_gate_drops_wrong_entity_chunks():
    # 'magnesium' is df=1 → rare in INDEX, so the gate fires; the
    # magnesium-free chunk fails identity even though its sentences
    # would score 0.9 — a wrong-entity chunk produces no candidates
    meta: dict = {}
    got, _ = extract.claims(
        "does magnesium help sleep",
        PACK,
        INDEX,
        ask_fn=_entity_routed({"Unrelated": 0.1}),
        meta=meta,
    )
    assert got and all(c["ids"] == ["1"] for c in got)
    assert meta["entity_dropped"] == 1


def test_claims_entity_gate_blocks_when_no_chunk_names_the_entity():
    # every chunk fails identity → empty claims despite 0.9 sentence
    # scores — the pack is about a different substance entirely
    got, _ = extract.claims(
        "does magnesium help sleep",
        PACK,
        INDEX,
        ask_fn=_entity_routed({}, default=0.1),
    )
    assert got == []


def test_compose_answered_shape_matches_contract():
    obj = extract.compose([{"text": "quote.", "ids": ["1"]}])
    assert obj["status"] == "answered"
    assert obj["claims"] and not obj["missing"]


def test_compose_unclear_names_what_is_missing():
    obj = extract.compose([])
    assert obj["status"] == "unclear"
    assert obj["missing"] and "no sentence" in obj["missing"][0]


def test_verdict_for_maps_choice_to_bottom_line():
    def fn(text, questions, **_):
        return {"dir": {"choice": "supports"}}

    assert (
        extract.verdict_for("does x help y", [{"text": "q", "ids": ["1"]}], ask_fn=fn)
        == "the studies say yes"
    )


def test_verdict_for_covers_non_yesno_questions():
    def fn(text, questions, **_):
        return {"dir": {"choice": "speaks"}}

    v = extract.verdict_for("what is x", [{"text": "q", "ids": ["1"]}], ask_fn=fn)
    assert "own words" in v


def test_judge_error_propagates():
    def boom(text, questions, **_):
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


def test_pipeline_extractive_lane_unclear_when_entity_is_wrong(monkeypatch):
    # 'magnesium' is df=1 → rare in INDEX: the entity gate fires, and a
    # pack whose sources fail identity goes unclear even at 0.9 sentence
    # scores — the integration point is pipeline passing `index` through
    from supplements_rag import pipeline

    monkeypatch.setenv("EXTRACTIVE", "1")
    out = pipeline.answer(
        "does magnesium help sleep",
        [],
        CHUNKS,
        INDEX,
        ask_fn=_entity_routed({}, default=0.05),
    )
    assert out["status"] == "unclear"


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
