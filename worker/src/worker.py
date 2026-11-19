"""supplements-rag on a Python Worker.

Same pipeline as `python -m supplements_rag.serve` (the src package is
vendored verbatim by deploy.sh), re-wired for the runtime:

- BM25 postings + chunk bodies live in D1 (loaded from data/*.jsonl by
  worker/scripts/load_d1.mjs — never bundled), so per-ask reads touch only
  the question's terms and the fused top ids.
- The dense leg is env.AI + env.VECTORIZE bindings — the exact calls the
  local node sidecar makes.
- judge/writer go to vllm-sr through httpx2's sync transport (JSPI) —
  injected via answer_events' ask_fn/stream_fn params, matching the
  signatures of judge.ask/generate.chat_stream so no shared code changes.

Heavy imports (FastAPI, the app package) stay lazy so the startup
snapshot clears the Worker memory limit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from functools import partial

from pyodide.ffi import create_proxy, run_sync, to_js
from workers import asgi, wait_until

log = logging.getLogger("supplements-rag")

EMBED_MODEL = "@cf/baai/bge-m3"
DENSE_TOP_K = 20
HTTP = None  # per-isolate httpx2 sync client


def _http():
    global HTTP
    if HTTP is None:
        import httpx2
        from httpx2_jsfetch import JavascriptFetchTransport

        HTTP = httpx2.Client(
            transport=JavascriptFetchTransport(),
            headers={"content-type": "application/json"},
        )
    return HTTP


def _env(env, name, default=None):
    v = getattr(env, name, None)
    if v is None:
        return default
    return v.to_py() if hasattr(v, "to_py") else v


# ── vllm-sr transports (twins of judge.ask / generate.chat_stream — same
# signatures so the injected pipeline can't tell them apart) ──────────────


def _w_ask(
    state: str,
    questions: dict,
    *,
    key: str | None = None,
    base: str = "https://app.vllm-sr.ai",
    model: str = "vllm-sr/d3-flash",
    timeout: float = 30,
    usage: list[dict] | None = None,
    stage: str | None = None,
) -> dict:
    """Twin of judge.ask — same payload, same 429-only retry, same usage
    row; only the transport differs (httpx2 sync instead of urllib)."""
    from supplements_rag import judge

    payload = {"model": model, "state": state, "questions": questions}
    answers = None
    for attempt in range(4):
        try:
            r = _http().post(
                f"{base}/v1/systemone",
                content=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {key}"},
                timeout=timeout,
            )
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1) + random.random())
                    continue
                raise judge.JudgeError("HTTP 429")
            if r.status_code >= 400:
                raise judge.JudgeError(f"HTTP {r.status_code}")
            body = r.json()
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
        except judge.JudgeError:
            raise
        except Exception as e:
            log.exception("_w_ask transport failure")
            raise judge.JudgeError(str(e)) from e
    if not isinstance(answers, dict) or not all(k in answers for k in questions):
        raise judge.JudgeError(f"missing answers: {answers!r}")
    return answers


def _w_chat_stream(
    messages: list[dict],
    *,
    key: str | None = None,
    base: str = "https://app.vllm-sr.ai",
    timeout: float = 90,
    usage: list[dict] | None = None,
    stage: str | None = None,
):
    """(kind, text) deltas — twin of generate.chat_stream over httpx2's
    sync stream (same 'token'/'think' contract the exhibit path shows)."""
    from supplements_rag.generate import GenerateError, _usage_row

    model = None
    try:
        with _http().stream(
            "POST",
            f"{base}/v1/chat/completions",
            content=json.dumps(
                {
                    "model": "vllm-sr/auto",
                    "messages": messages,
                    # reasoning counts against this budget — 8000 keeps the
                    # contract whole even when the router thinks a while
                    "max_tokens": 8000,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ).encode(),
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        ) as r:
            for line in r.iter_lines():
                line = line.strip()
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
        log.exception("_w_chat_stream transport failure")
        raise GenerateError(str(e)) from e


# ── D1-backed retrieval ─────────────────────────────────────────────────


class _DocsView(dict):
    """index["docs"] backed by per-term postings: len() must return the
    corpus size (BM25's idf divides by N), while items() yields only docs
    containing a query term — docs absent from every posting score 0, so
    iterating the subset gives identical scores."""

    def __init__(self, candidates: dict, n_docs: int):
        super().__init__(candidates)
        self._n = n_docs

    def __len__(self) -> int:
        return self._n


_META = None


def _meta(env) -> dict:
    """Corpus-wide BM25 stats — one row per key, cached per isolate."""
    global _META
    if _META is None:
        rows = run_sync(env.DB.prepare("SELECT k, v FROM bm25_meta").all())
        rows = rows.to_py() if hasattr(rows, "to_py") else rows
        _META = {
            r["k"]: json.loads(r["v"])
            for r in (
                rows.get("results") if isinstance(rows, dict) else rows.results.to_py()
            )
        }
    return _META


def _index_view(env, question: str) -> dict:
    """Per-term D1 postings → the index dict bm25.score/rare_hits expect,
    minus every doc that can't score. df is partial by the same argument —
    it's only ever looked up for query tokens."""
    from supplements_rag import bm25

    meta = _meta(env)
    terms = list(dict.fromkeys(bm25.tokenize(question)))
    candidates: dict[str, dict] = {}
    df: dict[str, int] = {}
    if terms:
        marks = ",".join("?" for _ in terms)
        # one row per (term, shard) — df repeats per shard, postings merge
        rows = run_sync(
            env.DB.prepare(
                f"SELECT term, df, postings FROM bm25_postings WHERE term IN ({marks})"
            )
            .bind(*terms)
            .all()
        )
        rows = rows.to_py() if hasattr(rows, "to_py") else rows
        results = (
            rows.get("results") if isinstance(rows, dict) else rows.results.to_py()
        )
        for r in results:
            term, df_val, postings = r["term"], r["df"], json.loads(r["postings"])
            df[term] = df_val
            # pairs carry the doc's own length — no separate lens lookup
            for cid, (tf, ln) in postings.items():
                d = candidates.get(cid)
                if d is None:
                    d = candidates[cid] = {"tf": {}, "len": ln}
                d["tf"][term] = tf
    return {
        "docs": _DocsView(candidates, meta["n_docs"]),
        "df": df,
        "avgdl": meta["avgdl"],
        "k1": meta["k1"],
        "b": meta["b"],
    }


def _dense(env, question: str) -> list:
    """The dense leg — same two calls the local node sidecar makes
    (bge-m3 embed → Vectorize topK), minus the subprocess."""
    from supplements_rag.bm25 import dense_query

    vec = run_sync(env.AI.run(EMBED_MODEL, {"text": [dense_query(question)]}))
    vec = vec.to_py() if hasattr(vec, "to_py") else vec
    r = run_sync(env.VECTORIZE.query(to_js(vec["data"][0]), {"topK": DENSE_TOP_K}))
    r = r.to_py() if hasattr(r, "to_py") else r
    return [[m["id"], m["score"]] for m in r["matches"]]


def _fetch_chunks(env, ids: list[str]) -> dict:
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    rows = run_sync(
        env.DB.prepare(f"SELECT chunk_id, body FROM chunks WHERE chunk_id IN ({marks})")
        .bind(*ids)
        .all()
    )
    rows = rows.to_py() if hasattr(rows, "to_py") else rows
    results = rows.get("results") if isinstance(rows, dict) else rows.results.to_py()
    return {r["chunk_id"]: json.loads(r["body"]) for r in results}


# ── trace persistence ───────────────────────────────────────────────────

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d .()\-]{7,14}\d")
_URLQ = re.compile(r"(\bhttps?://[^\s?\"']+)\?[^\s\"']*")


def _mask(s: str) -> str:
    return _URLQ.sub(r"\1?<query>", _PHONE.sub("<phone>", _EMAIL.sub("<email>", s)))


async def _store_run(
    env, rid: str, question: str, events: list[dict], source: str
) -> None:
    """Final write — fills the stub row written at stream start; inserts
    when no stub exists (run died before the first event). Token/think
    events are dropped: replay cosmetics, the status event carries the answer."""
    try:
        body = json.dumps(
            [e for e in events if e.get("event") not in ("token", "think")],
            ensure_ascii=False,
        )
        res = await (
            env.DB.prepare("UPDATE runs SET events=? WHERE rid=?").bind(body, rid).run()
        )
        res = res.to_py() if hasattr(res, "to_py") else res
        if isinstance(res, dict) and res.get("meta", {}).get("changes", 0) == 0:
            await (
                env.DB.prepare(
                    "INSERT INTO runs (rid, ts, q, events, source) VALUES (?, ?, ?, ?, ?)"
                )
                .bind(rid, time.time(), _mask(question), body, source)
                .run()
            )
    except Exception:
        log.exception("run write failed")


async def _insert_run_stub(
    env, rid: str, question: str, first: dict, source: str
) -> None:
    """Marked row written while the request context is certainly alive —
    wait_until scheduled after a client disconnect never runs, so this is
    what survives to record the abandoned ask."""
    try:
        await (
            env.DB.prepare(
                "INSERT INTO runs (rid, ts, q, events, source) VALUES (?, ?, ?, ?, ?)"
            )
            .bind(
                rid,
                time.time(),
                _mask(question),
                json.dumps([first], ensure_ascii=False),
                source,
            )
            .run()
        )
    except Exception:
        log.exception("run stub write failed")


def schedule(coro) -> None:
    wait_until(create_proxy(asyncio.ensure_future(coro)))


async def _rate_limited(env, request) -> bool:
    rl = getattr(env, "RATE_LIMITER", None)
    if rl is None:
        return False
    try:
        res = await rl.limit({"key": request.headers.get("cf-connecting-ip", "")})
        res = res.to_py() if hasattr(res, "to_py") else res
        ok = (
            res.get("success")
            if isinstance(res, dict)
            else getattr(res, "success", True)
        )
        return not bool(ok)
    except Exception:
        log.exception("rate-limit check failed — serving unthrottled")
        return False


# ── app ──────────────────────────────────────────────────────────────────

_APP = None


def _sse(ev: dict) -> str:
    # same wire shape serve.py writes: a single data channel with the
    # event name inside the JSON frame
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"


def _build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    # FastAPI resolves handler annotations against the module namespace —
    # the names live in _build_app's scope, so publish them for the
    # decorators below (the imports themselves stay deferred).
    globals().update(Request=Request, JSONResponse=JSONResponse)

    app = FastAPI()

    @app.get("/v1/ping")
    @app.get("/sr/v1/ping")
    async def ping():
        """Shallow liveness — no meta, no model call. Uptime badges probe
        this; /health stays the deep warm-up it was built to be."""
        return JSONResponse({"ok": True})

    @app.get("/v1/health")
    @app.get("/sr/v1/health")
    async def health(request: Request):
        """Real warm-up, not a shell: the scroll-into-view health hit pays
        the isolate's cold init (BM25 meta + model TLS) so the visitor's
        ask lands on a warm stack."""
        env = request.scope["env"]
        _meta(env)
        try:
            _w_ask(
                "ping",
                {"ok": {"type": "noul", "instructions": "Is this a ping?"}},
                key=_env(env, "VLLM_SR_API_KEY"),
                base=_env(env, "VLLM_SR_BASE", "https://app.vllm-sr.ai"),
            )
            model_ok = True
        except Exception:  # noqa: BLE001 — a health ping must never 500 the healthcheck
            model_ok = False
        return JSONResponse({"ok": True, "model": model_ok})

    @app.get("/v1/run/{rid}")
    @app.get("/sr/v1/run/{rid}")
    async def get_run(request: Request, rid: str):
        """The stored event stream for one run — the public rid rides the SSE
        `run` event, so a visitor can pull their own record back."""
        env = request.scope["env"]
        row = (
            await env.DB.prepare("SELECT ts, q, events FROM runs WHERE rid=?")
            .bind(rid)
            .first()
        )
        row = row.to_py() if hasattr(row, "to_py") else row
        if not row:
            return JSONResponse({"error": "run not found"}, status_code=404)
        return {
            "id": rid,
            "ts": row["ts"],
            "q": row["q"],
            "events": json.loads(row["events"]),
        }

    @app.get("/v1/ask")
    @app.get("/sr/v1/ask")
    async def ask(request: Request):
        env = request.scope["env"]
        if await _rate_limited(env, request):
            return JSONResponse({"error": "rate limited"}, status_code=429)
        q = (request.query_params.get("q") or "").strip()
        if not q:
            return JSONResponse({"error": "missing q"}, status_code=400)
        # preset click vs typed question — different demand signal for
        # traffic analysis; anything unrecognised counts as typed
        source = request.query_params.get("source")
        source = source if source in ("preset", "typed") else "typed"
        key = _env(env, "VLLM_SR_API_KEY")
        base = _env(env, "VLLM_SR_BASE", "https://app.vllm-sr.ai")
        # this ask builds the isolate's BM25 meta ⇔ it pays the cold init —
        # prepare_begin marks it so the UI names the slow stage honestly
        cold = _META is None

        async def events():
            # async gen — StreamingResponse thread-pools sync iterators and
            # Pyodide has no threads; the sync pipeline calls inside run on
            # this stack where the JSPI suspender lives (same as WSGI)
            from supplements_rag import bm25
            from supplements_rag.pipeline import answer_events

            collected: list[dict] = []
            # opaque public id — a sequential rowid would let anyone walk
            # /v1/run/1../N and read every visitor's stored question
            rid = uuid.uuid4().hex[:12]
            try:
                begin = {"event": "prepare_begin", **({"cold": True} if cold else {})}
                collected.append(begin)
                yield _sse(begin)
                # awaited, not scheduled — the client is connected here, so
                # the stub lands even if the run is abandoned mid-stream
                await _insert_run_stub(env, rid, q, begin, source)
                run_ev = {"event": "run", "id": rid}
                collected.append(run_ev)
                yield _sse(run_ev)
                t = time.monotonic()
                dense = _dense(env, q)
                ev = {"event": "prepare", "ms": round((time.monotonic() - t) * 1000)}
                collected.append(ev)
                yield _sse(ev)

                index = _index_view(env, q)
                # the chunk pool = every id the fusion can pick: the sparse
                # top, the dense list, and any rare-entity hits — computed
                # with the same shared fns retrieve() calls internally
                sparse = bm25.score(index, q, top_k=20)
                ids = [cid for cid, _ in sparse]
                ids += [cid for cid, _ in dense]
                ids += bm25.rare_hits(index, q)
                chunks = _fetch_chunks(env, ids)

                for ev in answer_events(
                    q,
                    dense,
                    chunks,
                    index,
                    ask_fn=partial(_w_ask, base=base),
                    stream_fn=partial(_w_chat_stream, key=key, base=base),
                    key=key,
                ):
                    collected.append(ev)
                    yield _sse(ev)
            except Exception as e:  # noqa: BLE001 — fail closed, same as serve.py
                ev = {
                    "event": "status",
                    "status": "unavailable",
                    "error": str(e)[:300],
                }
                collected.append(ev)
                yield _sse(ev)
            finally:
                # the row lands for completed, crashed, and abandoned asks
                # alike — finally runs on generator close (client disconnect)
                schedule(_store_run(env, rid, q, collected, source))

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache"},
        )

    return app


async def _lazy_app(scope, receive, send):
    """ASGI shim — builds the real app on first request, after the startup
    memory snapshot is already taken."""
    global _APP
    if _APP is None:
        _APP = _build_app()
    await _APP(scope, receive, send)


class Default(asgi.entrypoint(_lazy_app)):
    """Worker entrypoint."""
