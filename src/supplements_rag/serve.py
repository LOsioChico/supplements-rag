"""Local SSE server for the exhibit — the same event contract the Worker
will serve. Stdlib only: `python -m supplements_rag.serve` then
GET /v1/ask?q=... → text/event-stream of answer_events().

The dense leg (embed + Vectorize) runs through `scripts/dense-server.mjs`
— Workers AI + Vectorize need wrangler auth, so a warm node sidecar holds
the platform proxy instead of booting a runtime per request. It's spawned
here as a child process; prod does the same calls inline in the Worker.
"""

import atexit
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .pipeline import answer_events

ROOT = Path(__file__).resolve().parents[2]
STATE: dict = {}
RUNS_LOG = ROOT / "data" / "runs.jsonl"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d .()\-]{7,14}\d")
_URLQ = re.compile(r"(\bhttps?://[^\s?\"']+)\?[^\s\"']*")


def _mask(s: str) -> str:
    return _URLQ.sub(r"\1?<query>", _PHONE.sub("<phone>", _EMAIL.sub("<email>", s)))


def _log_run(question: str, events: list[dict]) -> None:
    """One JSON line per ask — the full event stream is the trace, kept so
    real runs can be classified (what failed, which stage) and folded back
    into the golden set. Question text is masked (email/phone/url-query)
    — it's free visitor input, same privacy contract as traces."""
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(
            json.dumps(
                {"ts": time.time(), "q": _mask(question), "events": events},
                ensure_ascii=False,
            )
            + "\n"
        )


def _load() -> None:
    if "chunks" in STATE:
        return
    STATE["chunks"] = {
        json.loads(l)["chunk_id"]: json.loads(l)
        for l in (ROOT / "data" / "chunks.jsonl").open()
    }
    STATE["index"] = json.loads((ROOT / "data" / "bm25.json").read_text())


DENSE_URL = os.environ.get("DENSE_URL", "http://127.0.0.1:8779")
_sidecar_ready = threading.Event()


def _dense(question: str) -> list:
    """Question → [[chunk_id, score], ...] via the warm node sidecar —
    only the embed + Vectorize calls, no per-request runtime boot."""
    _sidecar_ready.wait(timeout=45)  # first request may outlast the boot
    with urllib.request.urlopen(
        f"{DENSE_URL}/dense?q={quote(question)}", timeout=60
    ) as r:
        return json.loads(r.read())


def _start_sidecar() -> subprocess.Popen:
    """Spawn the node sidecar and wait until it answers /health — the
    wrangler boot is paid once here (~3s), not per request."""
    p = subprocess.Popen(
        ["node", "scripts/dense-server.mjs"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(p.terminate)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{DENSE_URL}/health", timeout=1)
            _sidecar_ready.set()
            return p
        except OSError:
            if p.poll() is not None:
                raise RuntimeError("dense sidecar exited during boot")
            time.sleep(0.2)
    raise RuntimeError("dense sidecar did not come up in 30s")


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET")
        self.send_header("access-control-allow-headers", "content-type")

    def do_OPTIONS(self) -> None:  # stdlib handler name
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # stdlib handler name
        parsed = urlparse(self.path)
        if parsed.path == "/v1/health":
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path != "/v1/ask":
            self.send_response(404)
            self.end_headers()
            return
        q = (parse_qs(parsed.query).get("q") or [""])[0].strip()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self._cors()
        self.end_headers()
        events: list[dict] = []
        try:
            _load()
            # the dense leg costs seconds (node handoff) — emit a prepare
            # frame around it so the client's record attributes the wait
            # to the stage that caused it instead of inflating the gate
            self.wfile.write(b'data: {"event": "prepare_begin"}\n\n')
            self.wfile.flush()
            t = time.monotonic()
            dense = _dense(q)
            ms = round((time.monotonic() - t) * 1000)
            self.wfile.write(f'data: {{"event": "prepare", "ms": {ms}}}\n\n'.encode())
            self.wfile.flush()
            events.append({"event": "prepare", "ms": ms})
            key = os.environ["VLLM_SR_API_KEY"]
            for ev in answer_events(q, dense, STATE["chunks"], STATE["index"], key=key):
                events.append(ev)
                self.wfile.write(
                    f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-stream — nothing to repair
        except Exception as e:  # noqa: BLE001 — fail closed: the stream ends with an error event
            try:
                self.wfile.write(
                    f"data: {json.dumps({'event': 'status', 'status': 'unavailable', 'error': str(e)[:300]})}\n\n".encode()
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            # disconnects and crashes exit above — logging in finally keeps
            # abandoned/failed runs in runs.jsonl, not just clean ones
            _log_run(q, events)

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"[serve] {format % args}\n")


def main() -> None:
    port = int(os.environ.get("SR_PORT", "8778"))
    threading.Thread(target=_load, daemon=True).start()
    threading.Thread(target=_start_sidecar, daemon=True).start()
    print(f"supplements-rag SSE on :{port} — GET /v1/ask?q=…", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
