// Dense sidecar: holds one wrangler platform proxy warm so each question
// pays only the two real API calls (embed + Vectorize query), not a
// workerd boot per request. Spawned as a child by `python -m
// supplements_rag.serve`; also runnable alone for the CLI path.
// Same calls query.mjs makes — the per-request version of it.
import { getPlatformProxy } from "wrangler";
import { createServer } from "node:http";

const PORT = Number(process.env.DENSE_PORT ?? 8779);
const MODEL = "@cf/baai/bge-m3";
const { env } = await getPlatformProxy({ configPath: "worker/wrangler.toml" });
// same question → same embedding → same hits; cache instead of re-paying
const cache = new Map();

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  if (url.pathname === "/health") {
    res.end(JSON.stringify({ ok: true }));
    return;
  }
  if (url.pathname !== "/dense") {
    res.writeHead(404);
    res.end();
    return;
  }
  const q = (url.searchParams.get("q") ?? "").trim();
  try {
    if (!cache.has(q)) {
      const vec = await env.AI.run(MODEL, { text: [q] });
      const r = await env.VECTORIZE.query(vec.data[0], { topK: 20 });
      cache.set(q, r.matches.map((m) => [m.id, m.score]));
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ model: MODEL, matches: cache.get(q) }));
  } catch (e) {
    res.writeHead(502);
    res.end(String(e));
  }
});
server.listen(PORT, "127.0.0.1", () =>
  console.error(`dense sidecar on :${PORT}`),
);
