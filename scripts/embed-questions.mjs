// Batch-eval glue: embed golden questions → pull each dense list from Vectorize
// → data/dense-golden.json. Dev-only handoff; prod does this inline in the worker.
import { getPlatformProxy } from "wrangler";
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const rows = readFileSync("evals/golden.jsonl", "utf8").trim().split("\n")
  .map((l, i) => ({ id: i, question: JSON.parse(l).question }));
// Keyword-form embedding (dense_query) when the map exists — interrogative
// scaffolding drags vectors; `python -m supplements_rag.query --dense-map
// evals/golden.jsonl` regenerates it.
const dq = existsSync("data/dense-queries.json")
  ? JSON.parse(readFileSync("data/dense-queries.json", "utf8")) : {};
const { env } = await getPlatformProxy();

const vecs = await env.AI.run("@cf/baai/bge-m3",
  { text: rows.map(r => dq[r.id] ?? r.question) });
const out = {};
for (let i = 0; i < rows.length; i++) {
  const res = await env.VECTORIZE.query(vecs.data[i], { topK: 20 });
  out[rows[i].id] = res.matches.map(m => [m.id, m.score]);
}
writeFileSync("data/dense-golden.json", JSON.stringify(out));
console.log(`${rows.length} questions embedded + queried → data/dense-golden.json`);
process.exit(0);
