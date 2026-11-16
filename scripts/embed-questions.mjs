// Batch-eval glue: embed eval questions → pull each dense list from Vectorize.
// Dev-only handoff; prod does this inline in the worker.
//   node scripts/embed-questions.mjs [rows.jsonl] [out.json] [queries.json]
import { getPlatformProxy } from "wrangler";
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const rowsPath = process.argv[2] ?? "evals/golden.jsonl";
const outPath = process.argv[3] ?? "data/dense-golden.json";
const dqPath = process.argv[4] ?? "data/dense-queries.json";

const rows = readFileSync(rowsPath, "utf8").trim().split("\n")
  .map((l, i) => ({ id: i, question: JSON.parse(l).question }));
// Keyword-form embedding (dense_query) when the map exists — interrogative
// scaffolding drags vectors; `python -m supplements_rag.query --dense-map
// <rows> --out <queries>` regenerates it.
const dq = existsSync(dqPath)
  ? JSON.parse(readFileSync(dqPath, "utf8")) : {};
const { env } = await getPlatformProxy();

const vecs = await env.AI.run("@cf/baai/bge-m3",
  { text: rows.map(r => dq[r.id] ?? r.question) });
const out = {};
for (let i = 0; i < rows.length; i++) {
  const res = await env.VECTORIZE.query(vecs.data[i], { topK: 20 });
  out[rows[i].id] = res.matches.map(m => [m.id, m.score]);
}
writeFileSync(outPath, JSON.stringify(out));
console.log(`${rows.length} questions embedded + queried → ${outPath}`);
process.exit(0);
