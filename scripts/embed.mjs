// Embeds chunks.jsonl with bge-m3 via the Workers AI binding and upserts
// into Vectorize via getPlatformProxy (wrangler bindings, remote mode).
// Checkpointed: re-runs skip already-upserted chunks (data/embed-ledger.json).
// Run: node scripts/embed.mjs [--limit N]

import { getPlatformProxy } from "wrangler";
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const MODEL = "@cf/baai/bge-m3";
const TOK_LIMIT = 15_000; // est. tokens per AI.run call — hard cap is 60k (verified)
const LIMIT = Number(process.argv[2] === "--limit" ? process.argv[3] : 0);

const chunks = readFileSync("data/chunks.jsonl", "utf-8")
  .split("\n").filter(Boolean).map(JSON.parse);
const ledgerPath = "data/embed-ledger.json";
const done = new Set(
  existsSync(ledgerPath) ? JSON.parse(readFileSync(ledgerPath, "utf-8")) : [],
);
const todo = chunks.filter((c) => !done.has(c.chunk_id));
const targets = LIMIT ? todo.slice(0, LIMIT) : todo;
console.log(`${chunks.length} chunks, ${done.size} done, ${targets.length} to embed`);

const { env } = await getPlatformProxy();
const estTok = (c) => (c.title.length + c.text.length) / 3.5; // chars→tokens
let embedded = 0;

async function embedBatch(batch) {
  // ADR-003: embed title + abstract (symmetric with the BM25 channel)
  const texts = batch.map((c) => `${c.title}\n${c.text}`);
  let out;
  try {
    out = await env.AI.run(MODEL, { text: texts });
  } catch (e) {
    if (batch.length > 1 && String(e).includes("Max context")) {
      // est. tokens undercount on number/table-heavy abstracts — split + retry
      const mid = Math.floor(batch.length / 2);
      await embedBatch(batch.slice(0, mid));
      await embedBatch(batch.slice(mid));
      return;
    }
    throw e;
  }
  await env.VECTORIZE.upsert(
    batch.map((c, j) => ({
      id: c.chunk_id,
      values: out.data[j],
      // small metadata only; full text lives in D1 later
      metadata: { pmid: c.pmid, pub_date: c.pub_date },
    })),
  );
  batch.forEach((c) => done.add(c.chunk_id));
  embedded += batch.length;
}

for (let i = 0; i < targets.length; ) {
  const batch = [];
  let tok = 0;
  while (i < targets.length && tok + estTok(targets[i]) <= TOK_LIMIT) {
    tok += estTok(targets[i]);
    batch.push(targets[i]);
    i++;
  }
  if (!batch.length) { batch.push(targets[i]); i++; } // lone oversized chunk still fits under 8192
  await embedBatch(batch);
  writeFileSync(ledgerPath, JSON.stringify([...done]));
  console.log(`upserted ${embedded}/${targets.length}`);
}
console.log("done");
process.exit(0);
