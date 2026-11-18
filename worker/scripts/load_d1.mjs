// node worker/scripts/load_d1.mjs — load data/ artifacts into remote D1.
//
// The worker never sees data/*.jsonl (87MB chunks + 35MB index can't ship
// in the bundle): chunks become pmid-keyed rows, the doc-keyed BM25 index
// gets inverted into one term-indexed postings row per vocabulary term,
// and corpus stats land in bm25_meta. Query time then reads only the
// question's terms — data/README.md prescribes exactly this split.
//
// SQL literals are single-quote-escaped; INSERT OR REPLACE keeps re-runs
// idempotent. Node (like embed.mjs) — the Workers tooling lives here.
import { execFileSync } from "node:child_process";
import {
  createReadStream,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createInterface } from "node:readline";

const DB = "supplements-rag";
const DIR = new URL("../.d1-parts/", import.meta.url).pathname;
// wrangler d1 execute stalls on very large files — ~7MB parts stay safe
const PART_BYTES = 7 << 20;
// --terms-only resumes after the chunk pass already shipped
const TERMS_ONLY = process.argv.includes("--terms-only");

mkdirSync(DIR, { recursive: true });
for (const f of readdirSync(DIR)) rmSync(DIR + f);

let part = 0;
let buf = [];
let bytes = 0;
const flush = () => {
  if (!buf.length) return;
  writeFileSync(
    `${DIR}part-${String(part++).padStart(3, "0")}.sql`,
    buf.join(""),
  );
  buf = [];
  bytes = 0;
};
const stmt = (s) => {
  buf.push(s);
  bytes += s.length;
  if (bytes > PART_BYTES) flush();
};
const q = (s) => `'${String(s).replaceAll("'", "''")}'`;

// 1) chunks — the whole line as body so no field is lost at read time
if (!TERMS_ONLY) {
  const rl = createInterface({ input: createReadStream("data/chunks.jsonl") });
  let n = 0;
  for await (const line of rl) {
    if (!line.trim()) continue;
    const c = JSON.parse(line);
    stmt(
      `INSERT OR REPLACE INTO chunks (chunk_id, body) VALUES (${q(c.chunk_id)}, ${q(line)});\n`,
    );
    if (++n % 4000 === 0) console.error(`chunks ${n}`);
  }
  flush();
  console.error(`chunks done (${n})`);
}

// 2) bm25 — docs {cid:{tf,len}} → term → {cid:[tf,len]} postings. The
//    doc length rides the pair so scoring never needs a second lookup;
//    postings shard at SHARD_PAIRS because stopwords sit in ~17k docs and
//    one row overshoots D1's statement size (SQLITE_TOOBIG).
const SHARD_PAIRS = 3000;
{
  const idx = JSON.parse(readFileSync("data/bm25.json", "utf8"));
  const postings = new Map();
  for (const [cid, d] of Object.entries(idx.docs)) {
    for (const [t, f] of Object.entries(d.tf)) {
      let p = postings.get(t);
      if (!p) postings.set(t, (p = {}));
      p[cid] = [f, d.len];
    }
  }
  console.error(`vocab ${postings.size} terms`);
  for (const [term, p] of postings) {
    const df = idx.df[term] ?? Object.keys(p).length;
    const entries = Object.entries(p);
    for (let shard = 0; shard * SHARD_PAIRS < entries.length; shard++) {
      const slice = Object.fromEntries(
        entries.slice(shard * SHARD_PAIRS, (shard + 1) * SHARD_PAIRS),
      );
      stmt(
        `INSERT OR REPLACE INTO bm25_postings (term, shard, df, postings) VALUES (${q(term)}, ${shard}, ${df}, ${q(JSON.stringify(slice))});\n`,
      );
    }
  }
  flush();
  for (const [k, v] of Object.entries({
    n_docs: Object.keys(idx.docs).length,
    avgdl: idx.avgdl,
    k1: idx.k1,
    b: idx.b,
  })) {
    stmt(
      `INSERT OR REPLACE INTO bm25_meta (k, v) VALUES (${q(k)}, ${q(JSON.stringify(v))});\n`,
    );
  }
  flush();
}

// 3) ship the parts
const files = readdirSync(DIR)
  .filter((f) => f.endsWith(".sql"))
  .sort();
// --from=N resumes at part N (execute is idempotent; failures are
// transient upload issues, not data problems)
const FROM = Number(process.argv.find((a) => a.startsWith("--from="))?.slice(7) ?? 0);
files.slice(FROM).forEach((f, i) => {
  console.error(`apply ${f} (${FROM + i + 1}/${files.length})`);
  execFileSync(
    "npx",
    ["wrangler", "d1", "execute", DB, "--remote", "--file", `${DIR}${f}`, "--yes"],
    { stdio: "inherit" },
  );
});
console.error(`done — ${files.length} parts applied`);
