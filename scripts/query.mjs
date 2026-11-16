// Dev smoke: embed a question + pull its dense list from Vectorize → data/dense.json
// (prod shape = the worker does this inline; the file handoff is local-test glue)
import { getPlatformProxy } from "wrangler";
import { writeFileSync } from "node:fs";
const q = process.argv[2] ?? "does magnesium improve sleep quality";
const { env } = await getPlatformProxy();
const vec = await env.AI.run("@cf/baai/bge-m3", { text: [q] });
const res = await env.VECTORIZE.query(vec.data[0], { topK: 20 });
writeFileSync("data/dense.json", JSON.stringify(res.matches.map(m => [m.id, m.score])));
console.log(`${res.matches.length} dense hits → data/dense.json`);
process.exit(0);
