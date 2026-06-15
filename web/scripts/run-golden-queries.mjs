// Hit the live backend with the 5 golden queries. Save the full responses to
// src/data/golden-queries.json so the home page renders them as proof of
// correctness. Re-run anytime the index changes — but be aware the saved
// snapshot is what the UI will display.
//
// Run:
//   cd web && node scripts/run-golden-queries.mjs

import { writeFile } from "node:fs/promises";

const BACKEND_URL = "https://nclawson4--acq-search-v2-backend-fastapi-app.modal.run";

const QUERIES = [
  "Leila talking about leadership, talking head video",
  "Sharran less than 3 weeks ago talking about real estate",
  "Animations talking about stress and anxiety",
  "Alex talking about churn",
  "Alex writing on a whiteboard",
];

async function run(q) {
  const t0 = Date.now();
  const resp = await fetch(`${BACKEND_URL}/search`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query: q, k: 5, rerank: true }),
  });
  const dt = Date.now() - t0;
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status} for query: ${q}`);
  }
  const data = await resp.json();
  return { q, latency_ms: dt, response: data };
}

async function main() {
  const out = { generated_at: new Date().toISOString(), backend: BACKEND_URL, queries: [] };
  for (const q of QUERIES) {
    process.stdout.write(`  ${q}... `);
    try {
      const row = await run(q);
      out.queries.push(row);
      const n = row.response.n;
      const top = row.response.results?.[0];
      console.log(`OK n=${n} top=${top?.video_id}/${top?.scene_idx} judge=${top?.judge_score} (${row.latency_ms} ms)`);
    } catch (e) {
      console.log(`FAIL ${e.message}`);
      out.queries.push({ q, error: String(e.message || e) });
    }
  }
  await writeFile("src/data/golden-queries.json", JSON.stringify(out, null, 2));
  console.log("\nwrote src/data/golden-queries.json");
}

main();
