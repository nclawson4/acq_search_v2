// Upload all ingest/frames/<vid>/scene_NNNN.jpg into the Vercel Blob store.
// Preserves the directory structure so the Next.js frames route resolves
// the same pathnames in cloud as it does from disk locally.
//
// Run:
//   cd web && node scripts/upload-frames.mjs
//
// Reads BLOB_READ_WRITE_TOKEN from .env.local (already pulled via vercel env).

import { put } from "@vercel/blob";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const FRAMES_DIR = path.join(REPO_ROOT, "ingest", "frames");

// Read .env.local so the script picks up BLOB_READ_WRITE_TOKEN.
const ENV_LOCAL = path.join(__dirname, "..", ".env.local");
try {
  const envText = await readFile(ENV_LOCAL, "utf-8");
  for (const line of envText.split(/\r?\n/)) {
    const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
    if (m && !process.env[m[1]]) {
      process.env[m[1]] = m[2].replace(/^"(.*)"$/, "$1");
    }
  }
} catch {
  /* .env.local missing — token must be in env */
}

if (!process.env.BLOB_READ_WRITE_TOKEN) {
  console.error("BLOB_READ_WRITE_TOKEN not set; run `vercel env pull .env.local` first.");
  process.exit(1);
}

const CONCURRENCY = 32;
const ALLOW_OVERWRITE = process.argv.includes("--overwrite");

// Walk frames dir and collect all (videoDir, file) pairs.
async function collectFiles() {
  const videoDirs = await readdir(FRAMES_DIR);
  const files = [];
  for (const vid of videoDirs) {
    const dir = path.join(FRAMES_DIR, vid);
    let st;
    try {
      st = await stat(dir);
    } catch {
      continue;
    }
    if (!st.isDirectory()) continue;
    const entries = await readdir(dir);
    for (const f of entries) {
      if (!f.startsWith("scene_") || !f.endsWith(".jpg")) continue;
      files.push({ vid, file: f, abs: path.join(dir, f) });
    }
  }
  return files;
}

async function uploadOne({ vid, file, abs }) {
  const pathname = `${vid}/${file}`;
  const data = await readFile(abs);
  // Retry-with-backoff on transient errors (rate-limit, network). "already exists"
  // surfaces immediately and the caller treats it as success.
  let lastErr;
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      await put(pathname, data, {
        access: "public",
        allowOverwrite: ALLOW_OVERWRITE,
        contentType: "image/jpeg",
      });
      return pathname;
    } catch (e) {
      const msg = String(e && e.message || e);
      if (msg.includes("already exists") || msg.includes("blob_already_exists")) throw e;
      lastErr = e;
      // Backoff: 200ms, 600ms, 1400ms
      await new Promise((r) => setTimeout(r, 200 * (3 ** attempt - 1) / 2));
    }
  }
  throw lastErr;
}

async function main() {
  console.log("Scanning", FRAMES_DIR);
  const files = await collectFiles();
  console.log(`Found ${files.length} frame files. Uploading with concurrency=${CONCURRENCY}...`);

  let done = 0;
  let failed = 0;
  const t0 = Date.now();

  // Simple worker pool
  const queue = files.slice();
  const workers = Array.from({ length: CONCURRENCY }, async () => {
    while (queue.length) {
      const item = queue.shift();
      if (!item) return;
      try {
        await uploadOne(item);
      } catch (e) {
        // Idempotent: blob exists and overwrite not allowed = treat as already-uploaded.
        const msg = String(e && e.message || e);
        if (msg.includes("already exists") || msg.includes("blob_already_exists")) {
          // OK — pre-existing
        } else {
          failed++;
          // Log every failure so we can post-mortem the remaining set.
          process.stderr.write(`FAIL ${item.vid}/${item.file}: ${msg}\n`);
        }
      }
      done++;
      if (done % 200 === 0) {
        const dt = (Date.now() - t0) / 1000;
        const rate = (done / dt).toFixed(1);
        console.log(`  ${done}/${files.length}  (${rate} files/s, ${failed} failed)`);
      }
    }
  });
  await Promise.all(workers);

  const dt = (Date.now() - t0) / 1000;
  console.log(`\nDone in ${dt.toFixed(1)}s — ${done - failed} uploaded (or already present), ${failed} failed.`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
