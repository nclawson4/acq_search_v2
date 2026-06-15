import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

const REPO_ROOT = path.resolve(process.cwd(), "..");
const SCENES_DIR = path.join(REPO_ROOT, "ingest", "cache", "scenes");
const LABELS_PATH = path.join(REPO_ROOT, "eval", "data", "human_scene_tags.json");
const BATCH_MANIFEST = path.join(REPO_ROOT, "eval", "data", "label_batch_manifest.json");

const TARGET_COUNT = 100;
const SEED = 20260612;

type Scene = { idx: number; start_s: number; end_s: number; frame_path: string };

// Deterministic seeded RNG (mulberry32)
function rng(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(arr: T[], rand: () => number): T[] {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

type FrameRef = {
  video_id: string;
  scene_idx: number;
  frame_path: string;
  start_s: number;
};

async function buildBatch(): Promise<{ frames: FrameRef[] }> {
  // Reuse a saved batch manifest if present so subsequent reloads show same set
  try {
    const txt = await fs.readFile(BATCH_MANIFEST, "utf-8");
    const parsed = JSON.parse(txt) as { frames: Array<Partial<FrameRef>> };
    // back-compat: if manifest predates start_s, look it up from scenes cache and rewrite
    if (parsed.frames.length && parsed.frames[0].start_s === undefined) {
      const withStart = await hydrateStartTimes(
        parsed.frames as Array<{ video_id: string; scene_idx: number; frame_path: string }>
      );
      await fs.writeFile(BATCH_MANIFEST, JSON.stringify({ frames: withStart }, null, 2), "utf-8");
      return { frames: withStart };
    }
    return parsed as { frames: FrameRef[] };
  } catch {
    // fall through and build fresh
  }

  // Load every scene file
  const files = await fs.readdir(SCENES_DIR);
  const perVideo: Array<{ video_id: string; scene: Scene }[]> = [];
  for (const f of files) {
    if (!f.endsWith(".json")) continue;
    const raw = await fs.readFile(path.join(SCENES_DIR, f), "utf-8");
    const data = JSON.parse(raw) as { video_id: string; scenes: Scene[] };
    if (!data.scenes?.length) continue;
    perVideo.push(data.scenes.map((s) => ({ video_id: data.video_id, scene: s })));
  }

  // Stratified sample: take ~2 frames per video, then top up randomly to TARGET_COUNT
  const rand = rng(SEED);
  const picked: Array<{ video_id: string; scene: Scene }> = [];
  for (const sceneList of perVideo) {
    const shuffled = shuffle(sceneList, rand);
    // pick scenes from different positions to diversify shot types
    const a = shuffled[0];
    if (a) picked.push(a);
    if (sceneList.length > 50 && shuffled[Math.floor(sceneList.length / 2)]) {
      picked.push(shuffled[Math.floor(sceneList.length / 2)]);
    }
  }

  // Cap or top up to TARGET_COUNT
  const final = shuffle(picked, rand).slice(0, TARGET_COUNT);

  const frames: FrameRef[] = final.map((p) => ({
    video_id: p.video_id,
    scene_idx: p.scene.idx,
    frame_path: p.scene.frame_path,
    start_s: p.scene.start_s,
  }));

  await fs.mkdir(path.dirname(BATCH_MANIFEST), { recursive: true });
  await fs.writeFile(BATCH_MANIFEST, JSON.stringify({ frames }, null, 2), "utf-8");
  return { frames };
}

async function hydrateStartTimes(
  rows: Array<{ video_id: string; scene_idx: number; frame_path: string }>
): Promise<FrameRef[]> {
  // Build a video_id -> scene_idx -> start_s lookup on demand
  const cache = new Map<string, Map<number, number>>();
  const out: FrameRef[] = [];
  for (const r of rows) {
    let byScene = cache.get(r.video_id);
    if (!byScene) {
      try {
        const raw = await fs.readFile(path.join(SCENES_DIR, `${r.video_id}.json`), "utf-8");
        const data = JSON.parse(raw) as { scenes: Scene[] };
        byScene = new Map(data.scenes.map((s) => [s.idx, s.start_s]));
      } catch {
        byScene = new Map();
      }
      cache.set(r.video_id, byScene);
    }
    out.push({ ...r, start_s: byScene.get(r.scene_idx) ?? 0 });
  }
  return out;
}

async function loadLabels(): Promise<Record<string, Record<string, unknown>>> {
  try {
    const txt = await fs.readFile(LABELS_PATH, "utf-8");
    const data = JSON.parse(txt);
    // accept either {labels: {...}} or {...} flat
    return data.labels ?? data ?? {};
  } catch {
    return {};
  }
}

export async function GET() {
  try {
    const { frames } = await buildBatch();
    const labels = await loadLabels();
    return NextResponse.json({
      total: frames.length,
      frames: frames.map((f, idx) => ({
        idx,
        video_id: f.video_id,
        scene_idx: f.scene_idx,
        frame_path: f.frame_path,
        frame_url: `/api/frames/${f.video_id}/${String(f.scene_idx).padStart(4, "0")}`,
        start_s: f.start_s,
        youtube_url: `https://www.youtube.com/watch?v=${f.video_id}&t=${Math.max(0, Math.floor(f.start_s) - 2)}s`,
      })),
      labels,
    });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : String(err) },
      { status: 500 }
    );
  }
}
