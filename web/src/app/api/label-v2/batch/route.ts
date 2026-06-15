// Dev-only: builds (or reuses) a 50-frame STRATIFIED HARD sample for validating
// the new 2-field visual classifier (is_animation + talking_head_pose).
//
// Strata (chosen to surface failure modes — NOT to be easy on the classifier):
//   12 frames: decorative-ish parents — topic_segment.lessons_categories=["none"]
//                or scenes shorter than 3 seconds (animation- and title-card-likely)
//   12 frames: talking-head-likely — solo speakers_count, scene duration >= 5s
//   12 frames: multi-voice scenes — voices_present has 2+ named speakers
//                (where front/side ambiguity is most common)
//   14 frames: random across all 66 videos (one per video, no overlap)
import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

const REPO_ROOT = path.resolve(process.cwd(), "..");
const SCENES_DIR = path.join(REPO_ROOT, "ingest", "cache", "scenes");
const AUDIO_TAGS_PATH = path.join(REPO_ROOT, "ingest", "cache", "audio_tags.json");
const TOPIC_SEGMENTS_PATH = path.join(REPO_ROOT, "ingest", "cache", "topic_segments.json");
const SCENE_TO_SEGMENT_PATH = path.join(REPO_ROOT, "ingest", "cache", "scene_to_segment.json");

const HUMAN_LABELS_V2 = path.join(REPO_ROOT, "eval", "data", "human_visual_v2_labels.json");
const BATCH_MANIFEST = path.join(REPO_ROOT, "eval", "data", "label_v2_batch_manifest.json");

const TARGET_COUNT = 50;
const SEED = 20260612;

type Scene = { idx: number; start_s: number; end_s: number; frame_path: string };

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
  stratum: string;
};

async function loadAll(): Promise<{
  scenesByVid: Map<string, Scene[]>;
  audio: Record<string, { voice?: string; voices_present?: string[]; speakers_count?: string }>;
  segments: Array<{ video_id: string; segment_idx: number; lessons_categories?: string[] }>;
  sceneToSegment: Record<string, { video_id: string; segment_idx: number | null }>;
}> {
  const scenesByVid = new Map<string, Scene[]>();
  const files = await fs.readdir(SCENES_DIR);
  for (const f of files) {
    if (!f.endsWith(".json")) continue;
    const raw = await fs.readFile(path.join(SCENES_DIR, f), "utf-8");
    const data = JSON.parse(raw) as { video_id: string; scenes: Scene[] };
    if (data.scenes?.length) scenesByVid.set(data.video_id, data.scenes);
  }

  let audio: Record<string, { voice?: string; voices_present?: string[]; speakers_count?: string }> = {};
  try {
    audio = JSON.parse(await fs.readFile(AUDIO_TAGS_PATH, "utf-8"));
  } catch {}

  let segments: Array<{ video_id: string; segment_idx: number; lessons_categories?: string[] }> = [];
  try {
    const parsed = JSON.parse(await fs.readFile(TOPIC_SEGMENTS_PATH, "utf-8"));
    segments = (Array.isArray(parsed) ? parsed : parsed.segments) ?? [];
  } catch {}

  let sceneToSegment: Record<string, { video_id: string; segment_idx: number | null }> = {};
  try {
    sceneToSegment = JSON.parse(await fs.readFile(SCENE_TO_SEGMENT_PATH, "utf-8"));
  } catch {}

  return { scenesByVid, audio, segments, sceneToSegment };
}

async function buildBatch(): Promise<{ frames: FrameRef[] }> {
  try {
    const txt = await fs.readFile(BATCH_MANIFEST, "utf-8");
    return JSON.parse(txt);
  } catch {}

  const { scenesByVid, audio, segments, sceneToSegment } = await loadAll();
  const segLookup = new Map<string, { lessons_categories: string[] }>();
  for (const s of segments) {
    segLookup.set(`${s.video_id}:${s.segment_idx}`, { lessons_categories: s.lessons_categories ?? [] });
  }

  // Build the candidate pools per stratum
  type Cand = { video_id: string; scene: Scene; stratum: string };

  const decorativeOrShort: Cand[] = [];
  const soloTalkingHead: Cand[] = [];
  const multiVoice: Cand[] = [];
  const allFrames: Cand[] = [];

  for (const [vid, scenes] of scenesByVid) {
    for (const sc of scenes) {
      allFrames.push({ video_id: vid, scene: sc, stratum: "random" });
      const dur = sc.end_s - sc.start_s;
      const audioKey = sc.frame_path;
      const a = audio[audioKey] || {};
      const present = a.voices_present || [];
      const segRef = sceneToSegment[audioKey];
      const seg = segRef ? segLookup.get(`${segRef.video_id}:${segRef.segment_idx}`) : null;
      const decorParent = seg && seg.lessons_categories.length === 1 && seg.lessons_categories[0] === "none";
      if (decorParent || dur < 3) {
        decorativeOrShort.push({ video_id: vid, scene: sc, stratum: "decorative_or_short" });
      } else if ((a.speakers_count === "solo" || (present.length === 1 && present[0])) && dur >= 5) {
        soloTalkingHead.push({ video_id: vid, scene: sc, stratum: "solo_talking_head" });
      }
      if (present.filter((n) => ["alex", "leila", "sharran"].includes(n)).length >= 2) {
        multiVoice.push({ video_id: vid, scene: sc, stratum: "multi_voice" });
      }
    }
  }

  const rand = rng(SEED);
  const dedupe = new Set<string>();
  const out: FrameRef[] = [];
  function pickFrom(pool: Cand[], n: number) {
    const shuffled = shuffle(pool, rand);
    for (const c of shuffled) {
      if (out.length - (TARGET_COUNT - 14) >= 0 && out.length >= TARGET_COUNT) break;
      const key = `${c.video_id}:${c.scene.idx}`;
      if (dedupe.has(key)) continue;
      dedupe.add(key);
      out.push({
        video_id: c.video_id,
        scene_idx: c.scene.idx,
        frame_path: c.scene.frame_path,
        start_s: c.scene.start_s,
        stratum: c.stratum,
      });
      if (out.filter((f) => f.stratum === c.stratum).length >= n) break;
    }
  }

  pickFrom(decorativeOrShort, 12);
  pickFrom(soloTalkingHead, 12);
  pickFrom(multiVoice, 12);

  // Final 14 random across videos NOT already picked
  const videosCovered = new Set(out.map((o) => o.video_id));
  const remaining = allFrames.filter(
    (c) => !dedupe.has(`${c.video_id}:${c.scene.idx}`)
  );
  // Prefer videos we haven't sampled from yet
  const preferred = remaining.filter((c) => !videosCovered.has(c.video_id));
  const fallback = remaining.filter((c) => videosCovered.has(c.video_id));
  const finalPool = shuffle(preferred, rand).concat(shuffle(fallback, rand));
  for (const c of finalPool) {
    if (out.length >= TARGET_COUNT) break;
    const key = `${c.video_id}:${c.scene.idx}`;
    if (dedupe.has(key)) continue;
    dedupe.add(key);
    out.push({ ...c, frame_path: c.scene.frame_path, scene_idx: c.scene.idx, start_s: c.scene.start_s });
  }

  await fs.mkdir(path.dirname(BATCH_MANIFEST), { recursive: true });
  await fs.writeFile(BATCH_MANIFEST, JSON.stringify({ frames: out }, null, 2), "utf-8");
  return { frames: out };
}

async function loadLabels(): Promise<Record<string, Record<string, unknown>>> {
  try {
    const data = JSON.parse(await fs.readFile(HUMAN_LABELS_V2, "utf-8"));
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
        stratum: f.stratum,
        frame_url: `/api/frames/${f.video_id}/${String(f.scene_idx).padStart(4, "0")}`,
        start_s: f.start_s,
        youtube_url: `https://www.youtube.com/watch?v=${f.video_id}&t=${Math.max(0, Math.floor(f.start_s) - 2)}s`,
      })),
      labels,
    });
  } catch (err) {
    return NextResponse.json({ error: err instanceof Error ? err.message : String(err) }, { status: 500 });
  }
}
