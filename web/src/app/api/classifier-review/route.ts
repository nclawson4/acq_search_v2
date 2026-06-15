// Dev-only: returns the 50 validation frames joined with model predictions
// from cache/scene_tags.json — used by /classifier-review to inspect misses.
import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

const REPO_ROOT = path.resolve(process.cwd(), "..");
const SCENE_TAGS_PATH = path.join(REPO_ROOT, "ingest", "cache", "scene_tags.json");
const LABELS_PATH = path.join(REPO_ROOT, "eval", "data", "human_visual_v2_labels.json");
const MANIFEST_PATH = path.join(REPO_ROOT, "eval", "data", "label_v2_batch_manifest.json");
const SCENES_DIR = path.join(REPO_ROOT, "ingest", "cache", "scenes");

type Tag = { is_animation?: boolean; talking_head_pose?: string };

async function loadJson<T>(p: string, fallback: T): Promise<T> {
  try {
    return JSON.parse(await fs.readFile(p, "utf-8")) as T;
  } catch {
    return fallback;
  }
}

export async function GET() {
  try {
    const tags = await loadJson<Record<string, Tag>>(SCENE_TAGS_PATH, {});
    const labelsRaw = await loadJson<{ labels?: Record<string, Tag & { video_id?: string; scene_idx?: number; notes?: string }> }>(
      LABELS_PATH,
      {}
    );
    const labels = labelsRaw.labels ?? {};
    const manifest = await loadJson<{ frames?: Array<{ video_id: string; scene_idx: number; frame_path: string; start_s: number; stratum: string }> }>(
      MANIFEST_PATH,
      {}
    );
    const frames = manifest.frames ?? [];
    const stratumByPath: Record<string, string> = {};
    const youtubeByPath: Record<string, string> = {};
    for (const f of frames) {
      stratumByPath[f.frame_path] = f.stratum;
      youtubeByPath[f.frame_path] = `https://www.youtube.com/watch?v=${f.video_id}&t=${Math.max(0, Math.floor(f.start_s) - 2)}s`;
    }

    const rows: Array<{
      frame_path: string;
      video_id: string;
      scene_idx: number;
      stratum: string;
      frame_url: string;
      youtube_url: string;
      gt_is_animation: boolean | null;
      pred_is_animation: boolean | null;
      gt_pose: string | null;
      pred_pose: string | null;
      anim_miss: boolean;
      pose_miss: boolean;
    }> = [];

    for (const [path, lab] of Object.entries(labels)) {
      const ingestKey = `ingest/${path}`;
      const pred = tags[ingestKey] || {};
      const gt_anim = lab.is_animation ?? null;
      const pred_anim = pred.is_animation ?? null;
      const gt_pose = lab.talking_head_pose ?? null;
      const pred_pose = pred.talking_head_pose ?? null;
      const video_id = lab.video_id ?? path.split("/")[1] ?? "";
      const scene_idx = lab.scene_idx ?? Number(path.match(/scene_(\d+)/)?.[1] ?? 0);
      rows.push({
        frame_path: path,
        video_id,
        scene_idx,
        stratum: stratumByPath[path] || "unknown",
        frame_url: `/api/frames/${video_id}/${String(scene_idx).padStart(4, "0")}`,
        youtube_url: youtubeByPath[path] || `https://www.youtube.com/watch?v=${video_id}`,
        gt_is_animation: gt_anim,
        pred_is_animation: pred_anim,
        gt_pose,
        pred_pose,
        anim_miss: gt_anim !== null && pred_anim !== null && gt_anim !== pred_anim,
        pose_miss: gt_pose !== null && pred_pose !== null && gt_pose !== pred_pose,
      });
    }

    rows.sort((a, b) => {
      // misses first, then by stratum
      const aMiss = (a.anim_miss ? 1 : 0) + (a.pose_miss ? 1 : 0);
      const bMiss = (b.anim_miss ? 1 : 0) + (b.pose_miss ? 1 : 0);
      if (aMiss !== bMiss) return bMiss - aMiss;
      return a.stratum.localeCompare(b.stratum);
    });

    // Summary
    const total = rows.length;
    const animMisses = rows.filter((r) => r.anim_miss).length;
    const poseMisses = rows.filter((r) => r.pose_miss).length;
    const byStratum: Record<string, { n: number; anim_miss: number; pose_miss: number }> = {};
    for (const r of rows) {
      const s = (byStratum[r.stratum] ||= { n: 0, anim_miss: 0, pose_miss: 0 });
      s.n++;
      if (r.anim_miss) s.anim_miss++;
      if (r.pose_miss) s.pose_miss++;
    }

    return NextResponse.json({
      total,
      anim_misses: animMisses,
      pose_misses: poseMisses,
      anim_acc: total > 0 ? (total - animMisses) / total : 0,
      pose_acc: total > 0 ? (total - poseMisses) / total : 0,
      by_stratum: byStratum,
      rows,
    });
  } catch (err) {
    return NextResponse.json({ error: err instanceof Error ? err.message : String(err) }, { status: 500 });
  }
}
