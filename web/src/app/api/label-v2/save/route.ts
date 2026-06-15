// Dev-only: persist a single label for the visual-v2 validation set.
import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const REPO_ROOT = path.resolve(process.cwd(), "..");
const LABELS_PATH = path.join(REPO_ROOT, "eval", "data", "human_visual_v2_labels.json");

type SavePayload = {
  frame_path: string;
  video_id: string;
  scene_idx: number;
  is_animation?: boolean;
  talking_head_pose?: "front_view" | "side_view" | "none";
  notes?: string;
};

async function load(): Promise<Record<string, Record<string, unknown>>> {
  try {
    const d = JSON.parse(await fs.readFile(LABELS_PATH, "utf-8"));
    return d.labels ?? d ?? {};
  } catch {
    return {};
  }
}

export async function POST(req: NextRequest) {
  if (process.env.VERCEL === "1") {
    return NextResponse.json({ error: "disabled in prod" }, { status: 403 });
  }
  let body: SavePayload;
  try {
    body = (await req.json()) as SavePayload;
  } catch {
    return NextResponse.json({ error: "invalid json" }, { status: 400 });
  }
  if (!body.frame_path) {
    return NextResponse.json({ error: "missing frame_path" }, { status: 400 });
  }
  const labels = await load();
  const prev = (labels[body.frame_path] as Record<string, unknown>) ?? {};
  const merged: Record<string, unknown> = {
    video_id: body.video_id,
    scene_idx: body.scene_idx,
    ...prev,
  };
  for (const k of ["is_animation", "talking_head_pose", "notes"] as const) {
    if (body[k] !== undefined) merged[k] = body[k];
  }
  labels[body.frame_path] = merged;
  await fs.mkdir(path.dirname(LABELS_PATH), { recursive: true });
  await fs.writeFile(LABELS_PATH, JSON.stringify({ labels, updated: new Date().toISOString() }, null, 2), "utf-8");
  return NextResponse.json({ ok: true, label: merged });
}
