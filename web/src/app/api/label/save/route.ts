// Dev-only route: writes ground-truth labels to disk. Labels then get
// committed to the repo and read by the search service at startup.
// This route is intentionally not exposed in the deployed product (the
// /label page is meant for local labeling only).
import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const REPO_ROOT = path.resolve(process.cwd(), "..");
const LABELS_PATH = path.join(REPO_ROOT, "eval", "data", "human_scene_tags.json");

type SavePayload = {
  video_id: string;
  scene_idx: number;
  frame_path: string;
  format?: string;
  shot_type?: string;
  who?: string;
  has_text?: boolean;
  notes?: string;
};

async function loadLabels(): Promise<Record<string, Record<string, unknown>>> {
  try {
    const txt = await fs.readFile(LABELS_PATH, "utf-8");
    const data = JSON.parse(txt);
    return data.labels ?? data ?? {};
  } catch {
    return {};
  }
}

export async function POST(req: NextRequest) {
  if (process.env.VERCEL === "1") {
    return NextResponse.json(
      { error: "label save disabled in production" },
      { status: 403 }
    );
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

  const labels = await loadLabels();
  const prev = (labels[body.frame_path] as Record<string, unknown>) ?? {};
  const merged: Record<string, unknown> = {
    video_id: body.video_id,
    scene_idx: body.scene_idx,
    ...prev,
  };
  for (const k of ["format", "shot_type", "who", "has_text", "notes"] as const) {
    if (body[k] !== undefined) merged[k] = body[k];
  }
  labels[body.frame_path] = merged;

  await fs.mkdir(path.dirname(LABELS_PATH), { recursive: true });
  await fs.writeFile(
    LABELS_PATH,
    JSON.stringify({ labels, updated: new Date().toISOString() }, null, 2),
    "utf-8"
  );

  return NextResponse.json({ ok: true, label: merged });
}
