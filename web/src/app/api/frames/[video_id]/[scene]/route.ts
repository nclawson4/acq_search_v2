// Dev-only frame server: streams JPEG keyframes from the ingest/frames/ dir.
// Production redirects to Vercel Blob (set FRAMES_BLOB_BASE env var to enable).
import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const REPO_ROOT = path.resolve(process.cwd(), "..");
const FRAMES_DIR = path.join(REPO_ROOT, "ingest", "frames");

export const dynamic = "force-dynamic";

export async function GET(
  _req: NextRequest,
  ctx: { params: Promise<{ video_id: string; scene: string }> }
) {
  const { video_id, scene } = await ctx.params;
  const sceneFile = `scene_${scene}.jpg`;

  // Production: redirect to Vercel Blob if configured
  const blobBase = process.env.FRAMES_BLOB_BASE;
  if (blobBase) {
    return NextResponse.redirect(`${blobBase}/${video_id}/${sceneFile}`, 302);
  }

  // Dev: stream from disk
  const filePath = path.join(FRAMES_DIR, video_id, sceneFile);
  // sanity: ensure filePath stays under FRAMES_DIR
  if (!path.resolve(filePath).startsWith(path.resolve(FRAMES_DIR))) {
    return NextResponse.json({ error: "invalid path" }, { status: 400 });
  }
  try {
    const data = await fs.readFile(filePath);
    return new NextResponse(new Uint8Array(data), {
      status: 200,
      headers: {
        "content-type": "image/jpeg",
        "cache-control": "public, max-age=3600",
      },
    });
  } catch {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
}
