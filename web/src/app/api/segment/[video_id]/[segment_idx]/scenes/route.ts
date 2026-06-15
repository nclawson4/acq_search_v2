import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export const dynamic = "force-dynamic";

export async function GET(
  _req: NextRequest,
  ctx: { params: Promise<{ video_id: string; segment_idx: string }> }
) {
  const { video_id, segment_idx } = await ctx.params;
  try {
    const resp = await fetch(`${BACKEND_URL}/segment/${encodeURIComponent(video_id)}/${encodeURIComponent(segment_idx)}/scenes`);
    const text = await resp.text();
    return new NextResponse(text, {
      status: resp.status,
      headers: { "content-type": resp.headers.get("content-type") || "application/json" },
    });
  } catch (err) {
    return NextResponse.json(
      { error: "backend unreachable", detail: err instanceof Error ? err.message : String(err) },
      { status: 502 }
    );
  }
}
