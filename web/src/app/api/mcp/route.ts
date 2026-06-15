// MCP server (Streamable HTTP transport, JSON-RPC). Gates on ?password=demo2026
// or X-MCP-Password header against env MCP_PASSWORD.
//
// Exposes three tools:
//   - search_videos({query, speaker?, format?, k?})
//   - find_similar({video_id, scene_idx?, k?})              (TODO — needs vector lookup)
//   - get_transcript({video_id, start_s, end_s})
//
// All three proxy to the Python FastAPI service via BACKEND_URL.
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const PASSWORD = process.env.MCP_PASSWORD || "demo2026";

const TOOLS = [
  {
    name: "search_videos",
    description:
      "Search the long-form video corpus for moments matching a natural-language query. Returns ranked scenes with YouTube deep-links and timestamps.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Natural-language search query" },
        speaker: {
          type: "string",
          enum: ["alex", "leila", "sharran"],
          description: "Optional speaker filter",
        },
        format: {
          type: "string",
          enum: ["talking_head", "podcast", "phone_qa", "live_qa", "low_production", "whiteboard", "mozi_tank", "animation", "none"],
          description: "Optional content format filter (mozi_tank = the shark-tank-style series where Alex helps a founder scale; animation = full-screen animated/motion-graphics sequence)",
        },
        k: { type: "integer", default: 20, description: "Number of results to return" },
      },
      required: ["query"],
    },
  },
  {
    name: "find_similar",
    description:
      "Given a video_id and scene_idx, find other scenes visually similar to that scene.",
    inputSchema: {
      type: "object",
      properties: {
        video_id: { type: "string" },
        scene_idx: { type: "integer" },
        k: { type: "integer", default: 10 },
      },
      required: ["video_id", "scene_idx"],
    },
  },
  {
    name: "get_transcript",
    description:
      "Get the transcript (utterances) for a video between start_s and end_s seconds.",
    inputSchema: {
      type: "object",
      properties: {
        video_id: { type: "string" },
        start_s: { type: "number" },
        end_s: { type: "number" },
      },
      required: ["video_id", "start_s", "end_s"],
    },
  },
];

function checkAuth(req: NextRequest): boolean {
  const url = new URL(req.url);
  const qp = url.searchParams.get("password");
  if (qp && qp === PASSWORD) return true;
  const header = req.headers.get("x-mcp-password");
  if (header && header === PASSWORD) return true;
  return false;
}

function jsonRpcError(id: string | number | null, code: number, message: string) {
  return { jsonrpc: "2.0" as const, id, error: { code, message } };
}

function jsonRpcResult(id: string | number | null, result: unknown) {
  return { jsonrpc: "2.0" as const, id, result };
}

async function callBackend(path: string, body?: unknown) {
  if (body !== undefined) {
    return fetch(`${BACKEND_URL}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }
  return fetch(`${BACKEND_URL}${path}`);
}

async function handleToolCall(name: string, args: Record<string, unknown>) {
  if (name === "search_videos") {
    const resp = await callBackend("/search", {
      query: args.query,
      speaker: args.speaker,
      format: args.format,
      k: args.k ?? 20,
    });
    if (!resp.ok) {
      const txt = await resp.text();
      return { content: [{ type: "text", text: `Backend error ${resp.status}: ${txt}` }], isError: true };
    }
    const data = await resp.json();
    return {
      content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
    };
  }
  if (name === "find_similar") {
    return {
      content: [{ type: "text", text: "find_similar not yet implemented in backend" }],
      isError: true,
    };
  }
  if (name === "get_transcript") {
    // Future: backend endpoint /transcript
    return {
      content: [{ type: "text", text: "get_transcript not yet implemented in backend" }],
      isError: true,
    };
  }
  return { content: [{ type: "text", text: `unknown tool: ${name}` }], isError: true };
}

export async function POST(req: NextRequest) {
  if (!checkAuth(req)) {
    return NextResponse.json(
      { error: "unauthorized — include ?password=... or X-MCP-Password header" },
      { status: 401 }
    );
  }

  let body: { jsonrpc?: string; id?: string | number | null; method?: string; params?: Record<string, unknown> };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(jsonRpcError(null, -32700, "parse error"), { status: 400 });
  }
  const id = body.id ?? null;
  const method = body.method;
  const params = body.params ?? {};

  if (method === "initialize") {
    return NextResponse.json(
      jsonRpcResult(id, {
        protocolVersion: "2025-06-18",
        capabilities: { tools: {} },
        serverInfo: { name: "acq-search-v2", version: "0.1.0" },
      })
    );
  }

  if (method === "tools/list") {
    return NextResponse.json(jsonRpcResult(id, { tools: TOOLS }));
  }

  if (method === "tools/call") {
    const name = params.name as string;
    const args = (params.arguments as Record<string, unknown>) ?? {};
    try {
      const result = await handleToolCall(name, args);
      return NextResponse.json(jsonRpcResult(id, result));
    } catch (err) {
      return NextResponse.json(jsonRpcError(id, -32603, err instanceof Error ? err.message : String(err)));
    }
  }

  return NextResponse.json(jsonRpcError(id, -32601, `method not found: ${method}`));
}

export async function GET(req: NextRequest) {
  // Lightweight introspection endpoint
  if (!checkAuth(req)) {
    return NextResponse.json(
      { error: "unauthorized — include ?password=..." },
      { status: 401 }
    );
  }
  return NextResponse.json({
    name: "acq-search-v2",
    version: "0.1.0",
    protocol: "mcp/2025-06-18",
    tools: TOOLS.map((t) => ({ name: t.name, description: t.description })),
    usage: "POST JSON-RPC messages to this endpoint. Methods: initialize, tools/list, tools/call.",
  });
}
