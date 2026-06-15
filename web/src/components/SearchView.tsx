"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

export type SegmentInfo = {
  video_id: string;
  segment_idx: number;
  start_s: number;
  end_s: number;
  topic_title?: string;
  summary?: string;
  lessons_categories?: string[];
  industries?: string[];
  audience?: string[];
};

export type SearchResult = {
  rank: number;
  score: number;
  judge_score?: number | null;
  why?: string;
  judge_reason?: string;
  structural_dims?: string[];
  video_id: string;
  scene_idx: number;
  start_s: number;
  end_s: number;
  frame_url: string;
  youtube_url: string;
  voice?: string | null;
  voices_present?: string[];
  speakers_count?: string | null;
  segment?: SegmentInfo | null;
  segment_youtube_url?: string | null;
  recency_days?: number | null;
  upload_date?: string | null;
  signals: Record<string, unknown>;
};

export type SegmentScene = {
  scene_idx: number;
  start_s: number;
  end_s: number;
  frame_url: string;
  youtube_url: string;
};

export type SearchResponse = {
  query: string;
  parsed: {
    speaker: string | null;
    required_speakers: string[] | null;
    is_animation: boolean | null;
    talking_head_pose: "front_view" | "none" | null;
    max_age_days: number | null;
    min_age_days: number | null;
  };
  filters_applied: Record<string, unknown>;
  n: number;
  results: SearchResult[];
};

const POSE_LABELS: Record<string, string> = {
  front_view: "front view talking head",
  none: "no clear foreground talking head",
};

// Threshold for "default" view. Results with judge_score < this are hidden
// unless the user opens /debug-mode. Results with no judge_score (rerank
// disabled or model unavailable) are always shown.
const DEFAULT_JUDGE_THRESHOLD = 0.4;

type Mode = "default" | "debug";

export function SearchView({ mode }: { mode: Mode }) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);

  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const resp = await fetch("/api/search", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query, k: 20 }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json: SearchResponse = await resp.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "search failed");
    } finally {
      setLoading(false);
    }
  }

  const all = data?.results ?? [];
  const visible =
    mode === "debug"
      ? all
      : all.filter(
          (r) => r.judge_score == null || r.judge_score >= DEFAULT_JUDGE_THRESHOLD
        );
  const hidden = all.length - visible.length;

  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      <main className="mx-auto max-w-5xl px-4 py-12">
        {mode === "debug" && (
          <div className="rounded-lg bg-amber-50 dark:bg-amber-950 text-amber-800 dark:text-amber-300 text-xs px-3 py-2 mb-6 flex items-center justify-between">
            <span>
              <strong>debug mode</strong> — all top-K shown, including results the judge marked irrelevant.
            </span>
            <Link href="/" className="font-medium underline">
              back to default view →
            </Link>
          </div>
        )}

        <header className="text-center mb-10">
          <h1 className="text-3xl font-semibold tracking-tight">
            Search the ACQ media database (demo - past 6 months)
          </h1>
        </header>

        <form onSubmit={runSearch} className="flex gap-2 mb-8">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. Alex at the whiteboard talking about churn"
            className="flex-1 rounded-full border border-zinc-300 dark:border-zinc-700 px-5 py-3 text-base focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-zinc-900"
            autoFocus
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded-full bg-blue-600 px-6 py-3 text-white font-medium disabled:opacity-50 hover:bg-blue-700"
          >
            {loading ? "searching..." : "search"}
          </button>
        </form>

        {error && (
          <div className="rounded-lg bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300 p-4 mb-6 text-sm">
            {error}
          </div>
        )}

        {data && (
          <>
            <div className="mb-6 text-sm text-zinc-500 flex flex-wrap gap-3 items-center">
              <span>
                {visible.length} {visible.length === 1 ? "result" : "results"}
              </span>
              {data.parsed.speaker && (
                <span className="rounded-full bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 px-3 py-1">
                  speaker: {data.parsed.speaker}
                </span>
              )}
              {data.parsed.required_speakers && data.parsed.required_speakers.length > 1 && (
                <span className="rounded-full bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 px-3 py-1">
                  co-present: {data.parsed.required_speakers.join(" + ")}
                </span>
              )}
              {data.parsed.is_animation === true && (
                <span className="rounded-full bg-purple-50 dark:bg-purple-950 text-purple-700 dark:text-purple-300 px-3 py-1">
                  animation only
                </span>
              )}
              {data.parsed.talking_head_pose && (
                <span className="rounded-full bg-purple-50 dark:bg-purple-950 text-purple-700 dark:text-purple-300 px-3 py-1">
                  pose: {POSE_LABELS[data.parsed.talking_head_pose] || data.parsed.talking_head_pose}
                </span>
              )}
              {data.parsed.max_age_days != null && (
                <span className="rounded-full bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 px-3 py-1">
                  uploaded ≤ {data.parsed.max_age_days}d ago
                </span>
              )}
              {data.parsed.min_age_days != null && (
                <span className="rounded-full bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 px-3 py-1">
                  uploaded ≥ {data.parsed.min_age_days}d ago
                </span>
              )}
            </div>

            <div className="space-y-4">
              {visible.map((r) => (
                <ResultCard key={`${r.video_id}-${r.scene_idx}`} r={r} />
              ))}
            </div>

            {visible.length === 0 && (
              <div className="text-center text-zinc-500 text-sm mt-8 space-y-2">
                <p>No results scored above {(DEFAULT_JUDGE_THRESHOLD * 100).toFixed(0)}% relevance.</p>
                {mode === "default" && hidden > 0 && (
                  <p>
                    <Link href="/debug-mode" className="text-blue-600 dark:text-blue-400 underline">
                      Open debug mode
                    </Link>{" "}
                    to see all {hidden} retrieved results (some may still be useful — the judge isn&apos;t perfect).
                  </p>
                )}
              </div>
            )}
          </>
        )}

        {!data && !loading && !error && (
          <div className="text-center text-zinc-500 text-sm mt-12 space-y-3">
            <p className="font-medium text-zinc-700 dark:text-zinc-300">try a search like:</p>
            <ul className="space-y-1">
              <li>&quot;Sharran in blue blazer with $100M books&quot;</li>
              <li>&quot;Alex at the whiteboard talking about acquisition&quot;</li>
              <li>&quot;yellow caption about two and a half million&quot;</li>
              <li>&quot;Sharran on real estate from over 1 month ago&quot;</li>
            </ul>
            {mode === "default" && (
              <p className="pt-3 text-xs text-zinc-400">
                Default view hides results scored below {(DEFAULT_JUDGE_THRESHOLD * 100).toFixed(0)}% relevance by the judge.{" "}
                <Link href="/debug-mode" className="underline">
                  Show everything →
                </Link>
              </p>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDuration(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return `${m}m${r > 0 ? ` ${r}s` : ""}`;
}

function ResultCard({ r }: { r: SearchResult }) {
  const [scenes, setScenes] = useState<SegmentScene[] | null>(null);

  const seg = r.segment;
  const segDuration = seg ? seg.end_s - seg.start_s : 0;
  const sceneDuration = r.end_s - r.start_s;
  const dim = typeof r.judge_score === "number" && r.judge_score < 0.4;

  // Auto-load all scenes in the segment on mount, so the scroll strip is ready immediately.
  useEffect(() => {
    if (!seg) return;
    let cancelled = false;
    fetch(`/api/segment/${seg.video_id}/${seg.segment_idx}/scenes`)
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) setScenes(data.scenes || []);
      })
      .catch(() => {
        if (!cancelled) setScenes([]);
      });
    return () => {
      cancelled = true;
    };
  }, [seg]);

  return (
    <div
      className={`rounded-xl border overflow-hidden transition-opacity ${
        dim
          ? "border-zinc-200/50 dark:border-zinc-800/50 opacity-60"
          : "border-zinc-200 dark:border-zinc-800"
      }`}
    >
      {/* ── TOP: text-only details strip ────────────────────────────────── */}
      {seg && (
        <div className="bg-zinc-50 dark:bg-zinc-900/50 px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="rounded bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 text-[10px] px-1.5 py-0.5 font-mono">
                  #{r.rank}
                </span>
                <h3 className="text-base font-semibold truncate">
                  {seg.topic_title || `Segment ${seg.segment_idx}`}
                </h3>
              </div>
              {seg.summary && (
                <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-snug">{seg.summary}</p>
              )}
              {r.judge_reason && (
                <p className="text-xs text-zinc-700 dark:text-zinc-300 leading-snug mt-1.5">
                  <span className="font-semibold not-italic">Why this matched:</span>{" "}
                  <span className="italic">{r.judge_reason}</span>
                </p>
              )}
              <div className="flex flex-wrap gap-1.5 mt-2">
                {r.voice && (
                  <span className="rounded-full bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 text-[10px] px-2 py-0.5">
                    voice: {r.voice}
                  </span>
                )}
                {r.speakers_count && (
                  <span className="rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 text-[10px] px-2 py-0.5">
                    {r.speakers_count}
                  </span>
                )}
                {(seg.lessons_categories || []).slice(0, 3).map((l, i) => (
                  <span key={i} className="rounded-full bg-purple-50 dark:bg-purple-950 text-purple-700 dark:text-purple-300 text-[10px] px-2 py-0.5">
                    {l}
                  </span>
                ))}
                {(seg.audience || []).slice(0, 2).map((a, i) => (
                  <span key={i} className="rounded-full bg-amber-50 dark:bg-amber-950 text-amber-700 dark:text-amber-300 text-[10px] px-2 py-0.5">
                    for: {a.replace(/_/g, " ")}
                  </span>
                ))}
                {r.recency_days != null && (
                  <span className="rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 text-[10px] px-2 py-0.5">
                    {r.recency_days < 30
                      ? `${r.recency_days}d ago`
                      : r.recency_days < 365
                      ? `${Math.round(r.recency_days / 30)}mo ago`
                      : `${(r.recency_days / 365).toFixed(1)}yr ago`}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── BOTTOM: left = primary segment thumbnail, right = horizontal scroll strip ── */}
      <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)] gap-3 p-3">
        {/* LEFT: primary segment scene with a transparent play icon overlay */}
        <div>
          <a
            href={r.segment_youtube_url || r.youtube_url}
            target="_blank"
            rel="noopener noreferrer"
            className="group block rounded-lg overflow-hidden bg-zinc-100 dark:bg-zinc-900 relative aspect-video"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={r.frame_url}
              alt={`segment ${seg?.segment_idx ?? r.scene_idx}`}
              className="object-cover w-full h-full group-hover:scale-105 transition-transform"
            />
            <div className="absolute top-2 left-2 rounded bg-black/70 text-white text-[10px] px-1.5 py-0.5 uppercase tracking-wide">
              segment
            </div>
            <div className="absolute bottom-2 right-2 rounded bg-black/70 text-white text-[10px] px-1.5 py-0.5 font-mono">
              {seg ? `${formatTime(seg.start_s)} · ${formatDuration(segDuration)}` : `${formatTime(r.start_s)} · ${formatDuration(sceneDuration)}`}
            </div>
            {/* Transparent play icon overlay (segment thumbnail only) */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <svg
                viewBox="0 0 24 24"
                fill="currentColor"
                aria-hidden="true"
                className="w-14 h-14 text-white/70 drop-shadow-[0_2px_6px_rgba(0,0,0,0.6)] group-hover:text-white group-hover:scale-110 transition-all"
              >
                <path d="M8 5v14l11-7z" />
              </svg>
            </div>
          </a>
        </div>

        {/* RIGHT: vertical 2-wide scroll strip of all scenes in the segment */}
        <SceneStrip scenes={scenes} currentSceneIdx={r.scene_idx} />
      </div>
    </div>
  );
}

function SceneStrip({
  scenes,
  currentSceneIdx,
}: {
  scenes: SegmentScene[] | null;
  currentSceneIdx: number;
}) {
  const stripRef = (el: HTMLDivElement | null) => {
    if (!el) return;
    // Reassign on every render — cheap and safe; ensures only one handler stays attached.
    el.onwheel = (e: WheelEvent) => {
      // Only hijack the wheel if the strip can actually scroll vertically.
      const canScroll = el.scrollHeight > el.clientHeight + 1;
      if (!canScroll) return;
      const atTop = el.scrollTop <= 0 && e.deltaY < 0;
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1 && e.deltaY > 0;
      if (atTop || atBottom) return; // let the page scroll past once we're at an edge
      e.preventDefault();
      el.scrollTop += e.deltaY;
    };
  };

  return (
    <div className="min-w-0 flex flex-col">
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-[11px] font-medium text-zinc-700 dark:text-zinc-300">
          {scenes === null
            ? "Scenes in this segment"
            : `${scenes.length} ${scenes.length === 1 ? "scene" : "scenes"} in this segment`}
        </div>
        <div className="text-[10px] text-zinc-400">scroll ↕</div>
      </div>
      {scenes === null ? (
        <div className="flex-1 min-h-[180px] flex items-center justify-center rounded bg-zinc-50 dark:bg-zinc-900/40 text-xs text-zinc-500">
          loading scenes…
        </div>
      ) : scenes.length === 0 ? (
        <div className="flex-1 min-h-[180px] flex items-center justify-center rounded bg-zinc-50 dark:bg-zinc-900/40 text-xs text-zinc-500">
          no scenes available
        </div>
      ) : (
        <div
          ref={stripRef}
          className="grid grid-cols-2 gap-2 overflow-y-auto pr-2 rounded max-h-[360px]"
          style={{ scrollbarWidth: "thin" }}
        >
          {scenes.map((s) => {
            const isCurrent = s.scene_idx === currentSceneIdx;
            return (
              <a
                key={s.scene_idx}
                href={s.youtube_url}
                target="_blank"
                rel="noopener noreferrer"
                className={`relative block w-full h-[120px] rounded-md overflow-hidden bg-zinc-100 dark:bg-zinc-900 border ${
                  isCurrent
                    ? "border-blue-500 ring-1 ring-blue-500"
                    : "border-zinc-200 dark:border-zinc-800 hover:border-blue-400 dark:hover:border-blue-500"
                }`}
                title={`scene ${s.scene_idx} · ${formatTime(s.start_s)}`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={s.frame_url}
                  alt={`scene ${s.scene_idx}`}
                  className="block w-full h-full object-cover"
                />
                <div className="absolute bottom-1 right-1 rounded bg-black/70 text-white text-[9px] px-1 py-0.5 font-mono">
                  {formatTime(s.start_s)}
                </div>
                {isCurrent && (
                  <div className="absolute top-1 left-1 rounded bg-blue-600 text-white text-[9px] px-1 py-0.5 uppercase tracking-wide">
                    match
                  </div>
                )}
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}
