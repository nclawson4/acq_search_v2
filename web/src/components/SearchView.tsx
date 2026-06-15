"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import HomeSections from "./HomeSections";

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

// Example queries shown in the empty state. Clicking one populates the input
// and immediately runs the search.
const EXAMPLE_QUERIES = [
  "Leila talking about leadership, talking head video",
  "Alex talking about churn",
  "Sharran less than 3 weeks ago talking about real estate",
  "Animations talking about stress and anxiety",
  "Alex writing on a whiteboard",
];

// Stages of a search request, in the order the editor sees them progress.
// percentStart drives the "current stage" indicator so the chip and the
// bar stay in sync.
type SearchStage = { label: string; help: string; percentStart: number };
const SEARCH_STAGES: SearchStage[] = [
  { label: "Understanding query",  help: "GPT-4o-mini extracts speaker, topic, visual, time", percentStart: 0 },
  { label: "Searching the index",  help: "CLIP visual + transcript BM25 + segment-text hybrid", percentStart: 25 },
  { label: "Ranking results",      help: "GPT-4o-mini judges each candidate against the topic", percentStart: 60 },
];

// Search latency model used to animate the progress bar. Paced to match the
// real backend stages: parse ~1 s, retrieve ~1.5 s, rerank ~2.5 s, then a
// short tail for response wrap-up. The bar fills evenly through the middle
// instead of sprinting at the start or stalling near the end. After the
// expected warm window it holds at 95% rather than creeping, so a cold
// start no longer looks "stuck near done."
function progressForElapsed(ms: number): number {
  const t = ms / 1000;
  if (t < 1.0) return t * 25;                       // parse:   0  -> 25  in 1.0 s
  if (t < 2.5) return 25 + (t - 1.0) * (35 / 1.5);  // retrieve:25 -> 60  in 1.5 s
  if (t < 5.0) return 60 + (t - 2.5) * 10;          // rerank:  60 -> 85  in 2.5 s
  if (t < 7.0) return 85 + (t - 5.0) * 5;           // wrap-up: 85 -> 95  in 2.0 s
  return 95;                                        // hold at 95 until response arrives
}

function currentStageIndex(percent: number): number {
  for (let i = SEARCH_STAGES.length - 1; i >= 0; i--) {
    if (percent >= SEARCH_STAGES[i].percentStart) return i;
  }
  return 0;
}

type Mode = "default" | "debug";

export function SearchView({ mode }: { mode: Mode }) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [progress, setProgress] = useState(0);

  // Drive the progress bar while a search is in flight. Resets to 0 whenever
  // loading goes false (success, error, or cancellation).
  useEffect(() => {
    if (!loading) {
      setProgress(0);
      return;
    }
    const start = Date.now();
    setProgress(progressForElapsed(0));
    const id = window.setInterval(() => {
      setProgress(progressForElapsed(Date.now() - start));
    }, 80);
    return () => window.clearInterval(id);
  }, [loading]);

  async function doSearch(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const resp = await fetch("/api/search", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: q, k: 20 }),
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

  function runSearch(e: React.FormEvent) {
    e.preventDefault();
    return doSearch(query);
  }

  function runExample(q: string) {
    setQuery(q);
    return doSearch(q);
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
      {/* Persistent top-right link to the source. Stays visible while scrolling
          through the long-form sections beneath the search. */}
      <a
        href="https://github.com/nclawson4/acq_search_v2"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="View source on GitHub"
        className="fixed top-4 right-4 z-50 inline-flex items-center gap-1.5 rounded-full border border-zinc-300 dark:border-zinc-700 bg-white/80 dark:bg-zinc-900/80 backdrop-blur px-3 py-1.5 text-xs font-medium text-zinc-700 dark:text-zinc-200 shadow-sm hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
      >
        <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className="h-4 w-4">
          <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56v-1.97c-3.2.7-3.88-1.54-3.88-1.54-.52-1.34-1.28-1.7-1.28-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.04 1.77 2.72 1.26 3.38.97.11-.75.41-1.26.74-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.18 1.18a11.1 11.1 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.63 1.58.23 2.75.11 3.04.74.8 1.19 1.83 1.19 3.09 0 4.43-2.69 5.4-5.25 5.69.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56C20.71 21.39 24 17.08 24 12 24 5.65 18.85.5 12 .5z"/>
        </svg>
        <span>GitHub</span>
      </a>
      <main className="mx-auto max-w-5xl px-4 py-12">
        {mode === "debug" && (
          <div className="rounded-lg bg-amber-50 dark:bg-amber-950 text-amber-800 dark:text-amber-300 text-xs px-3 py-2 mb-6 flex items-center justify-between">
            <span>
              <strong>debug mode.</strong> All top-K shown, including results the judge marked irrelevant.
            </span>
            <Link href="/" className="font-medium underline">
              back to default view →
            </Link>
          </div>
        )}

        <header className="text-center mb-10">
          <h1 className="text-3xl font-semibold tracking-tight">
            Search the ACQ media database
          </h1>
          <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
            Alex, Leila, and Sharran&apos;s long-form videos analyzed from the last 6 months
          </p>
        </header>

        <form onSubmit={runSearch} className="flex gap-2 mb-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="describe a clip: speaker, topic, time, visual…"
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

        {/* Progress bar + stage indicator — only while a search is in flight.
            Reserves the row even when idle so the layout below doesn't jump on
            every search. Stages name what's happening behind the scenes so
            the editor can read the system. */}
        <div className="mb-6 min-h-[2.75rem]" aria-hidden={!loading}>
          {loading && (
            <>
              <div
                role="progressbar"
                aria-label="search progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(progress)}
                className="h-1.5 w-full rounded-full bg-zinc-200/60 dark:bg-zinc-800/60 overflow-hidden"
              >
                <div
                  className="h-full bg-blue-600 transition-[width] duration-200 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <ol className="mt-2 flex items-center justify-between gap-2 text-[11px]">
                {SEARCH_STAGES.map((s, i) => {
                  const stageIdx = currentStageIndex(progress);
                  const done = i < stageIdx;
                  const active = i === stageIdx;
                  return (
                    <li
                      key={s.label}
                      title={s.help}
                      className={`flex items-center gap-1.5 ${
                        active
                          ? "text-blue-700 dark:text-blue-300 font-medium"
                          : done
                          ? "text-zinc-700 dark:text-zinc-300"
                          : "text-zinc-400 dark:text-zinc-500"
                      }`}
                    >
                      <span
                        aria-hidden
                        className={`inline-flex h-4 w-4 items-center justify-center rounded-full text-[10px] ${
                          done
                            ? "bg-blue-600 text-white"
                            : active
                            ? "bg-blue-600/20 text-blue-700 dark:text-blue-300 ring-2 ring-blue-600 animate-pulse"
                            : "bg-zinc-200 dark:bg-zinc-800 text-zinc-500"
                        }`}
                      >
                        {done ? "✓" : i + 1}
                      </span>
                      <span>{s.label}</span>
                    </li>
                  );
                })}
              </ol>
            </>
          )}
        </div>

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
                    to see all {hidden} retrieved results (some may still be useful; the judge isn&apos;t perfect).
                  </p>
                )}
              </div>
            )}
          </>
        )}

        {!data && !loading && !error && (
          <div className="text-center mt-12 space-y-4">
            <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">try a search like:</p>
            <div className="flex flex-wrap justify-center gap-2">
              {EXAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => runExample(q)}
                  className="rounded-full border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 px-4 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-blue-50 hover:border-blue-400 hover:text-blue-700 dark:hover:bg-blue-950/50 dark:hover:text-blue-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Always-visible long-form sections that explain how the system works,
            what it costs, and how we keep it honest. These appear under the hero
            and the results area both — they're part of the page's narrative, not
            a marketing afterthought. */}
        <HomeSections />
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
  const containerRef = useRef<HTMLDivElement | null>(null);
  const matchRef = useRef<HTMLAnchorElement | null>(null);

  // Wheel-hijack: scroll the strip vertically when the cursor is over it,
  // unless we're already at an edge (then let the page scroll past).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const canScroll = el.scrollHeight > el.clientHeight + 1;
      if (!canScroll) return;
      const atTop = el.scrollTop <= 0 && e.deltaY < 0;
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1 && e.deltaY > 0;
      if (atTop || atBottom) return;
      e.preventDefault();
      el.scrollTop += e.deltaY;
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [scenes]);

  // Once scenes render, center the matching scene inside the strip so the
  // editor doesn't have to hunt for it. Uses getBoundingClientRect math so
  // we scroll ONLY the strip — not the page.
  useEffect(() => {
    const container = containerRef.current;
    const match = matchRef.current;
    if (!container || !match || !scenes || scenes.length === 0) return;
    const cr = container.getBoundingClientRect();
    const mr = match.getBoundingClientRect();
    const matchTopInContainer = mr.top - cr.top + container.scrollTop;
    const target = matchTopInContainer - container.clientHeight / 2 + match.offsetHeight / 2;
    container.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
  }, [scenes, currentSceneIdx]);

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
          ref={containerRef}
          className="grid grid-cols-2 gap-2 overflow-y-auto pr-2 rounded max-h-[360px]"
          style={{ scrollbarWidth: "thin" }}
        >
          {scenes.map((s) => {
            const isCurrent = s.scene_idx === currentSceneIdx;
            return (
              <a
                key={s.scene_idx}
                ref={isCurrent ? matchRef : undefined}
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
