// Marengo 3 vs Google Gemini Embedding 2 — multimodal video retrieval, side by side.
// Same golden queries, two pipelines. Top-5 moments each; the reader looks at the frames.
import Link from "next/link";
import data from "@/data/marengo-google-comparison.json";

export const metadata = {
  title: "Marengo vs Google",
};

type Hit = { video_id: string; t: number; score: number; frame: string; matched: boolean };
type Row = { query: string; marengo: Hit[]; google: Hit[] };

function yt(h: Hit) {
  return `https://www.youtube.com/watch?v=${h.video_id}&t=${Math.floor(h.t)}s`;
}
function mmss(t: number) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function Column({ label, accent, hits }: { label: string; accent: string; hits: Hit[] }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.18em] mb-3 font-semibold" style={{ color: accent }}>
        {label}
      </p>
      {hits.length === 0 ? (
        <div className="grid grid-cols-5 gap-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="aspect-video rounded-md bg-zinc-100 dark:bg-zinc-900 border border-dashed border-zinc-300 dark:border-zinc-700 flex items-center justify-center"
            >
              <span className="text-[10px] text-zinc-400">awaiting labels</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-5 gap-2">
          {hits.slice(0, 5).map((h, i) => (
            <a
              key={`${h.video_id}-${h.t}-${i}`}
              href={yt(h)}
              target="_blank"
              rel="noreferrer"
              className="group relative aspect-video rounded-md overflow-hidden bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800"
              title={`${h.video_id} @ ${mmss(h.t)} · score ${h.score.toFixed(3)}`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={h.frame} alt="" className="w-full h-full object-cover" loading="lazy" />
              <span className="absolute bottom-1 left-1 rounded bg-black/70 px-1 text-[10px] leading-tight text-white">
                #{i + 1} · {mmss(h.t)}
              </span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

const PIN_LAST = "side profile of man talking about real estate";

export default function MarengoVsGooglePage() {
  const { queries, has_google } = data as { queries: Row[]; has_google: boolean };
  const rows = [...queries].sort(
    (a, b) => (a.query === PIN_LAST ? 1 : 0) - (b.query === PIN_LAST ? 1 : 0)
  );
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      <main className="mx-auto max-w-5xl px-4 py-12">
        <div className="mb-8">
          <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100">
            ← back to search
          </Link>
        </div>

        <header className="mb-12 text-center">
          <h1 className="text-3xl font-semibold tracking-tight">
            <a
              href="https://www.twelvelabs.io/blog/marengo-3-0"
              target="_blank"
              rel="noreferrer"
              className="underline decoration-dotted underline-offset-4 hover:opacity-80"
            >
              Marengo 3
            </a>{" "}
            vs Google Gemini Embedding 2
          </h1>
          <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400 max-w-2xl mx-auto leading-relaxed">
            Four golden searches run against 8 hours of content with two native multimodal
            (video&nbsp;+&nbsp;audio) retrieval pipelines, Twelve&nbsp;Labs&nbsp;Marengo&nbsp;3
            and Google&apos;s Gemini&nbsp;Embedding&nbsp;2. The top five moments each one returned
            are shown below; click a frame to open that moment on YouTube.
          </p>
          {!has_google && (
            <p className="mt-3 text-xs text-amber-600 dark:text-amber-500">
              Google column pending labels &mdash; will populate once embeddings + selections are in.
            </p>
          )}
        </header>

        <div className="space-y-14">
          {rows.map((row) => (
            <section key={row.query}>
              <h2 className="text-base font-medium text-zinc-700 dark:text-zinc-300 mb-5">
                <span className="font-bold">Query:</span> &ldquo;{row.query}&rdquo;
              </h2>
              <div className="space-y-6">
                <Column label="Marengo 3" accent="#50c878" hits={row.marengo} />
                <Column label="Google Gemini Embedding 2" accent="#4a90e2" hits={row.google} />
              </div>
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}
