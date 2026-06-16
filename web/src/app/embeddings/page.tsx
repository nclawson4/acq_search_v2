// Embedding comparison — minimal view.
// Same 6 queries against the same database. Two image-search models surface
// their top results side by side. The reader looks at the pictures.
import Link from "next/link";
import data from "@/data/embedding-comparison.json";

export const metadata = {
  title: "Embedding comparison",
};

type Hit = { video_id: string; scene_idx: number };
type Row = {
  query: string;
  clip:     { hits: Hit[] };
  gem_3072: { hits: Hit[] };
};

function frameUrl(h: Hit) {
  return `/api/frames/${h.video_id}/${String(h.scene_idx).padStart(4, "0")}`;
}

function Column({ label, hits }: { label: string; hits: Hit[] }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.18em] text-zinc-500 mb-3 font-semibold">
        {label}
      </p>
      <div className="grid grid-cols-5 gap-2">
        {hits.slice(0, 5).map((h, i) => (
          <div
            key={`${h.video_id}-${h.scene_idx}-${i}`}
            className="aspect-video rounded-md overflow-hidden bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={frameUrl(h)}
              alt=""
              className="w-full h-full object-cover"
              loading="lazy"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function EmbeddingsPage() {
  const rows = (data as { queries: Row[] }).queries;
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      <main className="mx-auto max-w-5xl px-4 py-12">
        <div className="mb-8">
          <Link
            href="/"
            className="text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
          >
            ← back to search
          </Link>
        </div>

        <header className="mb-12 text-center">
          <h1 className="text-3xl font-semibold tracking-tight">CLIP vs Gemini Embedding 2</h1>
          <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400 max-w-2xl mx-auto leading-relaxed">
            Six real searches an editor might run. The same database was searched with two image
            embedding models &mdash; OpenAI&apos;s CLIP (ViT-L-14) and Google&apos;s Gemini
            Embedding 2 at 3072 dimensions. The top five scenes each one returned are shown below.
          </p>
        </header>

        <div className="space-y-14">
          {rows.map((row) => (
            <section key={row.query}>
              <h2 className="text-base font-medium text-zinc-700 dark:text-zinc-300 mb-5">
                &ldquo;{row.query}&rdquo;
              </h2>
              <div className="space-y-6">
                <Column label="CLIP (ViT-L-14)" hits={row.clip.hits} />
                <Column label="Gemini Embedding 2 (3072 dim)" hits={row.gem_3072.hits} />
              </div>
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}
