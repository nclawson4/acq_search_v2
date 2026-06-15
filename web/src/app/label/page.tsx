// Deprecated — original 4-tag labeling page (format / shot_type / who / has_text).
// The visual classification schema was simplified to is_animation + talking_head_pose
// and the labeling moved to /label-v2 (also now deprecated).
import Link from "next/link";

export default function LabelDeprecated() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-white dark:bg-zinc-950 text-zinc-700 dark:text-zinc-300 p-8">
      <div className="max-w-md text-center space-y-4">
        <h1 className="text-2xl font-semibold">This page is deprecated</h1>
        <p className="text-sm text-zinc-500 leading-relaxed">
          The original 4-tag labeling page was used during early classifier work. The visual
          classification schema has since been simplified and validated; this page is no longer
          maintained.
        </p>
        <Link
          href="/"
          className="inline-block rounded-full bg-blue-600 text-white px-5 py-2 text-sm font-medium hover:bg-blue-700"
        >
          ← back to search
        </Link>
      </div>
    </div>
  );
}
