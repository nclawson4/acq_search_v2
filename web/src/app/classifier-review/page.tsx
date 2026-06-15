// Deprecated — kept in repo as reference, no longer routed to from the UI.
// Originally compared the model's visual classifier output against the user's
// 50-frame ground truth. The shipping classifier passed its 90% gate; this
// page is no longer maintained.
import Link from "next/link";

export default function ClassifierReviewDeprecated() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-white dark:bg-zinc-950 text-zinc-700 dark:text-zinc-300 p-8">
      <div className="max-w-md text-center space-y-4">
        <h1 className="text-2xl font-semibold">This page is deprecated</h1>
        <p className="text-sm text-zinc-500 leading-relaxed">
          The classifier-review dashboard was used during development to inspect where the
          visual classifier disagreed with hand-labeled validation frames. The classifier has
          since passed its accuracy gate and this dashboard is no longer maintained.
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
