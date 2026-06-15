// Deprecated — replaced by the shipping visual classifier whose evaluation
// passed its 90% accuracy gate. The labeling workflow is no longer required.
import Link from "next/link";

export default function LabelV2Deprecated() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-white dark:bg-zinc-950 text-zinc-700 dark:text-zinc-300 p-8">
      <div className="max-w-md text-center space-y-4">
        <h1 className="text-2xl font-semibold">This page is deprecated</h1>
        <p className="text-sm text-zinc-500 leading-relaxed">
          The visual labeling workflow was used to validate the classifier during development.
          The classifier is now shipping and this labeling page is no longer maintained.
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
