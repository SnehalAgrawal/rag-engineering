// What the traces alone tell you: the operator's view, without the test questions or the hand labels.
// Same behaviour and output as ../python/report.py.
//   node report.js                      reads the results files that exist in this folder
//   node report.js FILE [FILE ...]
// Each line of a results file holds one request's trace. The report answers: what happened to the
// requests, which call failed and how, where the time went, which index and prompt versions answered,
// and how close the answered questions came to the relevance threshold.
import { existsSync, readFileSync } from "node:fs";
import { pyRepr } from "./text.js";

const DEFAULT_FILES = ["results-llama3.2_3b.jsonl", "results-faults.jsonl", "results-update.jsonl"];
const NEAR_THRESHOLD = 0.02; // an arbitrary review band above the threshold, not a tuned value

const medianLow = (values) => [...values].sort((a, b) => a - b)[Math.floor((values.length - 1) / 2)];
const seconds = (ms) => (ms / 1000).toFixed(2).padStart(7) + " s";
const tally = (values) => {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
  return counts;
};

const args = process.argv.slice(2);
const files = args.length ? args : DEFAULT_FILES.filter((f) => existsSync(f));
const traces = files.flatMap((name) => readFileSync(name, "utf8").trim().split("\n").map((line) => JSON.parse(line).trace));
const profiles = [...new Set(traces.map((t) => t.profile))].sort();
console.log(`${traces.length} traces from ${files.join(", ")}\n`);

console.log("1. Outcomes\n");
console.log("  " + "outcome".padEnd(22) + profiles.map((p) => p.padStart(12)).join(""));
const counts = Object.fromEntries(profiles.map((p) => [p, tally(traces.filter((t) => t.profile === p).map((t) => t.outcome))]));
for (const outcome of [...new Set(traces.map((t) => t.outcome))].sort()) {
  console.log("  " + outcome.padEnd(22) + profiles.map((p) => String(counts[p].get(outcome) || 0).padStart(12)).join(""));
}
console.log("  " + "(chat calls made)".padEnd(22) +
  profiles.map((p) => String(traces.filter((t) => t.profile === p && "generate" in t.stages).length).padStart(12)).join(""));

console.log("\n2. Failed calls\n");
const failures = new Map();
for (const t of traces.filter((x) => x.error)) {
  const key = [t.profile, t.error.stage, t.error.kind].join("\u0000");
  const entry = failures.get(key) || { n: 0, slowest: 0 };
  failures.set(key, { n: entry.n + 1, slowest: Math.max(entry.slowest, t.latency_ms) });
}
for (const key of [...failures.keys()].sort()) {
  const [profile, stage, kind] = key.split("\u0000");
  const { n, slowest } = failures.get(key);
  console.log(`  ${profile.padEnd(12)}${(stage + ": " + kind).padEnd(28)}requests: ${n}, longest wait ${(slowest / 1000).toFixed(1)} s`);
}

console.log("\n3. Where the time went (answered requests)\n");
console.log("  " + "profile".padEnd(12) + "stage".padEnd(10) + "median".padStart(9) + "max".padStart(9));
for (const profile of profiles) {
  const answered = traces.filter((t) => t.profile === profile && t.outcome === "answered");
  for (const stage of ["retrieve", "generate"]) {
    const ms = answered.map((t) => t.stages[stage]);
    console.log("  " + profile.padEnd(12) + stage.padEnd(10) + seconds(medianLow(ms)) + seconds(Math.max(...ms)));
  }
  const ms = answered.map((t) => t.latency_ms);
  console.log("  " + profile.padEnd(12) + "total".padEnd(10) + seconds(medianLow(ms)) + seconds(Math.max(...ms)));
}

console.log("\n4. What answered: versions\n");
for (const key of ["prompt_version", "chat_model", "embedding_model", "index_version"]) {
  console.log("  " + key.padEnd(16) + [...tally(traces.map((t) => t[key]))].map(([value, n]) => `${value} (${n})`).join(", "));
}
const stale = traces.filter((t) => t.index_stale_articles);
console.log(`  ${stale.length} requests were served from an index built before articles changed ` +
  `(up to ${Math.max(0, ...stale.map((t) => t.index_stale_articles))} changed).`);

console.log("\n5. The relevance gate\n");
const gated = traces.filter((t) => t.gate && t.relevance_threshold !== null);
const blocked = gated.filter((t) => !t.gate.passed);
console.log(`  ${blocked.length} of ${gated.length} requests blocked; top scores of the blocked: ` +
  blocked.map((t) => t.gate.top_score.toFixed(4)).join(", "));
const near = gated.filter((t) => t.outcome === "answered" && t.gate.top_score < t.relevance_threshold + NEAR_THRESHOLD);
console.log(`  answered with a top score within ${NEAR_THRESHOLD} of the threshold: ${near.length}`);
for (const t of near) {
  console.log(`    request ${t.request_id.slice(0, 8)}: top score ${t.gate.top_score.toFixed(4)}, ` +
    `retrieved ${pyRepr(t.retrieved.map((r) => r.id))}`);
}
console.log("\n  A trace shows what the model saw and what the pipeline decided. It doesn't show whether an answer");
console.log("  was right: that takes labeled questions and reading replies (the evaluate script).");
