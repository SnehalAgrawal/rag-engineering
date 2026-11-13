// The index lifecycle: what happens between "an article changed" and "the assistant answers with it"?
// Same behaviour and output as ../python/update.py.
//   node update.js
// Works on a copy of the help center and its own index in a temporary folder; ../data is not touched.
//   1. Build the index, then build it again with nothing changed.
//   2. Billing changes five articles (the same changes as in 02-rag-vs-fine-tuning).
//   3. Ask the five affected test questions before re-indexing: the assistant still serves the old version,
//      and every trace says how many articles changed since it was built.
//   4. Re-index (only the changed articles are embedded) and ask again. The running assistant loads the new
//      version on its next request.
//   5. An article is removed: re-indexing drops it without embedding anything.
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, update } from "./index.js";
import * as store from "./index_store.js";
import { chatModel, embeddingModel, unload } from "./llm.js";
import { Assistant } from "./pipeline.js";
import { pyRepr } from "./text.js";

const data = (name) => fileURLToPath(new URL(`../data/${name}`, import.meta.url));
// [article id, old text, new text]
const CHANGES = [
  ["plans", "The Pro plan costs $25 per month", "The Pro plan costs $30 per month"],
  ["annual-billing", "$250 per year for Pro", "$300 per year for Pro"],
  ["password", "The reset link expires after 1 hour.", "The reset link expires after 2 hours."],
  ["invite-members", "expire after 7 days", "expire after 14 days"],
  ["invite-members", "up to 5 members", "up to 10 members"],
  ["change-plan", "no more than 5 remain", "no more than 10 remain"],
];
const QUESTIONS = ["a01", "a02", "a11", "a21", "a24"]; // the test questions about the changed facts
const REMOVED = "sso";

function edit(path, change) {
  const content = JSON.parse(readFileSync(path, "utf8"));
  change(content.articles);
  writeFileSync(path, JSON.stringify(content, null, 2) + "\n");
}

function applyChanges(articles) {
  const byId = Object.fromEntries(articles.map((a) => [a.id, a]));
  for (const [id, oldText, newText] of CHANGES) {
    if (!byId[id].text.includes(oldText)) throw new Error(`${id}: ${pyRepr(oldText)} not found`);
    byId[id].text = byId[id].text.replace(oldText, () => newText);
  }
}

const questions = Object.fromEntries(JSON.parse(readFileSync(data("questions.json"), "utf8")).answerable.map((q) => [q.id, q.question]));
const records = [];
const folder = mkdtempSync(join(tmpdir(), "production-rag-"));
try {
  const helpCenter = join(folder, "help_center.json"), indexPath = join(folder, "index.json");
  copyFileSync(data("help_center.json"), helpCenter);
  const reindex = async (step) => {
    const [index, stats] = await update(helpCenter, indexPath);
    console.log(`${step}: ${describe(index, stats)}`);
  };
  let assistant;
  const ask = async (step) => {
    for (const id of QUESTIONS) {
      const result = await assistant.answer(questions[id]);
      const { trace } = result;
      console.log(`  ${id} [index ${trace.index_version}, ${trace.index_stale_articles} stale] ${pyRepr(questions[id])}\n      ${result.message}`);
      records.push({ step, question_id: id, question: questions[id], message: result.message, trace });
    }
  };

  await unload(embeddingModel());
  await unload(chatModel());
  console.log("1. Build the index");
  await reindex("   from scratch");
  await reindex("   again, nothing changed");
  assistant = new Assistant("production", { indexPath, helpCenterPath: helpCenter });

  edit(helpCenter, applyChanges);
  const stale = store.staleness(store.read(indexPath), store.readArticles(helpCenter));
  console.log(`\n2. Billing changes ${new Set(CHANGES.map((c) => c[0])).size} articles (${CHANGES.length} edits). ` +
    `Articles changed since the index was built: ${stale}`);
  console.log("\n3. Before re-indexing, the running assistant still serves the old version:");
  await ask("before re-indexing");

  console.log();
  await reindex("4. Re-index");
  console.log("   The same assistant, on its next requests:");
  await ask("after re-indexing");

  edit(helpCenter, (articles) => articles.splice(articles.findIndex((a) => a.id === REMOVED), 1));
  console.log();
  await reindex(`5. The '${REMOVED}' article is removed. Re-index`);
  console.log(`   entries in the index: ${store.read(indexPath).entries.length}`);
} finally {
  rmSync(folder, { recursive: true, force: true });
}
await unload(chatModel());
await unload(embeddingModel());

writeFileSync("results-update.jsonl", records.map((r) => JSON.stringify(r)).join("\n") + "\n");
console.log("\nEvery reply, with its trace, was saved to results-update.jsonl.");
