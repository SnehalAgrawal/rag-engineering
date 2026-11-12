// Evaluates the assistant on the questions in ../data/questions.json. Same behaviour and output as
// ../python/evaluate.py.
//   node evaluate.js                    retrieval metrics, then all four modes
//   node evaluate.js --retrieval-only   only the retrieval metrics (no chat model needed)
//   node evaluate.js --regrade FILE     re-grade saved replies without calling any model
// Part 1 measures retrieval alone: for each answerable question, does an article that answers it
// appear in the top K? (Recall@K; every question is labeled with the article(s) that answer it, so
// it is the hit rate.) Part 2 sends every question through the application four ways (see MODES).
// Answerable and not-in-docs questions are scored separately.
import { readFileSync, writeFileSync } from "node:fs";
import { chatModel, embeddingModel, unload } from "./llm.js";

const QUESTIONS_FILE = new URL("../data/questions.json", import.meta.url);
const MODES = [
  { name: "all, default context", mode: "all", numCtx: null }, // the whole help center; Ollama picks the context length
  { name: "all, 8K context", mode: "all", numCtx: 8192 }, // the whole help center, with room for all of it
  { name: "rag, top 1", mode: "rag", k: 1 }, // only the best-matching article
  { name: "rag, top 3", mode: "rag", k: 3 }, // only the 3 best-matching articles
];

// Python-style repr of a string or a list of strings, so both languages print the same text.
function pyRepr(value) {
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(", ")}]`;
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  const escaped = value.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
  return quote + (quote === "'" ? escaped.replace(/'/g, "\\'") : escaped) + quote;
}

const pad = (value, width) => String(value).padStart(width);
const padEnd = (value, width) => String(value).padEnd(width);
const center = (text, width) => {
  const total = Math.max(0, width - text.length), left = Math.floor(total / 2);
  return " ".repeat(left) + text + " ".repeat(total - left);
};

// What the customer saw: the key fact, an answer without it, or the decline message.
function grade(record, question) {
  const shown = record.log.outcome === "answered";
  if (record.kind === "answerable") {
    if (!shown) return "declined";
    return question.patterns.some((p) => new RegExp(p, "i").test(record.message)) ? "key fact" : "no key fact";
  }
  return shown ? "answered" : "declined";
}

function short(text, width = 110) {
  text = text.split(/\s+/).filter(Boolean).join(" ");
  return text.length <= width ? text : text.slice(0, width - 1) + "…";
}

const count = (list, value) => list.filter((x) => x === value).length;
const medianLow = (values) => [...values].sort((a, b) => a - b)[Math.floor((values.length - 1) / 2)];

function report(records, questions) {
  console.log("\n2. The whole application, four ways\n");
  console.log(padEnd("", 22) + center("answerable questions", 36) + center("not in docs", 16));
  console.log(padEnd("mode", 22) + pad("key fact", 12) + pad("no key fact", 13) + pad("declined", 11) + pad("declined", 14) + pad("prompt tokens", 16));
  for (const mode of MODES) {
    const rows = records.filter((r) => r.mode_name === mode.name);
    const grades = rows.map((r) => [r, grade(r, questions[r.question_id])]);
    const answerable = grades.filter(([r]) => r.kind === "answerable").map(([, g]) => g);
    const unanswerable = grades.filter(([r]) => r.kind === "not_in_docs").map(([, g]) => g);
    const tokens = medianLow(rows.filter((r) => r.log.prompt_tokens !== null).map((r) => r.log.prompt_tokens));
    console.log(padEnd(mode.name, 22) + pad(count(answerable, "key fact"), 9) + "/" + padEnd(answerable.length, 2) +
      pad(count(answerable, "no key fact"), 13) + pad(count(answerable, "declined"), 11) +
      pad(count(unanswerable, "declined"), 11) + "/" + padEnd(unanswerable.length, 2) + pad(tokens.toLocaleString("en-US"), 16));
  }
  console.log("\nkey fact: the reply shown to the customer contains the question's key fact (a pattern check).");
  console.log("no key fact: an answer was shown, but without the key fact.");
  console.log("declined: the customer got the decline message (the model declined, or a check in code rejected its answer).");
  console.log("not in docs: the help center doesn't answer these, so declining is the right behavior.");
  console.log("prompt tokens: median per question (the lower middle value), as counted by the model server.");
  console.log("A pattern can't tell a right answer from a wrong one that mentions the same fact: read the replies too.");

  console.log("\nSpeed in this run (latencies vary between runs and machines):\n");
  console.log(padEnd("mode", 22) + pad("reused from cache", 19) + pad("first request", 15) + pad("median", 10));
  for (const mode of MODES) {
    const rows = records.filter((r) => r.mode_name === mode.name);
    const cached = rows.map((r) => r.log.cached_tokens).filter((c) => c !== null && c !== undefined);
    const latencies = rows.map((r) => r.log.latency_ms);
    const reused = cached.length ? medianLow(cached).toLocaleString("en-US") : "n/a";
    console.log(padEnd(mode.name, 22) + pad(reused, 19) + pad((latencies[0] / 1000).toFixed(2), 13) + " s" +
      pad((medianLow(latencies) / 1000).toFixed(2), 8) + " s");
  }
  console.log("\nreused from cache: median prompt tokens the server reused from earlier requests (prompt_eval_cached_count).");
  console.log("first request: each mode starts with the chat model unloaded, so this includes loading it.");

  console.log("\nEvery reply without the key fact, and every answer to a question the help center doesn't answer:");
  for (const mode of MODES) {
    for (const r of records.filter((r) => r.mode_name === mode.name)) {
      const g = grade(r, questions[r.question_id]);
      if (g === "key fact" || (r.kind === "not_in_docs" && g === "declined")) continue;
      const where = r.log.retrieved ? ` retrieved ${pyRepr(r.log.retrieved)}` : "";
      console.log(`  [${mode.name}] ${r.question_id} ${g}: ${pyRepr(r.question)}${where}`);
      console.log(`      outcome ${r.log.outcome}; shown: ${short(r.message)}`);
      if (r.log.outcome !== "answered" && r.model_output) console.log(`      model output: ${short(r.model_output)}`);
    }
  }
}

const args = process.argv.slice(2);
const data = JSON.parse(readFileSync(QUESTIONS_FILE, "utf8"));
const questions = {};
for (const q of data.answerable) questions[q.id] = { ...q, kind: "answerable" };
for (const q of data.not_in_docs) questions[q.id] = { ...q, kind: "not_in_docs" };

if (args[0] === "--regrade") {
  const records = readFileSync(args[1], "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
  report(records, questions);
  process.exit(0);
}

const { ARTICLES, answer, retrieve } = await import("./rag.js"); // imported here so --regrade needs no index

await unload(chatModel());
await unload(embeddingModel());
console.log(`Chat model: ${chatModel()} | embedding model: ${embeddingModel()} | ${ARTICLES.length} articles | ` +
  `${data.answerable.length} answerable questions, ${data.not_in_docs.length} not in docs\n`);

// Part 1: retrieval alone.
console.log("1. Retrieval: does an article that answers the question appear in the top K?\n");
const ranks = [], topScores = {};
const answerableIds = new Set(data.answerable.map((a) => a.id));
for (const q of [...data.answerable, ...data.not_in_docs]) {
  const results = await retrieve(q.question, 5);
  topScores[q.id] = results[0].score;
  if (answerableIds.has(q.id)) {
    const index = results.findIndex((r) => q.relevant.includes(r.id));
    const rank = index === -1 ? null : index + 1;
    ranks.push(rank);
    if (rank !== 1) {
      const top3 = results.slice(0, 3).map((r) => `${r.id} ${r.score.toFixed(4)}`).join(", ");
      console.log(`  ${q.id} ${pyRepr(q.question)}: relevant ${pyRepr(q.relevant)}, rank ${rank ?? ">5"}; top 3: ${top3}`);
    }
  }
}
for (const k of [1, 3, 5]) {
  const hits = ranks.filter((r) => r !== null && r <= k).length;
  console.log(`  Recall@${k} = ${hits}/${ranks.length} = ${(hits / ranks.length).toFixed(2)}`);
}
const answerableTop = data.answerable.map((q) => topScores[q.id]);
const unanswerableTop = data.not_in_docs.map((q) => topScores[q.id]);
console.log(`  Top-1 similarity: answerable ${Math.min(...answerableTop).toFixed(4)} to ${Math.max(...answerableTop).toFixed(4)}, ` +
  `not in docs ${Math.min(...unanswerableTop).toFixed(4)} to ${Math.max(...unanswerableTop).toFixed(4)}`);
if (args[0] === "--retrieval-only") {
  await unload(embeddingModel());
  process.exit(0);
}

// Part 2: the whole application, four ways. Each mode starts from an unloaded chat model.
const records = [];
for (const mode of MODES) {
  await unload(chatModel());
  for (const q of [...data.answerable, ...data.not_in_docs]) {
    const result = await answer(q.question, { mode: mode.mode, k: mode.k ?? 3, numCtx: mode.numCtx ?? null });
    records.push({
      mode_name: mode.name, question_id: q.id, question: q.question, kind: questions[q.id].kind, message: result.message,
      sources: result.sources, retrieved: result.retrieved, model_output: result.model_output, log: result.log,
    });
    process.stderr.write(".");
  }
}
await unload(chatModel());
await unload(embeddingModel());
process.stderr.write("\n");

const out = `results-${chatModel().replace(":", "_")}.jsonl`;
writeFileSync(out, records.map((r) => JSON.stringify(r)).join("\n") + "\n");
report(records, questions);
console.log(`\nEvery reply, with its log record, was saved to ${out}.`);
