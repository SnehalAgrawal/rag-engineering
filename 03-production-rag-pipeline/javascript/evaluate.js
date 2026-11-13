// Evaluates the pipeline. Same behaviour and output as ../python/evaluate.py.
//   node evaluate.js                    parts 1-3
//   node evaluate.js --retrieval-only   parts 1-2 (no chat model needed)
//   node evaluate.js --regrade FILE     part 3's report from saved replies, without calling any model
// Part 1: retrieval on the 30 test questions (Recall@K, as in 01-basic-rag).
// Part 2: the relevance threshold, chosen on the development questions only. The rule: the highest
//         threshold, rounded down to 0.01, that blocks at most 1% of the answerable dev questions.
// Part 3: the 30 test questions through both profiles (basic = 01-basic-rag's behavior; production =
//         deadline and relevance gate). Replies are graded by hand labels (../data/hand_labels.json),
//         each stored with a hash of the exact reply; a reply without a label is reported as unlabeled.
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { chatModel, embeddingModel, unload } from "./llm.js";
import { pyRepr } from "./text.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../data/${name}`, import.meta.url), "utf8"));
const TEST = load("questions.json");
const DEV = load("dev_questions.json");
const SETTINGS = load("settings.json");
const LABELS = load("hand_labels.json").labels;
const PROFILES = ["basic", "production"];
const SWEEP = [0.55, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70];
const MAX_BLOCKED_SHARE = 0.01;

const pad = (value, width) => String(value).padStart(width);
const padEnd = (value, width) => String(value).padEnd(width);
const center = (text, width) => {
  const total = Math.max(0, width - text.length), left = Math.floor(total / 2);
  return " ".repeat(left) + text + " ".repeat(total - left);
};
const count = (list, value) => list.filter((x) => x === value).length;
const medianLow = (values) => [...values].sort((a, b) => a - b)[Math.floor((values.length - 1) / 2)];
const replyHash = (message) => createHash("sha256").update(message, "utf8").digest("hex").slice(0, 16);

// What the customer got. Declines need no label; answers to answerable questions are labeled by hand.
function label(record) {
  if (record.trace.outcome !== "answered") return "declined";
  if (record.kind === "not_in_docs") return "answered";
  return LABELS[`${record.question_id} | ${replyHash(record.message)}`] ?? null;
}

function short(text, width = 110) {
  text = text.split(/\s+/).filter(Boolean).join(" ");
  return text.length <= width ? text : text.slice(0, width - 1) + "…";
}

async function topScores(assistant, questions) {
  const results = {};
  for (const q of questions) results[q.id] = await assistant.retrieve(q.question, 5, { timeout: 60 });
  return results;
}

async function part1(assistant) {
  console.log("1. Retrieval on the test questions: does an article that answers the question appear in the top K?\n");
  const results = await topScores(assistant, [...TEST.answerable, ...TEST.not_in_docs]);
  const ranks = [];
  for (const q of TEST.answerable) {
    const ids = results[q.id].map((r) => r.id);
    const index = ids.findIndex((id) => q.relevant.includes(id));
    const rank = index === -1 ? null : index + 1;
    ranks.push(rank);
    if (rank !== 1) {
      const top3 = results[q.id].slice(0, 3).map((r) => `${r.id} ${r.score.toFixed(4)}`).join(", ");
      console.log(`  ${q.id} ${pyRepr(q.question)}: relevant ${pyRepr(q.relevant)}, rank ${rank ?? ">5"}; top 3: ${top3}`);
    }
  }
  for (const k of [1, 3, 5]) {
    const hits = ranks.filter((r) => r !== null && r <= k).length;
    console.log(`  Recall@${k} = ${hits}/${ranks.length} = ${(hits / ranks.length).toFixed(2)}`);
  }
  const answerable = TEST.answerable.map((q) => results[q.id][0].score);
  const unanswerable = TEST.not_in_docs.map((q) => results[q.id][0].score);
  console.log(`  Top-1 similarity: answerable ${Math.min(...answerable).toFixed(4)} to ${Math.max(...answerable).toFixed(4)}, ` +
    `not in docs ${Math.min(...unanswerable).toFixed(4)} to ${Math.max(...unanswerable).toFixed(4)}`);
}

async function part2(assistant) {
  console.log("\n2. The relevance threshold, chosen on the development questions\n");
  const top = async (questions) => Object.fromEntries(Object.entries(await topScores(assistant, questions)).map(([id, r]) => [id, r[0].score]));
  const answerable = await top(DEV.answerable), unanswerable = await top(DEV.not_in_docs);
  const a = Object.values(answerable), n = Object.values(unanswerable);
  console.log(`  ${a.length} answerable dev questions, top-1 similarity ${Math.min(...a).toFixed(4)} to ${Math.max(...a).toFixed(4)}`);
  console.log(`  ${n.length} not-in-docs dev questions, top-1 similarity ${Math.min(...n).toFixed(4)} to ${Math.max(...n).toFixed(4)}\n`);
  console.log(`  ${pad("threshold", 9)}${pad("answerable blocked", 21)}${pad("not in docs caught", 21)}`);
  for (const t of SWEEP) {
    const blocked = a.filter((s) => s < t).length, caught = n.filter((s) => s < t).length;
    console.log(`  ${pad(t.toFixed(2), 9)}${pad(`${blocked}/${a.length}`, 21)}${pad(`${caught}/${n.length}`, 21)}`);
  }
  const allowed = Math.floor(MAX_BLOCKED_SHARE * a.length);
  const ceiling = [...a].sort((x, y) => x - y)[allowed]; // any threshold up to this blocks at most `allowed`
  const rule = Math.floor(ceiling * 100) / 100;
  const configured = SETTINGS.production.relevance_threshold;
  console.log(`\n  Rule: block at most ${Math.round(MAX_BLOCKED_SHARE * 100)}% of answerable dev questions (${allowed}), rounded down ` +
    `to 0.01 -> ${rule.toFixed(2)}. settings.json uses ${pyRepr(configured)}.`);
  if (configured !== rule) console.log("  WARNING: settings.json differs from the rule.");
  if (configured !== null) {
    const names = Object.fromEntries([...DEV.answerable, ...DEV.not_in_docs].map((q) => [q.id, q.question]));
    const byScore = (scores) => Object.entries(scores).sort((x, y) => x[1] - y[1]);
    for (const [id, score] of byScore(answerable)) {
      if (score < configured) console.log(`  blocked (answerable): ${id} ${pyRepr(names[id])} ${score.toFixed(4)}`);
    }
    for (const [id, score] of byScore(unanswerable)) {
      if (score >= configured) console.log(`  passed (not in docs): ${id} ${pyRepr(names[id])} ${score.toFixed(4)}`);
    }
  }
}

function report(records) {
  console.log("\n3. The test questions through both profiles\n");
  console.log(padEnd("", 12) + center("answerable (hand labels)", 44) + center("not in docs", 20) + pad("chat", 6));
  console.log(padEnd("profile", 12) + pad("correct", 9) + pad("partly", 8) + pad("wrong", 7) + pad("declined", 10) +
    pad("unlabeled", 10) + pad("declined", 11) + pad("answered", 9) + pad("calls", 6));
  for (const profile of PROFILES) {
    const rows = records.filter((r) => r.profile === profile);
    const a = rows.filter((r) => r.kind === "answerable").map(label);
    const n = rows.filter((r) => r.kind === "not_in_docs").map(label);
    const calls = rows.filter((r) => "generate" in r.trace.stages).length;
    console.log(padEnd(profile, 12) + pad(count(a, "correct"), 9) + pad(count(a, "partly"), 8) + pad(count(a, "wrong"), 7) +
      pad(count(a, "declined"), 10) + pad(count(a, null), 10) + pad(count(n, "declined"), 8) + "/" + padEnd(n.length, 2) +
      pad(count(n, "answered"), 9) + pad(calls, 6));
  }
  console.log("\ncorrect: the key fact is right and nothing in the reply is false. partly: the key fact is right, but the");
  console.log("reply also says something false. wrong: answered without the right key fact. declined: the customer got");
  console.log("the decline message (the gate, the model, or a check in code). chat calls: requests that reached the chat model.");

  const basic = Object.fromEntries(records.filter((r) => r.profile === "basic").map((r) => [r.question_id, r]));
  const production = records.filter((r) => r.profile === "production");
  const passed = production.filter((r) => r.trace.gate && r.trace.gate.passed);
  const same = passed.filter((r) => r.message === basic[r.question_id].message && r.model_output === basic[r.question_id].model_output).length;
  console.log(`\nWhere the gate let a question through (${passed.length} of ${production.length}), the production reply was ` +
    `identical to the basic one for ${same} of ${passed.length}.`);

  console.log("\nLatency in this run (it varies between runs and machines):\n");
  console.log(padEnd("profile", 12) + pad("first request", 15) + pad("median", 10) + pad("median retrieve", 17) + pad("median generate", 17));
  for (const profile of PROFILES) {
    const rows = records.filter((r) => r.profile === profile);
    const latencies = rows.map((r) => r.trace.latency_ms);
    const stage = (name) => rows.filter((r) => name in r.trace.stages).map((r) => r.trace.stages[name]);
    console.log(padEnd(profile, 12) + pad((latencies[0] / 1000).toFixed(2), 13) + " s" + pad((medianLow(latencies) / 1000).toFixed(2), 8) + " s" +
      pad((medianLow(stage("retrieve")) / 1000).toFixed(2), 15) + " s" + pad((medianLow(stage("generate")) / 1000).toFixed(2), 15) + " s");
  }
  console.log("first request: each profile starts with both models unloaded, so this includes loading them.");

  console.log("\nBlocked by the gate (production):");
  for (const r of production) {
    const gate = r.trace.gate;
    if (gate && !gate.passed) {
      const kind = r.kind === "answerable" ? "answerable" : "not in docs";
      console.log(`  ${r.question_id} (${kind}) top score ${gate.top_score.toFixed(4)}: ${pyRepr(r.question)}`);
    }
  }
  console.log("\nEvery reply that isn't a correct answer or a correct decline:");
  for (const profile of PROFILES) {
    for (const r of records.filter((x) => x.profile === profile)) {
      const g = label(r);
      if (g === "correct" || (r.kind === "not_in_docs" && g === "declined")) continue;
      const top = r.trace.gate ? r.trace.gate.top_score : null;
      console.log(`  [${profile}] ${r.question_id} ${g ?? "UNLABELED"} (${r.trace.outcome}, top score ${pyRepr(top)}): ` +
        `${pyRepr(r.question)} retrieved ${pyRepr(r.trace.retrieved.map((x) => x.id))}`);
      console.log(`      shown: ${short(r.message)}`);
      if (g === null) console.log(`      label key: ${r.question_id} | ${replyHash(r.message)}`);
    }
  }
}

const { values: args } = parseArgs({ options: { "retrieval-only": { type: "boolean" }, regrade: { type: "string" } } });

if (args.regrade) {
  report(readFileSync(args.regrade, "utf8").trim().split("\n").map((line) => JSON.parse(line)));
} else {
  const { Assistant } = await import("./pipeline.js"); // imported here so --regrade needs no index
  await unload(chatModel());
  await unload(embeddingModel());
  let assistant = new Assistant("basic");
  console.log(`Chat model: ${chatModel()} | embedding model: ${embeddingModel()} | index ${assistant.index.version} ` +
    `(${assistant.index.entries.length} articles) | ${TEST.answerable.length} answerable test questions, ` +
    `${TEST.not_in_docs.length} not in docs\n`);
  await part1(assistant);
  await part2(assistant);
  await unload(embeddingModel());
  if (!args["retrieval-only"]) {
    const questions = [...TEST.answerable.map((q) => [q, "answerable"]), ...TEST.not_in_docs.map((q) => [q, "not_in_docs"])];
    const records = [];
    for (const profile of PROFILES) {
      await unload(chatModel()); // each profile starts from unloaded models
      await unload(embeddingModel());
      assistant = new Assistant(profile);
      for (const [q, kind] of questions) {
        const result = await assistant.answer(q.question);
        records.push({
          profile, question_id: q.id, question: q.question, kind, message: result.message, sources: result.sources,
          model_output: result.model_output, trace: result.trace,
        });
        process.stderr.write(".");
      }
    }
    await unload(chatModel());
    await unload(embeddingModel());
    process.stderr.write("\n");

    const out = `results-${chatModel().replace(":", "_")}.jsonl`;
    writeFileSync(out, records.map((r) => JSON.stringify(r)).join("\n") + "\n");
    report(records);
    console.log(`\nEvery reply, with its trace, was saved to ${out}.`);
  }
}
