// The answer stage: does better retrieval give better answers? Same behaviour and output as
// ../python/answer.py, where the method and the hand labels are described.
// Run with --regrade to print the table again from the saved replies without calling the models.
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import * as chunkers from "./chunkers.js";
import * as evaluate from "./evaluate.js";
import * as llm from "./llm.js";
import { Tokenizer } from "./tokens.js";

const LABELS = ["correct", "partly", "wrong"];

const replyHash = (answer) => createHash("sha256").update(answer, "utf8").digest("hex").slice(0, 16);

function strategiesToAnswer(settings, questions) {
  const saved = JSON.parse(readFileSync(join(evaluate.ROOT, "results-retrieval.json"), "utf8"));
  const order = evaluate.ranking(saved, settings, questions);
  const { baseline, best } = settings.answer;
  return [baseline, ...order.filter((n) => n !== baseline).slice(0, best)];
}

async function run(settings, documents, questions, tokenizer) {
  const prompt = JSON.parse(readFileSync(join(evaluate.DATA, "prompt.json"), "utf8"));
  const test = questions.filter((q) => q.split === "test");
  const byName = Object.fromEntries(settings.strategies.map((s) => [s.name, s]));
  const queryVectors = await new evaluate.Embedder(settings.embedding.batch_size)
    .embed(test.map((q) => settings.embedding.query_prefix + q.question));
  const records = [];
  for (const name of strategiesToAnswer(settings, questions)) {
    const { chunks, vectors } = await evaluate.buildIndex(byName[name], documents, tokenizer, settings);
    const tokens = chunks.map((c) => c.tokens);
    for (const [qi, question] of test.entries()) {
      const ranked = evaluate.rank(vectors.map((v) => chunkers.cosine(queryVectors[qi], v)));
      const chosen = evaluate.withinBudget(ranked, tokens, settings.answer.budget);
      const excerpts = chosen.map((i, n) => `--- excerpt ${n + 1} (${chunks[i].document})\n${chunks[i].text}`).join("\n\n");
      const messages = [
        { role: "system", content: prompt.system.replace("{excerpts}", excerpts) },
        { role: "user", content: question.question },
      ];
      const reply = await llm.chat(messages, { schema: prompt.schema, maxTokens: settings.answer.max_output_tokens });
      const parsed = JSON.parse(reply.content);
      records.push({
        strategy: name, question_id: question.id, category: question.category, question: question.question,
        chunks: chosen.map((i) => chunks[i].id), context_tokens: chosen.reduce((s, i) => s + tokens[i], 0),
        evidence_found: evaluate.found(question, chosen.map((i) => chunks[i].text)),
        answered: parsed.answered, answer: parsed.answer, prompt_version: prompt.version,
        prompt_tokens: reply.promptTokens, output_tokens: reply.outputTokens, done_reason: reply.doneReason,
      });
    }
  }
  return records;
}

function label(record, labels) {
  if (!record.answered) return "declined";
  return labels[`${record.question_id} | ${replyHash(record.answer)}`] ?? null;
}

const r = (value, width) => String(value).padStart(width);
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function report(records, questions) {
  const { labels } = JSON.parse(readFileSync(join(evaluate.DATA, "hand_labels.json"), "utf8"));
  const names = [...new Set(records.map((x) => x.strategy))];
  const width = Math.max(...names.map((n) => n.length)) + 2;
  const l = (value) => String(value).padEnd(width);
  console.log(`Answers to the ${new Set(records.map((x) => x.question_id)).size} test questions (hand labels)`);
  console.log(`${l("strategy")}${r("correct", 8)}${r("partly", 8)}${r("wrong", 7)}${r("declined", 10)}${r("unlabeled", 11)}`
    + `${r("evidence found", 16)}${r("context tokens", 16)}${r("prompt tokens", 15)}`);
  for (const name of names) {
    const rows = records.filter((x) => x.strategy === name);
    const got = rows.map((x) => label(x, labels));
    const n = (k) => got.filter((g) => g === k).length;
    console.log(`${l(name)}${r(n("correct"), 8)}${r(n("partly"), 8)}${r(n("wrong"), 7)}${r(n("declined"), 10)}`
      + `${r(n(null), 11)}${r(rows.filter((x) => x.evidence_found).length, 16)}`
      + `${r(`median ${median(rows.map((x) => x.context_tokens))}`, 16)}${r(`median ${median(rows.map((x) => x.prompt_tokens))}`, 15)}`);
  }

  console.log("\nBy kind of question: correct / questions");
  const categories = [...new Set(questions.map((q) => q.category))];
  const test = questions.filter((q) => q.split === "test");
  console.log(l("strategy") + categories.map((c) => r(`${c} (${test.filter((q) => q.category === c).length})`, 14)).join(""));
  for (const name of names) {
    const rows = records.filter((x) => x.strategy === name);
    console.log(l(name) + categories.map(
      (c) => r(rows.filter((x) => x.category === c && label(x, labels) === "correct").length, 14)).join(""));
  }

  console.log("\nEvidence found in the context, and the answer");
  for (const name of names) {
    const rows = records.filter((x) => x.strategy === name);
    for (const isFound of [true, false]) {
      const subset = rows.filter((x) => x.evidence_found === isFound).map((x) => label(x, labels));
      if (subset.length) {
        const counts = [...LABELS, "declined"].map((k) => [k, subset.filter((s) => s === k).length])
          .filter(([, c]) => c).map(([k, c]) => `${c} ${k}`).join(", ");
        console.log(`  ${l(name)}evidence ${isFound ? "found" : "missing"} (${subset.length}): ${counts}`);
      }
    }
  }

  const unlabeled = records.filter((x) => label(x, labels) === null);
  if (unlabeled.length) {
    console.log(`\nUnlabeled replies (${unlabeled.length}), to read and add to data/hand_labels.json:`);
    for (const x of unlabeled) {
      console.log(`  "${x.question_id} | ${replyHash(x.answer)}"  [${x.strategy}] ${x.question}`);
      console.log(`      ${x.answer}`);
    }
  }
}

async function main() {
  const { settings, documents, questions } = evaluate.load();
  const path = join(evaluate.ROOT, "results-answers.jsonl");
  if (!process.argv.includes("--regrade")) {
    const tokenizer = new Tokenizer();
    let records;
    try {
      records = await run(settings, documents, questions, tokenizer);
    } finally {
      await llm.unload(llm.chatModel());
      await llm.unload(llm.embeddingModel());
    }
    writeFileSync(path, records.map((x) => `${JSON.stringify(x)}\n`).join(""));
  }
  const records = readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
  report(records, questions);
}

await main();
