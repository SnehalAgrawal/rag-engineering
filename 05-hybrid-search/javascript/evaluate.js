// Vector, keyword, and hybrid retrieval on the same chunks and the same budget. Same behaviour and output
// as ../python/evaluate.py, where the steps are described. Timings (the last section) vary from run to run.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as chunkers from "./chunkers.js";
import * as llm from "./llm.js";
import * as search from "./search.js";
import { Tokenizer } from "./tokens.js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const DATA = join(ROOT, "..", "data");
const SETS = { new: "questions.json", "04": "questions-04.json" };
const SET_NAMES = Object.keys(SETS);

function load() {
  const settings = JSON.parse(readFileSync(join(DATA, "settings.json"), "utf8"));
  const documents = {};
  for (const name of readdirSync(join(DATA, "documents")).filter((n) => n.endsWith(".md")).sort()) {
    documents[name] = readFileSync(join(DATA, "documents", name), "utf8");
  }
  const questions = Object.fromEntries(Object.entries(SETS).map(
    ([name, file]) => [name, JSON.parse(readFileSync(join(DATA, file), "utf8")).questions]));
  return { settings, documents, questions };
}

const found = (question, texts) => question.evidence.some(
  (alternative) => alternative.every((p) => texts.some((t) => t.includes(p.text))));

// Chunks in rank order; one that doesn't fit what is left of the budget is skipped.
function withinBudget(ranked, tokens, budget) {
  const chosen = [];
  let used = 0;
  for (const i of ranked) {
    if (used + tokens[i] <= budget) {
      chosen.push(i);
      used += tokens[i];
    }
  }
  return chosen;
}

async function embedAll(texts, batchSize) {
  const vectors = [];
  for (let i = 0; i < texts.length; i += batchSize) vectors.push(...(await llm.embed(texts.slice(i, i + batchSize))).vectors);
  return vectors;
}

const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

class Run {
  async init(settings, documents, questions, tokenizer) {
    Object.assign(this, { settings, questions, timing: {} });
    this.chunks = await chunkers.chunk(settings.chunking, documents, tokenizer);
    this.texts = this.chunks.map((c) => c.text);
    this.tokens = this.chunks.map((c) => c.tokens);

    let started = performance.now();
    const prefix = settings.embedding.document_prefix;
    this.vectors = await embedAll(this.texts.map((t) => prefix + t), settings.embedding.batch_size);
    this.timing["embed chunks (s)"] = (performance.now() - started) / 1000;

    const { bm25 } = settings;
    this.indexes = {};
    for (const analyzer of Object.keys(bm25.analyzers)) {
      started = performance.now();
      this.indexes[analyzer] = new search.BM25(this.texts, analyzer, bm25.k1, bm25.b);
      this.timing[`build BM25 ${analyzer} (ms)`] = performance.now() - started;
    }

    // Questions are embedded one at a time, as a live system would, so the time per query is measured.
    this.scores = {};
    const querySeconds = [];
    const keywordSeconds = [];
    for (const q of Object.values(questions).flat()) {
      started = performance.now();
      const query = (await llm.embed([settings.embedding.query_prefix + q.question])).vectors[0];
      querySeconds.push((performance.now() - started) / 1000);
      const entry = { vector: this.vectors.map((v) => chunkers.cosine(query, v)) };
      for (const [analyzer, index] of Object.entries(this.indexes)) {
        started = performance.now();
        entry[analyzer] = index.scores(q.question);
        keywordSeconds.push((performance.now() - started) / 1000);
      }
      this.scores[q.id] = entry;
    }
    this.timing["vector query: embed the question (ms, median)"] = median(querySeconds) * 1000;
    this.timing["keyword query: BM25 over all chunks (ms, median)"] = median(keywordSeconds) * 1000;
    return this;
  }

  ranking(qid, method) {
    const s = this.scores[qid];
    if (method === "vector") return search.rank(s.vector);
    if (method.startsWith("keyword-")) return search.rank(s[method.slice("keyword-".length)], true);
    if (method === "rrf") {
      const lists = [search.rank(s.vector), search.rank(s[this.analyzer], true)];
      return search.rank(search.rrf(lists, this.settings.rrf.k, this.chunks.length));
    }
    if (method.startsWith("weighted-")) {
      const alpha = Number(method.slice("weighted-".length));
      return search.rank(search.weighted(s.vector, s[this.analyzer], alpha), true);
    }
    throw new Error(method);
  }

  foundAt(q, method, budget) {
    const chosen = withinBudget(this.ranking(q.id, method), this.tokens, budget);
    return found(q, chosen.map((i) => this.texts[i]));
  }

  count(method, budget, setName, where = {}) {
    return this.questions[setName].filter((q) => Object.entries(where).every(([f, v]) => q[f] === v))
      .filter((q) => this.foundAt(q, method, budget)).length;
  }

  devCount(method) {
    return SET_NAMES.reduce((s, name) => s + this.count(method, this.settings.primary_budget, name, { split: "dev" }), 0);
  }

  // The two choices made on development questions only (rules in data/settings.json).
  choose() {
    const words = this.devCount("keyword-words");
    const identifiers = this.devCount("keyword-identifiers");
    this.analyzer = identifiers > words ? "identifiers" : "words";
    this.sweep = Object.fromEntries(this.settings.weighted.alphas.map(
      (a) => [a.toFixed(1), this.devCount(`weighted-${a.toFixed(1)}`)]));
    const best = Math.max(...Object.values(this.sweep));
    this.alpha = Object.keys(this.sweep).filter((a) => this.sweep[a] === best)
      .sort((a, b) => Math.abs(Number(a) - 0.5) - Math.abs(Number(b) - 0.5) || Number(a) - Number(b))[0];
  }
}

const r = (value, width) => String(value).padStart(width);
const pyList = (items) => `[${items.map((x) => `'${x}'`).join(", ")}]`;

function report(run) {
  const s = run.settings;
  const budget = s.primary_budget;
  const n = Object.fromEntries(Object.entries(run.questions).map(([name, qs]) => [name, {
    dev: qs.filter((q) => q.split === "dev").length, test: qs.filter((q) => q.split === "test").length,
  }]));
  const methods = ["vector", "keyword-words", "keyword-identifiers", "rrf", `weighted-${run.alpha}`];
  const width = Math.max(...methods.map((m) => m.length)) + 2;
  const l = (value) => String(value).padEnd(width);

  console.log("1. How the analyzers turn identifiers into tokens");
  for (const example of ["SSO-104", "cursor_expired", "X-RateLimit-Remaining", "https://api.example.com/scim/v2"]) {
    console.log(`   ${example.padEnd(34)}words: ${pyList(search.analyze(example, "words"))}`);
    console.log(`   ${"".padEnd(34)}identifiers: ${pyList(search.analyze(example, "identifiers"))}`);
  }
  console.log(`\n   ${run.chunks.length} chunks (${s.chunking.name}), ${run.tokens.reduce((a, b) => a + b, 0)} tokens. BM25 index: `
    + Object.entries(run.indexes).map(([a, i]) => `${a} ${i.idf.size} distinct tokens`).join(", "));

  console.log(`\n2. Choices made on the development questions (${n.new.dev} new + ${n["04"].dev} from 04), at ${budget} tokens`);
  console.log(`   keyword analyzer: words ${run.devCount("keyword-words")}, identifiers `
    + `${run.devCount("keyword-identifiers")} -> hybrids use '${run.analyzer}'`);
  console.log("   weighted fusion, alpha = weight of the vector score:");
  console.log(`   alpha ${Object.keys(run.sweep).map((a) => r(a, 5)).join("")}`);
  console.log(`   found ${Object.values(run.sweep).map((c) => r(c, 5)).join("")}   -> alpha ${run.alpha}`);

  const splits = ["new", "04"].flatMap((name) => ["dev", "test"].map((split) => [name, split]));
  console.log(`\n3. Evidence found at ${budget} tokens`);
  console.log(l("method") + splits.map(([name, split]) => r(`${name} ${split} (${n[name][split]})`, 14)).join(""));
  for (const m of methods) {
    console.log(l(m) + splits.map(([name, split]) => r(run.count(m, budget, name, { split }), 14)).join(""));
  }

  const categories = [...new Set(run.questions.new.map((q) => q.category))];
  const test = run.questions.new.filter((q) => q.split === "test");
  console.log(`\n4. New test questions by kind, at ${budget} tokens`);
  console.log(l("method") + categories.map((c) => r(`${c} (${test.filter((q) => q.category === c).length})`, 16)).join(""));
  for (const m of methods) {
    console.log(l(m) + categories.map((c) => r(run.count(m, budget, "new", { split: "test", category: c }), 16)).join(""));
  }

  const total = SET_NAMES.reduce((a, name) => a + n[name].test, 0);
  console.log(`\n5. All test questions (${total}), by budget`);
  console.log(l("method") + s.budgets.map((b) => r(`${b} tokens`, 13)).join(""));
  for (const m of methods) {
    console.log(l(m) + s.budgets.map((b) => r(SET_NAMES.reduce((a, name) => a + run.count(m, b, name, { split: "test" }), 0), 13)).join(""));
  }

  const keyword = `keyword-${run.analyzer}`;
  const every = SET_NAMES.flatMap((name) => run.questions[name].map((q) => [name, q]));
  const line = ([name, q]) => `     ${q.id} [${name} ${q.split}, ${q.category}] ${q.question}`;
  console.log(`\n6. Questions only one method found, at ${budget} tokens (all ${every.length} questions)`);
  for (const [a, b] of [["vector", keyword], [keyword, "vector"]]) {
    const only = every.filter(([, q]) => run.foundAt(q, a, budget) && !run.foundAt(q, b, budget));
    console.log(`   ${a} only (${only.length}):`);
    only.forEach((x) => console.log(line(x)));
  }
  for (const m of ["rrf", `weighted-${run.alpha}`]) {
    const lost = every.filter(([, q]) => !run.foundAt(q, m, budget)
      && (run.foundAt(q, "vector", budget) || run.foundAt(q, keyword, budget)));
    console.log(`   found by vector or ${keyword}, but not by ${m} (${lost.length}):`);
    lost.forEach((x) => console.log(line(x)));
  }
  const neither = every.filter(([, q]) => !methods.some((m) => run.foundAt(q, m, budget)));
  console.log(`   found by no method (${neither.length}):`);
  neither.forEach((x) => console.log(line(x)));

  console.log("\n7. Timings (vary from run to run)");
  for (const [key, value] of Object.entries(run.timing)) console.log(`   ${key}: ${value.toFixed(2)}`);
}

function save(run) {
  const budget = run.settings.primary_budget;
  const methods = ["vector", "keyword-words", "keyword-identifiers", "rrf", `weighted-${run.alpha}`];
  const out = { analyzer: run.analyzer, alpha: run.alpha, sweep: run.sweep, questions: {} };
  for (const qs of Object.values(run.questions)) {
    for (const q of qs) {
      out.questions[q.id] = Object.fromEntries(methods.map((m) => [m, {
        ranked: run.ranking(q.id, m).slice(0, 10).map((i) => run.chunks[i].id),
        found: Object.fromEntries(run.settings.budgets.map((b) => [String(b), run.foundAt(q, m, b)])),
        context: withinBudget(run.ranking(q.id, m), run.tokens, budget).map((i) => run.chunks[i].id),
      }]));
    }
  }
  writeFileSync(join(ROOT, "results-retrieval.json"), `${JSON.stringify(out, null, 1)}\n`);
}

const { settings, documents, questions } = load();
let run;
try {
  run = await new Run().init(settings, documents, questions, new Tokenizer());
} finally {
  await llm.unload(llm.embeddingModel());
}
run.choose();
save(run);
report(run);
