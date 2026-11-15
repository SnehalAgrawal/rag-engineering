// Retrieval for every chunking strategy in data/settings.json, scored against the evidence in
// data/questions.json. Same behaviour and output as ../python/evaluate.py, where the method is described.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as chunkers from "./chunkers.js";
import * as llm from "./llm.js";
import { Tokenizer } from "./tokens.js";

export const ROOT = dirname(fileURLToPath(import.meta.url));
export const DATA = join(ROOT, "..", "data");

export function load() {
  const settings = JSON.parse(readFileSync(join(DATA, "settings.json"), "utf8"));
  const documents = {};
  for (const name of readdirSync(join(DATA, "documents")).filter((n) => n.endsWith(".md")).sort()) {
    documents[name] = readFileSync(join(DATA, "documents", name), "utf8");
  }
  const { questions } = JSON.parse(readFileSync(join(DATA, "questions.json"), "utf8"));
  return { settings, documents, questions };
}

// Embeds in batches and keeps count of what it cost.
export class Embedder {
  constructor(batchSize) {
    Object.assign(this, { batchSize, texts: 0, tokens: 0, calls: 0, seconds: 0 });
  }

  async embed(texts, truncate = false) {
    const vectors = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const started = performance.now();
      const result = await llm.embed(batch, { truncate });
      this.seconds += (performance.now() - started) / 1000;
      this.texts += batch.length;
      this.tokens += result.tokens;
      this.calls += 1;
      vectors.push(...result.vectors);
    }
    return vectors;
  }

  cost() {
    return { texts: this.texts, tokens: this.tokens, calls: this.calls, seconds: Math.round(this.seconds * 100) / 100 };
  }
}

export const found = (question, texts) => question.evidence.some(
  (alternative) => alternative.every((p) => texts.some((t) => t.includes(p.text))));

// Chunks in rank order; one that doesn't fit what is left of the budget is skipped.
export function withinBudget(ranked, tokens, budget) {
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

// How many characters of a too-long text the embedding model actually reads with truncate: true:
// the longest prefix, ending at a line break, that it accepts with truncate: false (a binary search).
async function embeddedLength(text, prefix) {
  const breaks = [...text].flatMap((ch, i) => (ch === "\n" ? [i] : []));
  let low = 0, high = breaks.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high + 1) / 2);
    try {
      await llm.embed([prefix + text.slice(0, breaks[middle])]);
      low = middle;
    } catch (error) {
      if (!(error instanceof llm.EmbeddingError)) throw error;
      high = middle - 1;
    }
  }
  return breaks[low];
}

export async function buildIndex(strategy, documents, tokenizer, settings) {
  const prefix = settings.embedding.document_prefix;
  const splitCost = new Embedder(settings.embedding.batch_size);
  const chunks = await chunkers.chunk(strategy, documents, tokenizer,
    (texts) => splitCost.embed(texts.map((t) => prefix + t)));
  const indexCost = new Embedder(settings.embedding.batch_size);
  const notes = [];
  const cut = {};
  let vectors = [];
  if (strategy.kind === "whole") {
    // One document per call, so a document that doesn't fit is identified. It is then embedded the way
    // Ollama does by default (truncate: true): only its beginning.
    for (const chunk of chunks) {
      try {
        vectors.push(...await indexCost.embed([prefix + chunk.text]));
      } catch (error) {
        if (!(error instanceof llm.EmbeddingError)) throw error;
        const seen = await embeddedLength(chunk.text, prefix);
        notes.push(`${chunk.document}: ${error.message}; embedded with truncate=true, which reads about the first `
          + `${tokenizer.count(chunk.text.slice(0, seen))} of ${chunk.tokens} tokens`);
        cut[chunk.document] = seen;
        vectors.push(...await indexCost.embed([prefix + chunk.text], true));
      }
    }
  } else {
    vectors = await indexCost.embed(chunks.map((c) => prefix + c.text));
  }
  return { chunks, vectors, splitCost: splitCost.cost(), indexCost: indexCost.cost(), notes, cut };
}

export const rank = (scores) => scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);
const round4 = (x) => Math.round(x * 10000) / 10000;

async function evaluate(settings, documents, questions, tokenizer) {
  const queryVectors = await new Embedder(settings.embedding.batch_size)
    .embed(questions.map((q) => settings.embedding.query_prefix + q.question));
  const results = {};
  for (const strategy of settings.strategies) {
    const index = await buildIndex(strategy, documents, tokenizer, settings);
    const { chunks, vectors } = index;
    const tokens = chunks.map((c) => c.tokens);
    const perQuestion = {};
    questions.forEach((question, qi) => {
      const scores = vectors.map((v) => chunkers.cosine(queryVectors[qi], v));
      const ranked = rank(scores);
      const entry = {
        ranked: ranked.slice(0, 10).map((i) => chunks[i].id), scores: ranked.slice(0, 10).map((i) => round4(scores[i])),
        budget: {}, top_k: {},
      };
      for (const budget of settings.budgets) {
        const chosen = withinBudget(ranked, tokens, budget);
        entry.budget[String(budget)] = {
          found: found(question, chosen.map((i) => chunks[i].text)), chunks: chosen.length,
          tokens: chosen.reduce((s, i) => s + tokens[i], 0),
        };
      }
      for (const k of settings.recall_k) {
        const top = ranked.slice(0, k);
        entry.top_k[String(k)] = {
          found: found(question, top.map((i) => chunks[i].text)), tokens: top.reduce((s, i) => s + tokens[i], 0),
        };
      }
      perQuestion[question.id] = entry;
    });
    const allTexts = chunks.map((c) => c.text);
    const hasCut = Object.keys(index.cut).length > 0;
    const unseen = hasCut ? questions.filter((q) => !found(q, Object.keys(documents).map(
      (d) => (d in index.cut ? documents[d].slice(0, index.cut[d]) : documents[d])))).map((q) => q.id) : [];
    const sorted = [...tokens].sort((a, b) => a - b);
    results[strategy.name] = {
      chunks: chunks.length, indexed_tokens: tokens.reduce((s, t) => s + t, 0),
      intact: questions.filter((q) => found(q, allTexts)).map((q) => q.id),
      chunk_tokens: { min: sorted[0], median: sorted[Math.floor(sorted.length / 2)], max: sorted[sorted.length - 1] },
      split_cost: index.splitCost, index_cost: index.indexCost, notes: index.notes, unseen_by_embedding: unseen,
      questions: perQuestion,
      chunk_list: chunks.map((c) => ({
        id: c.id, document: c.document, start: c.start, end: c.end, prefix: c.prefix, tokens: c.tokens,
        sha256: createHash("sha256").update(c.text).digest("hex").slice(0, 16),
      })),
    };
  }
  return results;
}

function countFound(result, questions, mode, key, where = {}) {
  return questions.filter((q) => Object.entries(where).every(([f, v]) => q[f] === v))
    .filter((q) => result.questions[q.id][mode][key].found).length;
}

// The selection rule: most development questions found at the primary budget; ties: fewer indexed tokens.
export function ranking(results, settings, questions) {
  const budget = String(settings.primary_budget);
  const dev = (name) => countFound(results[name], questions, "budget", budget, { split: "dev" });
  return Object.keys(results).sort((a, b) => dev(b) - dev(a) || results[a].indexed_tokens - results[b].indexed_tokens);
}

function meanTokens(result, questions, k) {
  return Math.round(questions.reduce((s, q) => s + result.questions[q.id].top_k[k].tokens, 0) / questions.length);
}

const r = (value, width) => String(value).padStart(width);

function report(results, settings, questions) {
  const names = settings.strategies.map((s) => s.name);
  const width = Math.max(...names.map((n) => n.length)) + 2;
  const l = (value) => String(value).padEnd(width);
  console.log("1. The chunks, and what indexing cost (embedding tokens are nomic-embed-text tokens)");
  console.log(`${l("strategy")}${r("chunks", 7)}${r("tokens: min", 13)}${r("median", 8)}${r("max", 6)}${r("indexed", 9)}`
    + `${r("intact", 8)}${r("embedded texts", 16)}${r("embedding tokens", 18)}`);
  for (const name of names) {
    const x = results[name];
    const texts = x.index_cost.texts + x.split_cost.texts;
    const embedded = x.index_cost.tokens + x.split_cost.tokens;
    const extra = x.split_cost.texts ? `  (incl. ${x.split_cost.texts} sentence windows to find the splits)` : "";
    const intact = `${x.intact.length}/${questions.length}`;
    console.log(`${l(name)}${r(x.chunks, 7)}${r(x.chunk_tokens.min, 13)}${r(x.chunk_tokens.median, 8)}`
      + `${r(x.chunk_tokens.max, 6)}${r(x.indexed_tokens, 9)}${r(intact, 8)}${r(texts, 16)}${r(embedded, 18)}${extra}`);
    for (const note of x.notes) console.log(`${l("")}  ${note}`);
    if (x.unseen_by_embedding.length) {
      console.log(`${l("")}  questions whose evidence the embedding never read: ${x.unseen_by_embedding.length}`);
    }
  }

  for (const split of ["dev", "test"]) {
    const chosen = questions.filter((q) => q.split === split);
    console.log(`\n2. Evidence found, ${split} questions (${chosen.length}): within a context budget, and in the top k chunks`);
    let header = settings.budgets.map((b) => r(`${b} tokens`, 12)).join("");
    header += settings.recall_k.map((k) => `${r(`top ${k}`, 8)}${r("(tokens)", 9)}`).join("");
    console.log(`${l("strategy")}${header}`);
    for (const name of names) {
      const x = results[name];
      let row = settings.budgets.map((b) => r(countFound(x, chosen, "budget", String(b)), 12)).join("");
      row += settings.recall_k.map((k) => `${r(countFound(x, chosen, "top_k", String(k)), 8)}`
        + `${r(`(${meanTokens(x, chosen, String(k))})`, 9)}`).join("");
      console.log(`${l(name)}${row}`);
    }
  }

  const budget = String(settings.primary_budget);
  const categories = [...new Set(questions.map((q) => q.category))];
  console.log(`\n3. Evidence found at ${budget} tokens, by kind of question (dev + test)`);
  console.log(l("strategy") + categories.map(
    (c) => r(`${c} (${questions.filter((q) => q.category === c).length})`, 14)).join(""));
  for (const name of names) {
    console.log(l(name) + categories.map(
      (c) => r(countFound(results[name], questions, "budget", budget, { category: c }), 14)).join(""));
  }

  const order = ranking(results, settings, questions);
  const answer = settings.answer;
  const best = order.filter((n) => n !== answer.baseline).slice(0, answer.best);
  console.log(`\n4. Selection on the dev questions at ${budget} tokens: `
    + order.map((n) => `${n} ${countFound(results[n], questions, "budget", budget, { split: "dev" })}`).join(", "));
  console.log(`   Answer stage: ${answer.baseline} (baseline), ${best.join(", ")}`);
}

async function main() {
  const { settings, documents, questions } = load();
  const tokenizer = new Tokenizer();
  let results;
  try {
    results = await evaluate(settings, documents, questions, tokenizer);
  } finally {
    await llm.unload(llm.embeddingModel());
  }
  const lines = [];
  for (const [name, x] of Object.entries(results)) {
    for (const c of x.chunk_list) lines.push(JSON.stringify({ strategy: name, ...c }));
  }
  writeFileSync(join(ROOT, "results-chunks.jsonl"), `${lines.join("\n")}\n`);
  const saved = Object.fromEntries(Object.entries(results).map(([name, x]) => {
    const { chunk_list: _, ...rest } = x;
    return [name, rest];
  }));
  writeFileSync(join(ROOT, "results-retrieval.json"), `${JSON.stringify(saved, null, 1)}\n`);
  report(results, settings, questions);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
