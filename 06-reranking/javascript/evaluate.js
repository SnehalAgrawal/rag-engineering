// Reranking: does a second pass that reads the question and each candidate together choose better than
// the first stage's order? Same behaviour and output as ../python/evaluate.py, where the steps are described.
// Pairs are scored once each, in the same fixed order as in Python. Timings (the last section) vary
// from run to run.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as firstStage from "./first_stage.js";
import * as llm from "./llm.js";
import * as rerank from "./rerank.js";
import { Tokenizer } from "./tokens.js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const STAGES = ["vector", "rrf"];

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

class Experiment {
  async init(settings, documents, questions, tokenizer) {
    Object.assign(this, { settings, questions });
    this.every = Object.entries(questions).flatMap(([name, qs]) => qs.map((q) => [name, q]));
    this.byId = Object.fromEntries(this.every.map(([, q]) => [q.id, q]));
    const { chunks, ranked } = await firstStage.rankings(settings, documents, this.every.map(([, q]) => q), tokenizer);
    Object.assign(this, { chunks, ranked, texts: chunks.map((c) => c.text), tokens: chunks.map((c) => c.tokens) });
    return this;
  }

  async scoreAll() {
    const prompt = rerank.loadPrompt();
    const r = this.settings.reranker;
    this.promptVersion = prompt.version;
    this.scores = new Map();
    this.records = [];
    for (const [qid, i] of firstStage.candidatePairs(this.ranked, Math.max(...this.settings.candidates))) {
      const started = performance.now();
      const result = await rerank.judge(prompt, this.byId[qid].question, this.texts[i], r);
      const ms = performance.now() - started;
      this.scores.set(`${qid}|${i}`, result.score);
      this.records.push({
        question_id: qid, chunk: this.chunks[i].id, score: result.score, token: result.token,
        prompt_tokens: result.promptTokens, ms: Math.round(ms * 10) / 10,
      });
    }
  }

  order(qid, stage, n, reranked) {
    const top = this.ranked[qid][stage].slice(0, n);
    if (!reranked) return top;
    const position = new Map(top.map((i, p) => [i, p]));
    return [...top].sort((a, b) => this.scores.get(`${qid}|${b}`) - this.scores.get(`${qid}|${a}`)
      || position.get(a) - position.get(b));
  }

  foundAt(q, order, budget) {
    const chosen = budget === null ? order : withinBudget(order, this.tokens, budget);
    return found(q, chosen.map((i) => this.texts[i]));
  }

  count(rows, stage, n, reranked, budget) {
    return rows.filter(([, q]) => this.foundAt(q, this.order(q.id, stage, n, reranked), budget)).length;
  }

  rows(setName = null, split = null, category = null) {
    return this.every.filter(([name, q]) => (setName === null || name === setName)
      && (split === null || q.split === split) && (category === null || q.category === category));
  }
}

const r = (value, width) => String(value).padStart(width);
const l = (value, width) => String(value).padEnd(width);
const pyList = (items) => `[${items.map((v) => (v === null ? "None" : String(v))).join(", ")}]`;
const round3 = (x) => Math.round(x * 1000) / 1000;

function report(x) {
  const s = x.settings;
  const groups = [["new", "dev"], ["new", "test"], ["04", "dev"], ["04", "test"]];
  const sizes = groups.map((g) => x.rows(...g).length);
  const header = groups.map(([a, b], k) => r(`${a} ${b} (${sizes[k]})`, 14)).join("");
  const width = 30;
  const label = (stage, n, reranked) => `${stage}, top ${n}, ${reranked ? "reranked" : "first stage"}`;

  console.log("1. Candidate recall: the evidence is somewhere in the first stage's top N (no budget)");
  console.log(`${l("first stage", width)}${header}`);
  for (const stage of STAGES) {
    for (const n of s.candidates) {
      console.log(l(`${stage}, top ${n}`, width) + groups.map((g) => r(x.count(x.rows(...g), stage, n, false, null), 14)).join(""));
    }
  }
  console.log(l("(all 98 chunks)", width) + sizes.map((n) => r(n, 14)).join(""));

  const test = x.rows(null, "test");
  console.log(`\n2. Evidence found within each budget, all test questions (${test.length})`);
  console.log(l("first stage, candidates, order", width) + s.budgets.map((b) => r(`${b} tokens`, 13)).join(""));
  for (const stage of STAGES) {
    const full = x.chunks.length;
    console.log(l(`${stage}, all, first stage`, width) + s.budgets.map((b) => r(x.count(test, stage, full, false, b), 13)).join(""));
    for (const n of s.candidates) {
      for (const reranked of [false, true]) {
        console.log(l(label(stage, n, reranked), width) + s.budgets.map((b) => r(x.count(test, stage, n, reranked, b), 13)).join(""));
      }
    }
  }

  const budget = s.primary_budget;
  console.log(`\n3. Evidence found at ${budget} tokens, by question group`);
  console.log(`${l("first stage, candidates, order", width)}${header}`);
  for (const stage of STAGES) {
    for (const n of s.candidates) {
      for (const reranked of [false, true]) {
        console.log(l(label(stage, n, reranked), width) + groups.map((g) => r(x.count(x.rows(...g), stage, n, reranked, budget), 14)).join(""));
      }
    }
  }

  const categories = [...new Set(x.questions.new.map((q) => q.category))];
  const newTest = Object.fromEntries(categories.map((c) => [c, x.rows("new", "test", c)]));
  console.log(`\n4. New test questions by kind, at ${budget} tokens`);
  console.log(l("first stage, candidates, order", width) + categories.map((c) => r(`${c} (${newTest[c].length})`, 16)).join(""));
  for (const stage of STAGES) {
    for (const n of s.candidates) {
      for (const reranked of [false, true]) {
        console.log(l(label(stage, n, reranked), width) + categories.map((c) => r(x.count(newTest[c], stage, n, reranked, budget), 16)).join(""));
      }
    }
  }

  console.log(`\n5. What reranking changed, at 500 and ${budget} tokens (all ${x.every.length} questions)`);
  for (const stage of STAGES) {
    for (const n of s.candidates) {
      for (const b of [500, budget]) {
        const after = (q) => x.foundAt(q, x.order(q.id, stage, n, true), b);
        const before = (q) => x.foundAt(q, x.order(q.id, stage, n, false), b);
        const fixed = x.every.map(([, q]) => q).filter((q) => after(q) && !before(q));
        const broken = x.every.map(([, q]) => q).filter((q) => !after(q) && before(q));
        console.log(`   ${stage}, top ${n}, ${b} tokens: fixed ${fixed.length}, broke ${broken.length}`);
        for (const [name, qs] of [["fixed", fixed], ["broke", broken]]) {
          for (const q of qs) console.log(`     ${name} ${q.id} [${q.split}, ${q.category}] ${q.question}`);
        }
      }
    }
  }

  console.log("\n6. The three cases from 05: where the chunk with the evidence ranks");
  for (const qid of ["r20", "r29", "r28"]) {
    const q = x.byId[qid];
    const holders = x.texts.flatMap((t, i) => (q.evidence.some((alt) => alt.every((p) => t.includes(p.text))) ? [i] : []));
    console.log(`   ${qid} ${q.question}`);
    for (const stage of STAGES) {
      const first = holders.map((i) => x.ranked[qid][stage].indexOf(i) + 1);
      const n = Math.max(...s.candidates);
      const order = x.order(qid, stage, n, true);
      const after = holders.map((i) => (x.ranked[qid][stage].slice(0, n).includes(i) ? order.indexOf(i) + 1 : null));
      const scores = holders.map((i) => (x.scores.has(`${qid}|${i}`) ? round3(x.scores.get(`${qid}|${i}`)) : null));
      console.log(`     ${stage}: first-stage rank ${pyList(first)}, rank after reranking the top ${n} ${pyList(after)}, score ${pyList(scores)}`);
    }
  }

  const ms = x.records.map((rec) => rec.ms).sort((a, b) => a - b);
  const median = ms[Math.floor(ms.length / 2)];
  console.log("\n7. Cost (timings vary from run to run)");
  console.log(`   ${x.records.length} pairs scored, one call each: median ${median.toFixed(0)} ms per call, total `
    + `${(ms.reduce((a, b) => a + b, 0) / 1000).toFixed(0)} s`);
  for (const n of s.candidates) {
    console.log(`   reranking the top ${n} for one question: ${n} calls, about ${(n * median / 1000).toFixed(1)} s one after another`);
  }
}

function save(x) {
  writeFileSync(join(ROOT, "results-scores.jsonl"), x.records.map((rec) => `${JSON.stringify(rec)}\n`).join(""));
  const out = { prompt_version: x.promptVersion, questions: {} };
  for (const [, q] of x.every) {
    const entry = {};
    for (const stage of STAGES) {
      for (const n of x.settings.candidates) {
        for (const rr of [false, true]) {
          entry[`${stage}-${n}-${rr ? "reranked" : "first"}`] = x.order(q.id, stage, n, rr).map((i) => x.chunks[i].id);
        }
      }
    }
    out.questions[q.id] = entry;
  }
  writeFileSync(join(ROOT, "results-rerank.json"), `${JSON.stringify(out, null, 1)}\n`);
}

const { settings, documents, questions } = firstStage.load();
const x = await new Experiment().init(settings, documents, questions, new Tokenizer());
await llm.unload(llm.embeddingModel());
try {
  await x.scoreAll();
} finally {
  await llm.unload(llm.chatModel());
}
save(x);
report(x);
