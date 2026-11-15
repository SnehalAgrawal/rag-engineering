// The first stage, as in 05-hybrid-search. Same behaviour as ../python/first_stage.py.
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as chunkers from "./chunkers.js";
import * as llm from "./llm.js";
import * as search from "./search.js";

const DATA = join(dirname(fileURLToPath(import.meta.url)), "..", "data");
const SETS = { new: "questions.json", "04": "questions-04.json" };

export function load() {
  const settings = JSON.parse(readFileSync(join(DATA, "settings.json"), "utf8"));
  const documents = {};
  for (const name of readdirSync(join(DATA, "documents")).filter((n) => n.endsWith(".md")).sort()) {
    documents[name] = readFileSync(join(DATA, "documents", name), "utf8");
  }
  const questions = Object.fromEntries(Object.entries(SETS).map(
    ([name, file]) => [name, JSON.parse(readFileSync(join(DATA, file), "utf8")).questions]));
  return { settings, documents, questions };
}

async function embedAll(texts, batchSize) {
  const vectors = [];
  for (let i = 0; i < texts.length; i += batchSize) vectors.push(...(await llm.embed(texts.slice(i, i + batchSize))).vectors);
  return vectors;
}

// The chunks, and for each question id: { vector: [chunk indexes], rrf: [chunk indexes] }, best first.
export async function rankings(settings, documents, questions, tokenizer) {
  const chunks = await chunkers.chunk(settings.chunking, documents, tokenizer);
  const texts = chunks.map((c) => c.text);
  const e = settings.embedding;
  const vectors = await embedAll(texts.map((t) => e.document_prefix + t), e.batch_size);
  const queries = await embedAll(questions.map((q) => e.query_prefix + q.question), e.batch_size);
  const bm25 = new search.BM25(texts, settings.bm25.analyzer, settings.bm25.k1, settings.bm25.b);
  const result = {};
  questions.forEach((q, n) => {
    const vector = search.rank(vectors.map((v) => chunkers.cosine(queries[n], v)));
    const keyword = search.rank(bm25.scores(q.question), true);
    const rrf = search.rank(search.rrf([vector, keyword], settings.rrf.k, chunks.length));
    result[q.id] = { vector, rrf };
  });
  return { chunks, ranked: result };
}

// Every [question id, chunk index] among the top n of any first stage, sorted as Python sorts tuples.
export function candidatePairs(ranked, n) {
  const seen = new Set();
  const pairs = [];
  for (const [qid, byStage] of Object.entries(ranked)) {
    for (const order of Object.values(byStage)) {
      for (const i of order.slice(0, n)) {
        const key = `${qid}\u0000${i}`;
        if (!seen.has(key)) {
          seen.add(key);
          pairs.push([qid, i]);
        }
      }
    }
  }
  return pairs.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] - b[1]));
}
