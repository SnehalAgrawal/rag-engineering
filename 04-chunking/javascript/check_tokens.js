// Check tokens.js against the model server. Same behaviour and output as ../python/check_tokens.py:
// Ollama adds the begin-of-text token to a raw prompt, so its count should be ours plus one.
import "dotenv/config";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { Tokenizer } from "./tokens.js";

const DATA = join(dirname(fileURLToPath(import.meta.url)), "..", "data");
const baseUrl = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
const model = process.env.OLLAMA_CHAT_MODEL || "llama3.2:3b";

async function serverCount(text) {
  const response = await fetch(`${baseUrl}/api/generate`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, prompt: text, raw: true, stream: false, options: { num_predict: 1, temperature: 0, seed: 1 } }),
  });
  return (await response.json()).prompt_eval_count; // includes tokens served from the prompt cache
}

const tokenizer = new Tokenizer();
const documents = readdirSync(join(DATA, "documents")).filter((n) => n.endsWith(".md")).sort()
  .map((n) => readFileSync(join(DATA, "documents", n), "utf8"));
const questions = JSON.parse(readFileSync(join(DATA, "questions.json"), "utf8")).questions.map((q) => q.question);
const pieces = documents.flatMap((doc) => [[0, 700], [1234, 999], [3001, 1777]].map(([s, n]) => doc.slice(s, s + n)));
const texts = [...documents, ...questions, ...pieces];
let mismatches = 0;
try {
  for (const text of texts) {
    const ours = tokenizer.count(text);
    const theirs = await serverCount(text);
    if (theirs !== ours + 1) {
      mismatches += 1;
      console.log(`MISMATCH ours=${ours} server=${theirs}: ${JSON.stringify(text.slice(0, 60))}`);
    }
  }
} finally {
  await fetch(`${baseUrl}/api/generate`, { method: "POST", body: JSON.stringify({ model, keep_alive: 0 }) });
}
const total = texts.reduce((s, t) => s + tokenizer.count(t), 0);
console.log(`${texts.length} texts (${documents.length} documents, ${questions.length} questions, ${pieces.length} pieces), `
  + `${total} tokens: ${mismatches} mismatches`);
