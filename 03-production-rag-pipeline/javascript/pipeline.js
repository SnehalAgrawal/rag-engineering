// The support assistant from 01-basic-rag as a pipeline with production concerns. Same behaviour as
// ../python/pipeline.py. The prompt, the model, the schema, and the output checks are the same as 01's;
// what changes is everything around them:
//   validate -> embed -> retrieve -> gate -> generate -> check -> reply
// - One deadline for the whole request. Each model call gets the time that is left.
// - Every failure has a name (the outcome), and the customer sees a message the application chose.
// - A relevance gate: when the best article scores below a threshold, decline without calling the chat model.
// - The index is a versioned snapshot (index_store.js). The assistant serves the last complete version,
//   loads a new one when it is written, and reports how many articles have changed since it was built.
// - Every request produces one trace: versions, settings, stage timings, retrieval scores, the gate
//   decision, token counts, and the outcome.
// The settings come from ../data/settings.json. The "basic" profile reproduces 01-basic-rag's behavior
// (no deadline, a 60 s timeout for embedding and 120 s for the chat call, no gate); "production" turns the
// deadline and the gate on. answer() never throws.
import { randomUUID } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import * as store from "./index_store.js";
import { LLMError, chat, chatModel, embed, embeddingModel } from "./llm.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../data/${name}`, import.meta.url), "utf8"));
const PROMPT = load("prompt.json");
export const PROFILES = Object.fromEntries(Object.entries(load("settings.json")).filter(([name]) => name !== "_comment"));
// Without a deadline (the basic profile), each call gets 01-basic-rag's fixed timeout.
const CALL_TIMEOUTS = { retrieve: 60, generate: 120 };

export function cosine(a, b) {
  if (a.length !== b.length) throw new Error(`dimension mismatch: ${a.length} vs ${b.length}`);
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) throw new Error("zero-length vector");
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function buildSystem(entries) {
  const text = entries.map((e) => `[${e.id}] ${e.title}\n${e.text}`).join("\n\n");
  return PROMPT.system.replace("{articles}", () => text);
}

const schemaFor = (ids) => ({
  type: "object",
  properties: {
    answered: { type: "boolean" },
    answer: { type: "string" },
    // The model can only cite articles it was given.
    source_ids: { type: "array", items: { type: "string", enum: ids } },
  },
  required: ["answered", "answer", "source_ids"],
  additionalProperties: false,
});

// Article #9's value-presence check: prices, durations, sizes, emails in the answer
// must appear in the articles it cites.
const VALUE = /\$\s?\d[\d,]*(?:\.\d+)?|\b\d+(?:\s*(?:-|–|to)\s*\d+)?\s*(?:business\s+)?(?:minutes?|hours?|days?|weeks?|months?)\b|\b\d+(?:\.\d+)?\s*(?:MB|GB|TB)\b|\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+/gi;
const normalize = (value) => value.toLowerCase().replace(/[\s,$]/g, "").replace(/\.0+(?!\d)/g, "");
const values = (text) => text.match(VALUE) || [];

export function unsupportedValues(answer, sources) {
  const known = new Set(sources.flatMap((text) => values(text).map(normalize)));
  return values(answer).filter((v) => !known.has(normalize(v)));
}

// The model's JSON reply, or null if it isn't exactly the agreed shape.
function parse(content, allowed) {
  let data;
  try {
    data = JSON.parse(content);
  } catch {
    return null;
  }
  const keys = data && typeof data === "object" && !Array.isArray(data) ? Object.keys(data).sort().join(",") : "";
  if (keys !== "answer,answered,source_ids" || typeof data.answered !== "boolean" || typeof data.answer !== "string"
      || !Array.isArray(data.source_ids) || !data.source_ids.every((s) => allowed.includes(s))) return null;
  return data;
}

const round4 = (x) => Number(x.toFixed(4));
const mtime = (path) => statSync(path, { bigint: true }).mtimeNs;

export class Assistant {
  constructor(profile = "production", { indexPath = store.INDEX_PATH, helpCenterPath = store.HELP_CENTER_PATH, url = null } = {}) {
    this.profile = profile;
    this.settings = PROFILES[profile];
    Object.assign(this, { indexPath, helpCenterPath, url });
    this.index = null;
    this.stale = null;
    this.seen = null; // [index mtime, help center mtime] at the last check
    this.refresh();
  }

  // Load the index if a new version was written, and recount the articles changed since it was built.
  refresh() {
    let seen;
    try {
      seen = [mtime(this.indexPath), mtime(this.helpCenterPath)];
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      if (this.index === null) {
        console.error("No index found. Run: node index.js");
        process.exit(1);
      }
      return; // keep serving the index already loaded
    }
    if (this.seen && seen[0] === this.seen[0] && seen[1] === this.seen[1]) return;
    if (this.index === null || seen[0] !== this.seen[0]) {
      const index = store.read(this.indexPath);
      for (const [key, value] of Object.entries(store.settings(embeddingModel()))) {
        if (index[key] !== value) {
          console.error(`The index was built with ${key}=${JSON.stringify(index[key])}, but this run uses ${JSON.stringify(value)}. Run: node index.js`);
          process.exit(1);
        }
      }
      this.index = index;
    }
    this.stale = store.staleness(this.index, store.readArticles(this.helpCenterPath));
    this.seen = seen;
  }

  // The k entries whose vectors are most similar to the question's, best first. Throws LLMError.
  async retrieve(question, k, { timeout }) {
    const [query] = await embed([store.QUERY_PREFIX + question], { timeout, url: this.url });
    const entries = this.index.entries;
    const scored = entries.map((e, position) => ({ score: cosine(query, e.vector), position }));
    scored.sort((x, y) => y.score - x.score || x.position - y.position); // ties keep help-center order
    return scored.slice(0, k).map(({ score, position }) => ({ id: entries[position].id, score, entry: entries[position] }));
  }

  async answer(text) {
    const start = performance.now();
    const s = this.settings;
    const deadline = s.deadline_seconds ? start + s.deadline_seconds * 1000 : null;
    this.refresh();
    const question = text.trim();
    const trace = {
      request_id: randomUUID(),
      profile: this.profile,
      prompt_version: PROMPT.version,
      chat_model: chatModel(),
      embedding_model: embeddingModel(),
      index_version: this.index.version,
      index_stale_articles: this.stale,
      k: s.k,
      deadline_seconds: s.deadline_seconds,
      relevance_threshold: s.relevance_threshold,
      input_chars: question.length,
      stages: {},
      retrieved: [],
      gate: null,
      prompt_tokens: null,
      cached_tokens: null,
      output_tokens: null,
      done_reason: null,
      error: null,
    };
    let modelOutput = null;

    const reply = (status, message, outcome, sources = []) => {
      Object.assign(trace, { status, outcome, latency_ms: Math.round(performance.now() - start) });
      return { status, message, sources: [...sources], model_output: modelOutput, trace };
    };
    const timeLeft = (stage) => {
      if (deadline === null) return CALL_TIMEOUTS[stage];
      const left = (deadline - performance.now()) / 1000;
      if (left <= 0) throw new LLMError("deadline");
      return left;
    };
    const timed = (stage, began) => { trace.stages[stage] = Math.round(performance.now() - began); };
    const failed = (stage, error) => {
      if (!(error instanceof LLMError)) throw error;
      trace.error = { stage, kind: error.kind, detail: error.detail };
      return reply(503, PROMPT.unavailable, "unavailable");
    };

    // 1. Validate the input.
    if (!question) return reply(400, "Please type a question.", "empty_input");
    if (question.length > s.max_input_chars) {
      return reply(413, `Your message is too long (${question.length.toLocaleString("en-US")} characters). Please keep questions under ` +
        `${s.max_input_chars.toLocaleString("en-US")} characters and leave out logs or attachments.`, "input_too_long");
    }

    // 2-3. Retrieve: embed the question (a model call), then rank the articles by cosine similarity.
    let began = performance.now(), retrieved;
    try {
      retrieved = await this.retrieve(question, s.k, { timeout: timeLeft("retrieve") });
    } catch (error) {
      timed("retrieve", began);
      return failed("retrieve", error);
    }
    timed("retrieve", began); // the embedding call and the in-memory search over 50 vectors
    trace.retrieved = retrieved.map((r) => ({ id: r.id, score: round4(r.score) }));

    // 4. The relevance gate: if even the best article is a weak match, don't ask the model.
    const top = retrieved[0].score;
    const threshold = s.relevance_threshold;
    trace.gate = { top_score: round4(top), threshold, passed: threshold === null || top >= threshold };
    if (!trace.gate.passed) return reply(200, PROMPT.decline, "no_relevant_article");

    // 5. Generate, with a schema, a token limit, and the time that is left.
    const entries = retrieved.map((r) => r.entry);
    const ids = entries.map((e) => e.id);
    const messages = [{ role: "system", content: buildSystem(entries) }, { role: "user", content: question }];
    began = performance.now();
    let result;
    try {
      result = await chat(messages, { schema: schemaFor(ids), maxTokens: s.max_output_tokens, timeout: timeLeft("generate"), url: this.url });
    } catch (error) {
      timed("generate", began);
      return failed("generate", error);
    }
    timed("generate", began);
    Object.assign(trace, {
      prompt_tokens: result.prompt_tokens, cached_tokens: result.cached_tokens, output_tokens: result.output_tokens,
      done_reason: result.done_reason,
    });
    modelOutput = result.content;

    // 6. Check the output: the agreed shape, cited ids, and values against the cited articles.
    const data = parse(result.content, ids);
    if (data === null) return reply(200, PROMPT.decline, "invalid_output");
    if (!data.answered) return reply(200, PROMPT.decline, "declined");
    const byId = Object.fromEntries(entries.map((e) => [e.id, e]));
    const cited = data.source_ids.map((id) => byId[id].text);
    if (!data.answer.trim() || !cited.length || unsupportedValues(data.answer, cited).length) {
      return reply(200, PROMPT.decline, "unsupported_answer");
    }

    // 7. The application decides the reply: the checked answer, with its sources.
    return reply(200, data.answer.trim(), "answered", data.source_ids);
  }
}
