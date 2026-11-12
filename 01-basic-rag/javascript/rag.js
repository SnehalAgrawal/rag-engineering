// The support assistant from Article #11, with one change: step 2 no longer sends the whole
// help center. It retrieves the articles most similar to the question and sends only those.
// Same behaviour as ../python/rag.py.
//   1. validate the input
//   2. retrieve, then build the context     <- the RAG step
//   3. call the model (schema, token limit, timeout)
//   4. check the output (schema, sources, values against the sources)
//   5. decide the reply, and log the request
// answer() never throws: every path returns a status, a message, and a log record.
// mode "all" sends every article instead (Article #11's approach), for comparison.
import { createHash, randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { LLMError, chat, chatModel, embed, embeddingModel } from "./llm.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../data/${name}`, import.meta.url), "utf8"));
export const ARTICLES = load("help_center.json").articles;
const BY_ID = Object.fromEntries(ARTICLES.map((a) => [a.id, a]));
const PROMPT = load("prompt.json");
export const INDEX_PATH = new URL("../data/index.json", import.meta.url);
export const MAX_INPUT_CHARS = 2000;

// nomic-embed-text was trained with task prefixes: one for questions, one for documents.
export const QUERY_PREFIX = process.env.QUERY_PREFIX ?? "search_query: ";
export const DOCUMENT_PREFIX = process.env.DOCUMENT_PREFIX ?? "search_document: ";

// What gets embedded for an article (without the prefix).
export const documentText = (article) => `${article.title}\n${article.text}`;
export const fingerprint = (text) => createHash("sha256").update(text, "utf8").digest("hex");

// The vectors built by index.js, after checking they still match the articles and the settings.
function loadIndex() {
  const fail = (message) => {
    console.error(message);
    process.exit(1);
  };
  if (!existsSync(INDEX_PATH)) fail("No index found. Run: node index.js");
  const index = JSON.parse(readFileSync(INDEX_PATH, "utf8"));
  const settings = { embedding_model: embeddingModel(), document_prefix: DOCUMENT_PREFIX, query_prefix: QUERY_PREFIX };
  for (const [key, value] of Object.entries(settings)) {
    if (index[key] !== value) {
      fail(`The index was built with ${key}=${JSON.stringify(index[key])}, but this run uses ${JSON.stringify(value)}. Run: node index.js`);
    }
  }
  const stored = JSON.stringify(index.articles.map((e) => [e.id, e.sha256]));
  const current = JSON.stringify(ARTICLES.map((a) => [a.id, fingerprint(documentText(a))]));
  if (stored !== current) fail("The help center changed since the index was built. Run: node index.js");
  return index.articles;
}

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

let index = null;

// The k articles whose vectors are most similar to the question's, best first.
export async function retrieve(question, k, url = null) {
  if (index === null) index = loadIndex();
  const [query] = await embed([QUERY_PREFIX + question], { url });
  const scored = index.map((entry, position) => ({ score: cosine(query, entry.vector), position, id: entry.id }));
  scored.sort((x, y) => y.score - x.score || x.position - y.position); // ties keep help-center order
  return scored.slice(0, k).map(({ id, score }) => ({ id, score }));
}

function buildSystem(articles) {
  const text = articles.map((a) => `[${a.id}] ${a.title}\n${a.text}`).join("\n\n");
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

export async function answer(text, { mode = "rag", k = 3, numCtx = null, url = null } = {}) {
  const start = performance.now();
  const question = text.trim();
  const log = {
    request_id: randomUUID(),
    prompt_version: PROMPT.version,
    model: chatModel(),
    embedding_model: mode === "rag" ? embeddingModel() : null,
    mode,
    k: mode === "rag" ? k : null,
    num_ctx: numCtx,
    input_chars: question.length,
    retrieved: null,
    prompt_tokens: null,
    cached_tokens: null,
    output_tokens: null,
    done_reason: null,
    error: null,
  };
  let retrieved = [], modelOutput = null;
  const reply = (status, message, outcome, sources = []) => {
    Object.assign(log, { status, outcome, latency_ms: Math.round(performance.now() - start) });
    return { status, message, sources: [...sources], retrieved, model_output: modelOutput, log };
  };

  // 1. Validate the input.
  if (!question) return reply(400, "Please type a question.", "empty_input");
  if (question.length > MAX_INPUT_CHARS) {
    return reply(413, `Your message is too long (${question.length.toLocaleString("en-US")} characters). Please keep questions under ` +
      `${MAX_INPUT_CHARS.toLocaleString("en-US")} characters and leave out logs or attachments.`, "input_too_long");
  }

  // 2. Retrieve, then build the context.
  let context, result;
  try {
    if (mode === "rag") {
      retrieved = await retrieve(question, k, url);
      context = retrieved.map((r) => BY_ID[r.id]);
      log.retrieved = retrieved.map((r) => r.id);
    } else if (mode === "all") {
      context = ARTICLES;
    } else {
      throw new Error(`unknown mode: ${mode}`);
    }

    // 3. Call the model with a schema, a token limit, and a timeout.
    const ids = context.map((a) => a.id);
    const messages = [{ role: "system", content: buildSystem(context) }, { role: "user", content: question }];
    result = await chat(messages, { schema: schemaFor(ids), maxTokens: 300, numCtx, url });
  } catch (error) {
    if (!(error instanceof LLMError)) throw error;
    log.error = error.message;
    return reply(503, PROMPT.unavailable, "model_unavailable");
  }
  Object.assign(log, {
    prompt_tokens: result.prompt_tokens, cached_tokens: result.cached_tokens, output_tokens: result.output_tokens,
    done_reason: result.done_reason,
  });
  modelOutput = result.content;

  // 4. Check the output.
  const data = parse(result.content, context.map((a) => a.id));
  if (data === null) return reply(200, PROMPT.decline, "invalid_output");
  if (!data.answered) return reply(200, PROMPT.decline, "declined");
  const cited = data.source_ids.map((s) => BY_ID[s].text);
  if (!data.answer.trim() || !cited.length || unsupportedValues(data.answer, cited).length) {
    return reply(200, PROMPT.decline, "unsupported_answer");
  }

  // 5. The application decides the reply: the checked answer, with its sources.
  return reply(200, data.answer.trim(), "answered", data.source_ids);
}
