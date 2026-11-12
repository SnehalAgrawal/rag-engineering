// The two model calls this demo makes, both to a local Ollama server. Same behaviour as ../python/llm.py.
//   embed()  turns text into vectors (nomic-embed-text by default)
//   chat()   generates the reply (llama3.2:3b by default)
// Every failure on the way (server down, timeout, HTTP error, a malformed response from the
// server) becomes one error type, LLMError, so the rest of the application has a single thing to handle.
import "dotenv/config";

export class LLMError extends Error {}

export const chatModel = () => process.env.OLLAMA_CHAT_MODEL || "llama3.2:3b";
export const embeddingModel = () => process.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text";
export const baseUrl = () => process.env.OLLAMA_BASE_URL || "http://localhost:11434";

async function post(path, body, timeout, url) {
  let response;
  try {
    response = await fetch(`${url || baseUrl()}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeout * 1000),
    });
  } catch (error) {
    throw new LLMError(error.name === "TimeoutError" ? "Timeout" : "ConnectionError");
  }
  if (!response.ok) throw new LLMError(`HTTPError ${response.status}`);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new LLMError("InvalidResponse");
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new LLMError("InvalidResponse");
  return data;
}

export async function embed(texts, { timeout = 60, url = null } = {}) {
  // truncate: false: an input longer than the model's context is an error, not silently shortened.
  const data = await post("/api/embed", { model: embeddingModel(), input: texts, truncate: false }, timeout, url);
  if (!Array.isArray(data.embeddings) || data.embeddings.length !== texts.length) throw new LLMError("InvalidResponse");
  return data.embeddings;
}

export async function chat(messages, { schema = null, maxTokens = 300, numCtx = null, timeout = 120, url = null } = {}) {
  const options = { temperature: 0, seed: 1, num_predict: maxTokens };
  if (numCtx !== null) options.num_ctx = numCtx; // otherwise Ollama chooses the context length
  const body = { model: chatModel(), messages, stream: false, options };
  if (schema !== null) body.format = schema; // Ollama constrains the output to this JSON schema
  const data = await post("/api/chat", body, timeout, url);
  const content = data.message?.content;
  if (typeof content !== "string") throw new LLMError("InvalidResponse");
  return {
    content,
    prompt_tokens: data.prompt_eval_count ?? 0,
    cached_tokens: data.prompt_eval_cached_count ?? 0, // prompt tokens reused from the server's cache
    output_tokens: data.eval_count ?? 0,
    done_reason: data.done_reason ?? null,
  };
}

// Unload a model, so runs start from the same state and memory is freed afterwards.
export async function unload(model) {
  const response = await fetch(`${baseUrl()}/api/generate`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model, keep_alive: 0 }),
  });
  if (!response.ok) throw new Error(`Ollama returned ${response.status}`);
}
