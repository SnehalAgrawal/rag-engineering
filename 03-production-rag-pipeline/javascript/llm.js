// The two model calls, both to a local Ollama server. Same behaviour as ../python/llm.py.
//   embed()  turns text into vectors (nomic-embed-text by default)
//   chat()   generates the reply (llama3.2:3b by default)
// Every failure becomes an LLMError with a kind, so the pipeline can say what failed:
//   unavailable       no connection: refused, reset, or closed without a response
//   timeout           no response within the time the caller allowed
//   http_error        the server answered with an error status
//   invalid_response  the server answered, but not with what its API promises
// The caller passes the timeout (in seconds). The pipeline passes what is left of the request's deadline.
import "dotenv/config";

export class LLMError extends Error {
  constructor(kind, detail = "") {
    super(detail ? `${kind} (${detail})` : kind);
    this.kind = kind;
    this.detail = detail;
  }
}

export const chatModel = () => process.env.OLLAMA_CHAT_MODEL || "llama3.2:3b";
export const embeddingModel = () => process.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text";
export const baseUrl = () => process.env.OLLAMA_BASE_URL || "http://localhost:11434";

async function post(path, body, timeout, url) {
  // AbortSignal.timeout limits the whole request, including reading the response body.
  const signal = AbortSignal.timeout(Math.max(1, Math.ceil(timeout * 1000)));
  let response, data;
  try {
    response = await fetch(`${url || baseUrl()}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    });
  } catch (error) {
    throw new LLMError(error.name === "TimeoutError" ? "timeout" : "unavailable");
  }
  if (!response.ok) throw new LLMError("http_error", `HTTP ${response.status}`);
  try {
    data = await response.json();
  } catch (error) {
    if (error.name === "TimeoutError") throw new LLMError("timeout");
    throw new LLMError("invalid_response", "not JSON");
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new LLMError("invalid_response", "not a JSON object");
  return data;
}

export async function embed(texts, { timeout, url = null }) {
  // truncate: false: an input longer than the model's context is an error, not silently shortened.
  const data = await post("/api/embed", { model: embeddingModel(), input: texts, truncate: false }, timeout, url);
  if (!Array.isArray(data.embeddings) || data.embeddings.length !== texts.length) {
    throw new LLMError("invalid_response", "no embeddings");
  }
  return data.embeddings;
}

export async function chat(messages, { schema, maxTokens, timeout, url = null }) {
  const options = { temperature: 0, seed: 1, num_predict: maxTokens };
  const body = {
    model: chatModel(), messages, stream: false, options,
    format: schema, // Ollama constrains the output to this JSON schema
  };
  const data = await post("/api/chat", body, timeout, url);
  const content = data.message && typeof data.message === "object" ? data.message.content : undefined;
  if (typeof content !== "string") throw new LLMError("invalid_response", "no message content");
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
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new Error(`Ollama returned ${response.status}`);
}
