// The two model calls, both to a local Ollama server. Same behaviour as ../python/llm.py.
//   embed()  turns text into vectors (nomic-embed-text by default)
//   chat()   generates an answer (llama3.2:3b by default)
import "dotenv/config";

export class EmbeddingError extends Error {}

export const chatModel = () => process.env.OLLAMA_CHAT_MODEL || "llama3.2:3b";
export const embeddingModel = () => process.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text";
export const baseUrl = () => process.env.OLLAMA_BASE_URL || "http://localhost:11434";

async function post(path, body) {
  const response = await fetch(`${baseUrl()}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    signal: AbortSignal.timeout(300_000),
  });
  return { response, data: await response.json() };
}

// Vectors for the texts, and the number of tokens the embedding model read.
// truncate: false makes a text longer than the model's context an error. With truncate: true, Ollama embeds
// only the beginning of such a text and says nothing.
export async function embed(texts, { truncate = false } = {}) {
  const { response, data } = await post("/api/embed", { model: embeddingModel(), input: texts, truncate });
  if (!response.ok) throw new EmbeddingError(data.error || `HTTP ${response.status}`);
  return { vectors: data.embeddings, tokens: data.prompt_eval_count };
}

export async function chat(messages, { schema, maxTokens }) {
  const options = { temperature: 0, seed: 1, num_predict: maxTokens };
  const { response, data } = await post("/api/chat", {
    model: chatModel(), messages, stream: false, options, format: schema,
  });
  if (!response.ok) throw new Error(`chat failed: HTTP ${response.status}`);
  return {
    content: data.message.content, promptTokens: data.prompt_eval_count,
    outputTokens: data.eval_count, doneReason: data.done_reason,
  };
}

// Unload a model, so memory is freed after a run.
export async function unload(model) {
  const { response } = await post("/api/generate", { model, keep_alive: 0 });
  if (!response.ok) throw new Error(`unload failed: HTTP ${response.status}`);
}
