"""
The two model calls, both to a local Ollama server:

  embed()  turns text into vectors (nomic-embed-text by default)
  chat()   generates the reply (llama3.2:3b by default)

Every failure becomes an LLMError with a kind, so the pipeline can say what failed:

  unavailable       no connection: refused, reset, or closed without a response
  timeout           no response within the time the caller allowed
  http_error        the server answered with an error status
  invalid_response  the server answered, but not with what its API promises

The caller passes the timeout. The pipeline passes what is left of the request's deadline.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()


class LLMError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind} ({detail})" if detail else kind)
        self.kind = kind
        self.detail = detail


def chat_model() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")


def embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _post(path: str, body: dict, timeout: float, url: str | None) -> dict:
    # requests applies the timeout to connecting and to each wait for data, not to the whole response.
    # With stream: false, Ollama replies once generation has finished, so here it limits the wait for the reply.
    try:
        response = requests.post(f"{url or base_url()}{path}", json=body, timeout=timeout)
    except requests.Timeout as error:
        raise LLMError("timeout") from error
    except requests.RequestException as error:
        raise LLMError("unavailable") from error
    if not response.ok:
        raise LLMError("http_error", f"HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError as error:
        raise LLMError("invalid_response", "not JSON") from error
    if not isinstance(data, dict):
        raise LLMError("invalid_response", "not a JSON object")
    return data


def embed(texts: list[str], *, timeout: float, url: str | None = None) -> list[list[float]]:
    # truncate=False: an input longer than the model's context is an error, not silently shortened.
    data = _post("/api/embed", {"model": embedding_model(), "input": texts, "truncate": False}, timeout, url)
    vectors = data.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise LLMError("invalid_response", "no embeddings")
    return vectors


def chat(messages: list[dict], *, schema: dict, max_tokens: int, timeout: float, url: str | None = None) -> dict:
    options = {"temperature": 0, "seed": 1, "num_predict": max_tokens}
    body = {"model": chat_model(), "messages": messages, "stream": False, "options": options,
            "format": schema}  # Ollama constrains the output to this JSON schema
    data = _post("/api/chat", body, timeout, url)
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMError("invalid_response", "no message content")
    return {
        "content": content,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "cached_tokens": data.get("prompt_eval_cached_count", 0),  # prompt tokens reused from the server's cache
        "output_tokens": data.get("eval_count", 0),
        "done_reason": data.get("done_reason"),
    }


def unload(model: str) -> None:
    """Unload a model, so runs start from the same state and memory is freed afterwards."""
    requests.post(f"{base_url()}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60).raise_for_status()
