"""
The two model calls this demo makes, both to a local Ollama server:

  embed()  turns text into vectors (nomic-embed-text by default)
  chat()   generates the reply (llama3.2:3b by default)

Every failure on the way (server down, timeout, HTTP error, a malformed
response from the server) becomes one exception type, LLMError, so the rest of
the application has a single thing to handle.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()


class LLMError(Exception):
    pass


def chat_model() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")


def embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _post(path: str, body: dict, timeout: float, url: str | None) -> dict:
    try:
        response = requests.post(f"{url or base_url()}{path}", json=body, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise LLMError(type(error).__name__) from error
    try:
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("response is not a JSON object")
        return data
    except (ValueError, TypeError) as error:
        raise LLMError("InvalidResponse") from error


def embed(texts: list[str], *, timeout: float = 60, url: str | None = None) -> list[list[float]]:
    # truncate=False: an input longer than the model's context is an error, not silently shortened.
    data = _post("/api/embed", {"model": embedding_model(), "input": texts, "truncate": False}, timeout, url)
    vectors = data.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise LLMError("InvalidResponse")
    return vectors


def chat(messages: list[dict], *, schema: dict | None = None, max_tokens: int = 300, num_ctx: int | None = None,
         timeout: float = 120, url: str | None = None) -> dict:
    options = {"temperature": 0, "seed": 1, "num_predict": max_tokens}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx  # otherwise Ollama chooses the context length
    body = {"model": chat_model(), "messages": messages, "stream": False, "options": options}
    if schema is not None:
        body["format"] = schema  # Ollama constrains the output to this JSON schema
    data = _post("/api/chat", body, timeout, url)
    try:
        content = data["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("content is not a string")
    except (KeyError, TypeError) as error:
        raise LLMError("InvalidResponse") from error
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
