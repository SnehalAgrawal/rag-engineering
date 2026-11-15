"""
The two model calls, both to a local Ollama server:

  embed()  turns text into vectors (nomic-embed-text by default)
  chat()   generates an answer (llama3.2:3b by default)
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()


class EmbeddingError(Exception):
    pass


def chat_model() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")


def embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def embed(texts: list[str], *, truncate: bool = False) -> tuple[list[list[float]], int]:
    """Vectors for the texts, and the number of tokens the embedding model read.

    truncate=False makes a text longer than the model's context an error. With truncate=True, Ollama embeds
    only the beginning of such a text and says nothing.
    """
    body = {"model": embedding_model(), "input": texts, "truncate": truncate}
    response = requests.post(f"{base_url()}/api/embed", json=body, timeout=300)
    if not response.ok:
        raise EmbeddingError(response.json().get("error", f"HTTP {response.status_code}"))
    data = response.json()
    return data["embeddings"], data["prompt_eval_count"]


def chat(messages: list[dict], *, schema: dict, max_tokens: int) -> dict:
    options = {"temperature": 0, "seed": 1, "num_predict": max_tokens}
    body = {"model": chat_model(), "messages": messages, "stream": False, "options": options, "format": schema}
    response = requests.post(f"{base_url()}/api/chat", json=body, timeout=300)
    response.raise_for_status()
    data = response.json()
    return {"content": data["message"]["content"], "prompt_tokens": data["prompt_eval_count"],
            "output_tokens": data["eval_count"], "done_reason": data["done_reason"]}


def unload(model: str) -> None:
    """Unload a model, so memory is freed after a run."""
    requests.post(f"{base_url()}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60).raise_for_status()


def first_token_logprobs(messages: list[dict], *, options: dict, top_logprobs: int) -> dict:
    """Generate one token and return the log-probabilities of the top alternatives at that position."""
    body = {"model": chat_model(), "messages": messages, "stream": False, "options": options,
            "logprobs": True, "top_logprobs": top_logprobs}
    response = requests.post(f"{base_url()}/api/chat", json=body, timeout=300)
    response.raise_for_status()
    data = response.json()
    first = data["logprobs"][0]
    return {"token": first["token"], "top": {t["token"]: t["logprob"] for t in first["top_logprobs"]},
            "prompt_tokens": data["prompt_eval_count"], "load_ns": data.get("load_duration", 0),
            "total_ns": data.get("total_duration", 0)}
