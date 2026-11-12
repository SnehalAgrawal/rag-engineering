"""
The support assistant from Article #11, with one change: step 2 no longer
sends the whole help center. It retrieves the articles most similar to the
question and sends only those.

  1. validate the input
  2. retrieve, then build the context     <- the RAG step
  3. call the model (schema, token limit, timeout)
  4. check the output (schema, sources, values against the sources)
  5. decide the reply, and log the request

answer() never raises: every path returns a status, a message, and a log record.
mode="all" sends every article instead (Article #11's approach), for comparison.
"""
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

from llm import LLMError, chat, chat_model, embed, embedding_model

DATA = Path(__file__).resolve().parent.parent / "data"
ARTICLES = json.loads((DATA / "help_center.json").read_text())["articles"]
BY_ID = {a["id"]: a for a in ARTICLES}
PROMPT = json.loads((DATA / "prompt.json").read_text())
INDEX_PATH = DATA / "index.json"
MAX_INPUT_CHARS = 2000

# nomic-embed-text was trained with task prefixes: one for questions, one for documents.
QUERY_PREFIX = os.environ.get("QUERY_PREFIX", "search_query: ")
DOCUMENT_PREFIX = os.environ.get("DOCUMENT_PREFIX", "search_document: ")


def document_text(article: dict) -> str:
    """What gets embedded for an article (without the prefix)."""
    return f"{article['title']}\n{article['text']}"


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_index() -> list[dict]:
    """The vectors built by index.py, after checking they still match the articles and the settings."""
    if not INDEX_PATH.exists():
        raise SystemExit("No index found. Run: python index.py")
    index = json.loads(INDEX_PATH.read_text())
    settings = {"embedding_model": embedding_model(), "document_prefix": DOCUMENT_PREFIX, "query_prefix": QUERY_PREFIX}
    for key, value in settings.items():
        if index[key] != value:
            raise SystemExit(f"The index was built with {key}={index[key]!r}, but this run uses {value!r}. Run: python index.py")
    stored = [(entry["id"], entry["sha256"]) for entry in index["articles"]]
    current = [(a["id"], fingerprint(document_text(a))) for a in ARTICLES]
    if stored != current:
        raise SystemExit("The help center changed since the index was built. Run: python index.py")
    return index["articles"]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        raise ValueError("zero-length vector")
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


_index = None


def retrieve(question: str, k: int, url: str | None = None) -> list[dict]:
    """The k articles whose vectors are most similar to the question's, best first."""
    global _index
    if _index is None:
        _index = load_index()
    query = embed([QUERY_PREFIX + question], url=url)[0]
    scored = [(cosine(query, entry["vector"]), position, entry["id"]) for position, entry in enumerate(_index)]
    scored.sort(key=lambda s: (-s[0], s[1]))  # ties keep help-center order
    return [{"id": article_id, "score": score} for score, _, article_id in scored[:k]]


def build_system(articles: list[dict]) -> str:
    text = "\n\n".join(f"[{a['id']}] {a['title']}\n{a['text']}" for a in articles)
    return PROMPT["system"].replace("{articles}", text)


def schema_for(ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "answered": {"type": "boolean"},
            "answer": {"type": "string"},
            # The model can only cite articles it was given.
            "source_ids": {"type": "array", "items": {"type": "string", "enum": ids}},
        },
        "required": ["answered", "answer", "source_ids"],
        "additionalProperties": False,
    }


# Article #9's value-presence check: prices, durations, sizes, emails in the answer
# must appear in the articles it cites.
VALUE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?"
    r"|\b\d+(?:\s*(?:-|–|to)\s*\d+)?\s*(?:business\s+)?(?:minutes?|hours?|days?|weeks?|months?)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:MB|GB|TB)\b"
    r"|\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return re.sub(r"\.0+(?!\d)", "", re.sub(r"[\s,$]", "", value.lower()))


def unsupported_values(answer: str, sources: list[str]) -> list[str]:
    known = {_normalize(v) for text in sources for v in VALUE.findall(text)}
    return [v for v in VALUE.findall(answer) if _normalize(v) not in known]


def _parse(content: str, allowed: list[str]):
    """The model's JSON reply, or None if it isn't exactly the agreed shape."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if (not isinstance(data, dict) or set(data) != {"answered", "answer", "source_ids"}
            or not isinstance(data["answered"], bool) or not isinstance(data["answer"], str)
            or not isinstance(data["source_ids"], list)
            or not all(s in allowed for s in data["source_ids"])):
        return None
    return data


def answer(text: str, *, mode: str = "rag", k: int = 3, num_ctx: int | None = None, url: str | None = None) -> dict:
    start = time.perf_counter()
    question = text.strip()
    log = {
        "request_id": str(uuid.uuid4()),
        "prompt_version": PROMPT["version"],
        "model": chat_model(),
        "embedding_model": embedding_model() if mode == "rag" else None,
        "mode": mode,
        "k": k if mode == "rag" else None,
        "num_ctx": num_ctx,
        "input_chars": len(question),
        "retrieved": None,
        "prompt_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
        "done_reason": None,
        "error": None,
    }
    retrieved, model_output = [], None

    def reply(status: int, message: str, outcome: str, sources=()):
        log.update(status=status, outcome=outcome, latency_ms=round((time.perf_counter() - start) * 1000))
        return {"status": status, "message": message, "sources": list(sources), "retrieved": retrieved,
                "model_output": model_output, "log": log}

    # 1. Validate the input.
    if not question:
        return reply(400, "Please type a question.", "empty_input")
    if len(question) > MAX_INPUT_CHARS:
        return reply(413, f"Your message is too long ({len(question):,} characters). Please keep questions under "
                          f"{MAX_INPUT_CHARS:,} characters and leave out logs or attachments.", "input_too_long")

    # 2. Retrieve, then build the context.
    try:
        if mode == "rag":
            retrieved = retrieve(question, k, url=url)
            context = [BY_ID[r["id"]] for r in retrieved]
            log["retrieved"] = [r["id"] for r in retrieved]
        elif mode == "all":
            context = ARTICLES
        else:
            raise ValueError(f"unknown mode: {mode}")

        # 3. Call the model with a schema, a token limit, and a timeout.
        ids = [a["id"] for a in context]
        messages = [{"role": "system", "content": build_system(context)}, {"role": "user", "content": question}]
        result = chat(messages, schema=schema_for(ids), max_tokens=300, num_ctx=num_ctx, url=url)
    except LLMError as error:
        log["error"] = str(error)
        return reply(503, PROMPT["unavailable"], "model_unavailable")
    log.update(prompt_tokens=result["prompt_tokens"], cached_tokens=result["cached_tokens"], output_tokens=result["output_tokens"],
               done_reason=result["done_reason"])
    model_output = result["content"]

    # 4. Check the output.
    data = _parse(result["content"], ids)
    if data is None:
        return reply(200, PROMPT["decline"], "invalid_output")
    if not data["answered"]:
        return reply(200, PROMPT["decline"], "declined")
    cited = [BY_ID[s]["text"] for s in data["source_ids"]]
    if not data["answer"].strip() or not cited or unsupported_values(data["answer"], cited):
        return reply(200, PROMPT["decline"], "unsupported_answer")

    # 5. The application decides the reply: the checked answer, with its sources.
    return reply(200, data["answer"].strip(), "answered", data["source_ids"])
