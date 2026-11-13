"""
The support assistant from 01-basic-rag as a pipeline with production concerns. The prompt, the
model, the schema, and the output checks are the same; what changes is everything around them:

  validate -> embed -> retrieve -> gate -> generate -> check -> reply

- One deadline for the whole request. Each model call gets the time that is left.
- Every failure has a name (the outcome), and the customer sees a message the application chose.
- A relevance gate: when the best article scores below a threshold, decline without calling the chat model.
- The index is a versioned snapshot (index_store.py). The assistant serves the last complete version,
  loads a new one when it is written, and reports how many articles have changed since it was built.
- Every request produces one trace: versions, settings, stage timings, retrieval scores, the gate
  decision, token counts, and the outcome.

The settings come from ../data/settings.json. The "basic" profile reproduces 01-basic-rag's behavior
(no deadline, a 60 s timeout for embedding and 120 s for the chat call, no gate); "production" turns the
deadline and the gate on. answer() never raises.
"""
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

import index_store
from llm import LLMError, chat, chat_model, embed, embedding_model

DATA = Path(__file__).resolve().parent.parent / "data"
PROMPT = json.loads((DATA / "prompt.json").read_text())
PROFILES = {name: value for name, value in json.loads((DATA / "settings.json").read_text()).items() if name != "_comment"}
# Without a deadline (the basic profile), each call gets 01-basic-rag's fixed timeout.
CALL_TIMEOUTS = {"retrieve": 60.0, "generate": 120.0}


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


def build_system(entries: list[dict]) -> str:
    text = "\n\n".join(f"[{e['id']}] {e['title']}\n{e['text']}" for e in entries)
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


class Assistant:
    def __init__(self, profile: str = "production", *, index_path=index_store.INDEX_PATH,
                 help_center_path=index_store.HELP_CENTER_PATH, url: str | None = None):
        self.profile = profile
        self.settings = PROFILES[profile]
        self.index_path, self.help_center_path, self.url = Path(index_path), Path(help_center_path), url
        self.index, self.stale = None, None
        self._seen = None  # (index mtime, help center mtime) at the last check
        self._refresh()

    def _refresh(self) -> None:
        """Load the index if a new version was written, and recount the articles changed since it was built."""
        try:
            seen = (os.stat(self.index_path).st_mtime_ns, os.stat(self.help_center_path).st_mtime_ns)
        except FileNotFoundError:
            if self.index is None:
                raise SystemExit("No index found. Run: python index.py") from None
            return  # keep serving the index already loaded
        if seen == self._seen:
            return
        if self.index is None or seen[0] != self._seen[0]:
            index = index_store.read(self.index_path)
            expected = index_store.settings(embedding_model())
            for key, value in expected.items():
                if index[key] != value:
                    raise SystemExit(f"The index was built with {key}={index[key]!r}, but this run uses {value!r}. "
                                     "Run: python index.py")
            self.index = index
        self.stale = index_store.staleness(self.index, index_store.read_articles(self.help_center_path))
        self._seen = seen

    def retrieve(self, question: str, k: int, *, timeout: float) -> list[dict]:
        """The k entries whose vectors are most similar to the question's, best first. Raises LLMError."""
        query = embed([index_store.QUERY_PREFIX + question], timeout=timeout, url=self.url)[0]
        entries = self.index["entries"]
        scored = [(cosine(query, e["vector"]), position) for position, e in enumerate(entries)]
        scored.sort(key=lambda s: (-s[0], s[1]))  # ties keep help-center order
        return [{"id": entries[position]["id"], "score": score, "entry": entries[position]} for score, position in scored[:k]]

    def answer(self, text: str) -> dict:
        start = time.perf_counter()
        s = self.settings
        deadline = start + s["deadline_seconds"] if s["deadline_seconds"] else None
        self._refresh()
        question = text.strip()
        trace = {
            "request_id": str(uuid.uuid4()),
            "profile": self.profile,
            "prompt_version": PROMPT["version"],
            "chat_model": chat_model(),
            "embedding_model": embedding_model(),
            "index_version": self.index["version"],
            "index_stale_articles": self.stale,
            "k": s["k"],
            "deadline_seconds": s["deadline_seconds"],
            "relevance_threshold": s["relevance_threshold"],
            "input_chars": len(question),
            "stages": {},
            "retrieved": [],
            "gate": None,
            "prompt_tokens": None,
            "cached_tokens": None,
            "output_tokens": None,
            "done_reason": None,
            "error": None,
        }
        model_output = None

        def reply(status: int, message: str, outcome: str, sources=()):
            trace.update(status=status, outcome=outcome, latency_ms=round((time.perf_counter() - start) * 1000))
            return {"status": status, "message": message, "sources": list(sources), "model_output": model_output,
                    "trace": trace}

        def time_left(stage: str) -> float:
            if deadline is None:
                return CALL_TIMEOUTS[stage]
            left = deadline - time.perf_counter()
            if left <= 0:
                raise LLMError("deadline")
            return left

        def timed(stage: str, began: float) -> None:
            trace["stages"][stage] = round((time.perf_counter() - began) * 1000)

        def failed(stage: str, error: LLMError):
            trace["error"] = {"stage": stage, "kind": error.kind, "detail": error.detail}
            return reply(503, PROMPT["unavailable"], "unavailable")

        # 1. Validate the input.
        if not question:
            return reply(400, "Please type a question.", "empty_input")
        if len(question) > s["max_input_chars"]:
            return reply(413, f"Your message is too long ({len(question):,} characters). Please keep questions under "
                              f"{s['max_input_chars']:,} characters and leave out logs or attachments.", "input_too_long")

        # 2-3. Retrieve: embed the question (a model call), then rank the articles by cosine similarity.
        began = time.perf_counter()
        try:
            retrieved = self.retrieve(question, s["k"], timeout=time_left("retrieve"))
        except LLMError as error:
            timed("retrieve", began)
            return failed("retrieve", error)
        timed("retrieve", began)  # the embedding call and the in-memory search over 50 vectors
        trace["retrieved"] = [{"id": r["id"], "score": round(r["score"], 4)} for r in retrieved]

        # 4. The relevance gate: if even the best article is a weak match, don't ask the model.
        top = retrieved[0]["score"]
        threshold = s["relevance_threshold"]
        trace["gate"] = {"top_score": round(top, 4), "threshold": threshold,
                         "passed": threshold is None or top >= threshold}
        if not trace["gate"]["passed"]:
            return reply(200, PROMPT["decline"], "no_relevant_article")

        # 5. Generate, with a schema, a token limit, and the time that is left.
        entries = [r["entry"] for r in retrieved]
        ids = [e["id"] for e in entries]
        messages = [{"role": "system", "content": build_system(entries)}, {"role": "user", "content": question}]
        began = time.perf_counter()
        try:
            result = chat(messages, schema=schema_for(ids), max_tokens=s["max_output_tokens"],
                          timeout=time_left("generate"), url=self.url)
        except LLMError as error:
            timed("generate", began)
            return failed("generate", error)
        timed("generate", began)
        trace.update(prompt_tokens=result["prompt_tokens"], cached_tokens=result["cached_tokens"],
                     output_tokens=result["output_tokens"], done_reason=result["done_reason"])
        model_output = result["content"]

        # 6. Check the output: the agreed shape, cited ids, and values against the cited articles.
        data = _parse(result["content"], ids)
        if data is None:
            return reply(200, PROMPT["decline"], "invalid_output")
        if not data["answered"]:
            return reply(200, PROMPT["decline"], "declined")
        by_id = {e["id"]: e for e in entries}
        cited = [by_id[i]["text"] for i in data["source_ids"]]
        if not data["answer"].strip() or not cited or unsupported_values(data["answer"], cited):
            return reply(200, PROMPT["decline"], "unsupported_answer")

        # 7. The application decides the reply: the checked answer, with its sources.
        return reply(200, data["answer"].strip(), "answered", data["source_ids"])
