"""
Check tokens.py against the model server: for each text, compare our count with the number of prompt tokens
Ollama reports for the same text sent as a raw prompt (no chat template).

Ollama adds the begin-of-text token to a raw prompt, so its count should be ours plus one. The texts are the
five documents, every question, and pieces of the documents cut at arbitrary points.
"""
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

from tokens import Tokenizer

load_dotenv()
DATA = Path(__file__).resolve().parent.parent / "data"


def server_count(text: str, base_url: str, model: str) -> int:
    body = {"model": model, "prompt": text, "raw": True, "stream": False,
            "options": {"num_predict": 1, "temperature": 0, "seed": 1}}
    data = requests.post(f"{base_url}/api/generate", json=body, timeout=120).json()
    return data["prompt_eval_count"]  # includes tokens served from the prompt cache (prompt_eval_cached_count)


def main() -> None:
    import os
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    tokenizer = Tokenizer()
    documents = [p.read_text() for p in sorted((DATA / "documents").glob("*.md"))]
    questions = [q["question"] for q in json.loads((DATA / "questions.json").read_text())["questions"]]
    pieces = [doc[start:start + length] for doc in documents for start, length in ((0, 700), (1234, 999), (3001, 1777))]
    texts = documents + questions + pieces
    mismatches = 0
    try:
        for text in texts:
            ours, theirs = tokenizer.count(text), server_count(text, base_url, model)
            if theirs != ours + 1:
                mismatches += 1
                print(f"MISMATCH ours={ours} server={theirs}: {text[:60]!r}")
    finally:
        requests.post(f"{base_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    print(f"{len(texts)} texts ({len(documents)} documents, {len(questions)} questions, {len(pieces)} pieces), "
          f"{sum(tokenizer.count(t) for t in texts)} tokens: {mismatches} mismatches")


if __name__ == "__main__":
    main()
