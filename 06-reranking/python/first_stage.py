"""
The first stage, as in 05-hybrid-search: chunk the documents, embed the chunks and the questions, and rank
every chunk for every question by vector similarity and by RRF of vector and BM25. Embeddings only.
"""
import json
from pathlib import Path

import chunkers
import llm
import search

DATA = Path(__file__).resolve().parent.parent / "data"
SETS = {"new": "questions.json", "04": "questions-04.json"}


def load() -> tuple[dict, dict[str, str], dict[str, list[dict]]]:
    settings = json.loads((DATA / "settings.json").read_text())
    documents = {p.name: p.read_text() for p in sorted((DATA / "documents").glob("*.md"))}
    questions = {name: json.loads((DATA / f).read_text())["questions"] for name, f in SETS.items()}
    return settings, documents, questions


def embed_all(texts: list[str], batch_size: int) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), batch_size):
        vectors += llm.embed(texts[i:i + batch_size])[0]
    return vectors


def rankings(settings: dict, documents: dict[str, str], questions: list[dict], tokenizer) -> tuple[list, dict]:
    """The chunks, and for each question id: {"vector": [chunk indexes], "rrf": [chunk indexes]}, best first."""
    chunks = chunkers.chunk(settings["chunking"], documents, tokenizer)
    texts = [c.text for c in chunks]
    e = settings["embedding"]
    vectors = embed_all([e["document_prefix"] + t for t in texts], e["batch_size"])
    queries = embed_all([e["query_prefix"] + q["question"] for q in questions], e["batch_size"])
    bm25 = search.BM25(texts, settings["bm25"]["analyzer"], settings["bm25"]["k1"], settings["bm25"]["b"])
    result = {}
    for q, query in zip(questions, queries):
        vector = search.rank([chunkers.cosine(query, v) for v in vectors])
        keyword = search.rank(bm25.scores(q["question"]), matched_only=True)
        rrf = search.rank(search.rrf([vector, keyword], settings["rrf"]["k"], len(chunks)))
        result[q["id"]] = {"vector": vector, "rrf": rrf}
    return chunks, result


def candidate_pairs(ranked: dict, n: int) -> list[tuple[str, int]]:
    """Every (question id, chunk index) among the top n of any first stage, in a fixed order."""
    pairs = set()
    for qid, by_stage in ranked.items():
        for order in by_stage.values():
            pairs.update((qid, i) for i in order[:n])
    return sorted(pairs)
