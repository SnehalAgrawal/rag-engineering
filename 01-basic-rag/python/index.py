"""
Index time: embed every help-center article once, and save the vectors.

  python index.py

Writes ../data/index.json. Run it again whenever the articles, the embedding
model, or the prefixes change; rag.py refuses to use an index built from
different articles or settings.
"""
import json
import time

from llm import embed, embedding_model, unload
from rag import ARTICLES, DOCUMENT_PREFIX, INDEX_PATH, QUERY_PREFIX, document_text, fingerprint

start = time.perf_counter()
texts = [document_text(a) for a in ARTICLES]
vectors = embed([DOCUMENT_PREFIX + t for t in texts])
unload(embedding_model())

index = {
    "embedding_model": embedding_model(),
    "document_prefix": DOCUMENT_PREFIX,
    "query_prefix": QUERY_PREFIX,
    "dimensions": len(vectors[0]),
    "articles": [{"id": a["id"], "sha256": fingerprint(t), "vector": v} for a, t, v in zip(ARTICLES, texts, vectors)],
}
INDEX_PATH.write_text(json.dumps(index) + "\n")
print(f"Indexed {len(ARTICLES)} articles with {embedding_model()} ({index['dimensions']} dimensions) "
      f"in {time.perf_counter() - start:.1f} s -> {INDEX_PATH.name}")
