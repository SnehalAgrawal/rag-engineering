"""
The index as a versioned build artifact.

An index is a snapshot: for every article, the text that was embedded, its SHA-256, and its vector,
plus the settings the vectors depend on (embedding model and prefixes). Its version is a hash of
those settings and of every article's fingerprint, so the same content always gets the same version.

  build()     re-embeds only the articles that are new or changed, reuses the other vectors, and
              drops articles that were removed. Different settings mean a full rebuild.
  write()     writes to a temporary file and renames it over the old index. A reader sees either the
              old index or the new one, never half of one.
  staleness() counts the articles that changed in the help center since the index was built.
"""
import hashlib
import json
import os
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
HELP_CENTER_PATH = DATA / "help_center.json"
INDEX_PATH = DATA / "index.json"

# nomic-embed-text was trained with task prefixes: one for questions, one for documents.
QUERY_PREFIX = os.environ.get("QUERY_PREFIX", "search_query: ")
DOCUMENT_PREFIX = os.environ.get("DOCUMENT_PREFIX", "search_document: ")


def document_text(article: dict) -> str:
    """What gets embedded for an article (without the prefix)."""
    return f"{article['title']}\n{article['text']}"


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def settings(embedding_model: str) -> dict:
    return {"embedding_model": embedding_model, "document_prefix": DOCUMENT_PREFIX, "query_prefix": QUERY_PREFIX}


def version_of(index_settings: dict, entries: list[dict]) -> str:
    lines = [index_settings["embedding_model"], index_settings["document_prefix"], index_settings["query_prefix"]]
    lines += [f"{e['id']} {e['sha256']}" for e in entries]
    return fingerprint("\n".join(lines))[:12]


def read_articles(path: Path = HELP_CENTER_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())["articles"]


def read(path: Path = INDEX_PATH) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def build(articles: list[dict], previous: dict | None, index_settings: dict, embed_documents) -> tuple[dict, dict]:
    """A new index for these articles, and what it took: {"embedded", "reused", "removed"}.

    embed_documents(texts) returns one vector per text; it is called once, only for new or changed articles.
    """
    reusable = {}
    if previous and all(previous[key] == value for key, value in index_settings.items()):
        reusable = {e["sha256"]: e["vector"] for e in previous["entries"]}
    entries, to_embed = [], []
    for article in articles:
        text = document_text(article)
        entry = {"id": article["id"], "title": article["title"], "text": article["text"], "sha256": fingerprint(text)}
        if entry["sha256"] in reusable:
            entry["vector"] = reusable[entry["sha256"]]
        else:
            to_embed.append((entry, text))
        entries.append(entry)
    if to_embed:
        vectors = embed_documents([DOCUMENT_PREFIX + text for _, text in to_embed])
        for (entry, _), vector in zip(to_embed, vectors):
            entry["vector"] = vector
    current_ids = {a["id"] for a in articles}
    removed = sum(1 for e in previous["entries"] if e["id"] not in current_ids) if previous else 0
    index = {**index_settings, "version": version_of(index_settings, entries), "dimensions": len(entries[0]["vector"]),
             "entries": entries}
    return index, {"embedded": len(to_embed), "reused": len(entries) - len(to_embed), "removed": removed}


def write(index: dict, path: Path = INDEX_PATH) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(index) + "\n")
    os.replace(temporary, path)  # atomic on the same file system


def staleness(index: dict, articles: list[dict]) -> int:
    """How many articles were added, changed, or removed in the help center since this index was built."""
    indexed = {e["id"]: e["sha256"] for e in index["entries"]}
    current = {a["id"]: fingerprint(document_text(a)) for a in articles}
    return sum(1 for article_id in indexed.keys() | current.keys() if indexed.get(article_id) != current.get(article_id))
