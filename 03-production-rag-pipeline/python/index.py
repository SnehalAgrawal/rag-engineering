"""
Index time: bring the index up to date with the help center.

  python index.py

Re-embeds only the articles that are new or changed since the last build, reuses the vectors of the
others, and drops removed articles. A different embedding model or different prefixes mean a full
rebuild. The new index replaces ../data/index.json in one step (index_store.write), so an assistant
that is serving keeps using the old version until the new one is complete.
"""
import time

import index_store
from llm import embed, embedding_model, unload


def update(help_center_path=index_store.HELP_CENTER_PATH, index_path=index_store.INDEX_PATH) -> tuple[dict, dict]:
    previous = index_store.read(index_path)
    index, stats = index_store.build(index_store.read_articles(help_center_path), previous,
                                     index_store.settings(embedding_model()),
                                     lambda texts: embed(texts, timeout=300))
    if previous is None or index["version"] != previous["version"]:
        index_store.write(index, index_path)
    stats["previous_version"] = previous["version"] if previous else None
    return index, stats


def describe(index: dict, stats: dict) -> str:
    change = "unchanged" if stats["previous_version"] == index["version"] else f"was {stats['previous_version'] or 'none'}"
    return (f"embedded {stats['embedded']}, reused {stats['reused']}, removed {stats['removed']} "
            f"-> version {index['version']} ({change})")


if __name__ == "__main__":
    start = time.perf_counter()
    index, stats = update()
    unload(embedding_model())
    print(f"{len(index['entries'])} articles, {embedding_model()} ({index['dimensions']} dimensions): "
          f"{describe(index, stats)}, {time.perf_counter() - start:.1f} s")
