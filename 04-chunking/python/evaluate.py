"""
Retrieval for every chunking strategy in data/settings.json, scored against the evidence in data/questions.json.

For each strategy: chunk the five documents, embed every chunk, rank the chunks for each question, and check
whether the retrieved chunks contain the question's evidence. A question's evidence is found when, for at
least one of its alternatives, every passage is inside the text of a single retrieved chunk.

Two ways to decide what "retrieved" means:
  budget   chunks in rank order until the context holds a fixed number of tokens (the fair comparison)
  top k    the first k chunks, however many tokens they hold (the usual metric, shown for contrast)

Writes results-retrieval.json (rankings and scores) and results-chunks.jsonl (every chunk, to compare
the Python and JavaScript chunkers), then prints the tables.
"""
import hashlib
import json
import time
from pathlib import Path

import chunkers
import llm
from tokens import Tokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"


def load() -> tuple[dict, dict[str, str], list[dict]]:
    settings = json.loads((DATA / "settings.json").read_text())
    documents = {p.name: p.read_text() for p in sorted((DATA / "documents").glob("*.md"))}
    questions = json.loads((DATA / "questions.json").read_text())["questions"]
    return settings, documents, questions


class Embedder:
    """Embeds in batches and keeps count of what it cost."""

    def __init__(self, batch_size: int):
        self.batch_size, self.texts, self.tokens, self.calls, self.seconds = batch_size, 0, 0, 0, 0.0

    def __call__(self, texts: list[str], truncate: bool = False) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            started = time.perf_counter()
            batch_vectors, tokens = llm.embed(batch, truncate=truncate)
            self.seconds += time.perf_counter() - started
            self.texts, self.tokens, self.calls = self.texts + len(batch), self.tokens + tokens, self.calls + 1
            vectors.extend(batch_vectors)
        return vectors

    def cost(self) -> dict:
        return {"texts": self.texts, "tokens": self.tokens, "calls": self.calls, "seconds": round(self.seconds, 2)}


def found(question: dict, texts: list[str]) -> bool:
    return any(all(any(p["text"] in t for t in texts) for p in alternative) for alternative in question["evidence"])


def within_budget(ranked: list[int], tokens: list[int], budget: int) -> list[int]:
    """Chunks in rank order; one that doesn't fit what is left of the budget is skipped."""
    chosen, used = [], 0
    for i in ranked:
        if used + tokens[i] <= budget:
            chosen.append(i)
            used += tokens[i]
    return chosen


def embedded_length(text: str, prefix: str) -> int:
    """How many characters of a too-long text the embedding model actually reads with truncate=true:
    the longest prefix, ending at a line break, that it accepts with truncate=false (a binary search)."""
    breaks = [i for i, ch in enumerate(text) if ch == "\n"]
    low, high = 0, len(breaks) - 1  # breaks[low] fits; breaks[high + 1] would not
    while low < high:
        middle = (low + high + 1) // 2
        try:
            llm.embed([prefix + text[:breaks[middle]]])
            low = middle
        except llm.EmbeddingError:
            high = middle - 1
    return breaks[low]


def build_index(strategy: dict, documents: dict[str, str], tokenizer, settings: dict) -> dict:
    prefix = settings["embedding"]["document_prefix"]
    split_cost = Embedder(settings["embedding"]["batch_size"])
    chunks = chunkers.chunk(strategy, documents, tokenizer,
                            embed_units=lambda texts: split_cost([prefix + t for t in texts]))
    index_cost = Embedder(settings["embedding"]["batch_size"])
    notes, cut = [], {}
    if strategy["kind"] == "whole":
        # One document per call, so a document that doesn't fit is identified. It is then embedded the way
        # Ollama does by default (truncate: true): only its beginning.
        vectors = []
        for chunk in chunks:
            try:
                vectors += index_cost([prefix + chunk.text])
            except llm.EmbeddingError as error:
                seen = embedded_length(chunk.text, prefix)
                notes.append(f"{chunk.document}: {error}; embedded with truncate=true, which reads about the first "
                             f"{tokenizer.count(chunk.text[:seen])} of {chunk.tokens} tokens")
                cut[chunk.document] = seen
                vectors += index_cost([prefix + chunk.text], truncate=True)
    else:
        vectors = index_cost([prefix + c.text for c in chunks])
    return {"chunks": chunks, "vectors": vectors, "split_cost": split_cost.cost(), "index_cost": index_cost.cost(),
            "notes": notes, "cut": cut}


def evaluate(settings: dict, documents: dict[str, str], questions: list[dict], tokenizer) -> dict:
    query_embedder = Embedder(settings["embedding"]["batch_size"])
    query_vectors = query_embedder([settings["embedding"]["query_prefix"] + q["question"] for q in questions])
    results = {}
    for strategy in settings["strategies"]:
        index = build_index(strategy, documents, tokenizer, settings)
        chunks, vectors = index["chunks"], index["vectors"]
        tokens = [c.tokens for c in chunks]
        per_question = {}
        for question, query in zip(questions, query_vectors):
            scores = [chunkers.cosine(query, v) for v in vectors]
            ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
            entry = {"ranked": [chunks[i].id for i in ranked[:10]], "scores": [round(scores[i], 4) for i in ranked[:10]],
                     "budget": {}, "top_k": {}}
            for budget in settings["budgets"]:
                chosen = within_budget(ranked, tokens, budget)
                entry["budget"][str(budget)] = {"found": found(question, [chunks[i].text for i in chosen]),
                                                "chunks": len(chosen), "tokens": sum(tokens[i] for i in chosen)}
            for k in settings["recall_k"]:
                top = ranked[:k]
                entry["top_k"][str(k)] = {"found": found(question, [chunks[i].text for i in top]),
                                          "tokens": sum(tokens[i] for i in top)}
            per_question[question["id"]] = entry
        all_texts = [c.text for c in chunks]
        unseen = [q["id"] for q in questions if index["cut"] and not found(
            q, [documents[d][:index["cut"][d]] if d in index["cut"] else documents[d] for d in documents])]
        results[strategy["name"]] = {
            "chunks": len(chunks), "indexed_tokens": sum(tokens),
            # questions whose evidence survives chunking: found if the whole index were retrieved
            "intact": [q["id"] for q in questions if found(q, all_texts)],
            "chunk_tokens": {"min": min(tokens), "median": sorted(tokens)[len(tokens) // 2], "max": max(tokens)},
            "split_cost": index["split_cost"], "index_cost": index["index_cost"], "notes": index["notes"],
            "unseen_by_embedding": unseen,
            "questions": per_question,
            "chunk_list": [{"id": c.id, "document": c.document, "start": c.start, "end": c.end, "prefix": c.prefix,
                            "tokens": c.tokens, "sha256": hashlib.sha256(c.text.encode()).hexdigest()[:16]}
                           for c in chunks],
        }
    return results


def count_found(result: dict, questions: list[dict], mode: str, key: str, **where) -> int:
    chosen = [q for q in questions if all(q[f] == v for f, v in where.items())]
    return sum(result["questions"][q["id"]][mode][key]["found"] for q in chosen)


def ranking(results: dict, settings: dict, questions: list[dict]) -> list[str]:
    """The selection rule: most development questions found at the primary budget; ties: fewer indexed tokens."""
    budget = str(settings["primary_budget"])
    return sorted(results, key=lambda name: (-count_found(results[name], questions, "budget", budget, split="dev"),
                                            results[name]["indexed_tokens"]))


def mean_tokens(result: dict, questions: list[dict], k: str) -> int:
    return round(sum(result["questions"][q["id"]]["top_k"][k]["tokens"] for q in questions) / len(questions))


def report(results: dict, settings: dict, questions: list[dict]) -> None:
    names = [s["name"] for s in settings["strategies"]]
    width = max(len(n) for n in names) + 2
    print("1. The chunks, and what indexing cost (embedding tokens are nomic-embed-text tokens)")
    print(f"{'strategy':<{width}}{'chunks':>7}{'tokens: min':>13}{'median':>8}{'max':>6}{'indexed':>9}"
          f"{'intact':>8}{'embedded texts':>16}{'embedding tokens':>18}")
    for name in names:
        r = results[name]
        texts = r["index_cost"]["texts"] + r["split_cost"]["texts"]
        embedded = r["index_cost"]["tokens"] + r["split_cost"]["tokens"]
        extra = f"  (incl. {r['split_cost']['texts']} sentence windows to find the splits)" if r["split_cost"]["texts"] else ""
        intact = f"{len(r['intact'])}/{len(questions)}"
        print(f"{name:<{width}}{r['chunks']:>7}{r['chunk_tokens']['min']:>13}{r['chunk_tokens']['median']:>8}"
              f"{r['chunk_tokens']['max']:>6}{r['indexed_tokens']:>9}{intact:>8}{texts:>16}{embedded:>18}{extra}")
        for note in r["notes"]:
            print(f"{'':<{width}}  {note}")
        if r["unseen_by_embedding"]:
            print(f"{'':<{width}}  questions whose evidence the embedding never read: {len(r['unseen_by_embedding'])}")

    for split in ("dev", "test"):
        chosen = [q for q in questions if q["split"] == split]
        print(f"\n2. Evidence found, {split} questions ({len(chosen)}): within a context budget, and in the top k chunks")
        header = "".join(f"{f'{b} tokens':>12}" for b in settings["budgets"])
        header += "".join(f"{f'top {k}':>8}{'(tokens)':>9}" for k in settings["recall_k"])
        print(f"{'strategy':<{width}}{header}")
        for name in names:
            r = results[name]
            row = "".join(f"{count_found(r, chosen, 'budget', str(b)):>12}" for b in settings["budgets"])
            row += "".join(f"{count_found(r, chosen, 'top_k', str(k)):>8}{f'({mean_tokens(r, chosen, str(k))})':>9}"
                           for k in settings["recall_k"])
            print(f"{name:<{width}}{row}")

    budget = str(settings["primary_budget"])
    categories = list(dict.fromkeys(q["category"] for q in questions))
    print(f"\n3. Evidence found at {budget} tokens, by kind of question (dev + test)")
    sizes = {c: sum(q["category"] == c for q in questions) for c in categories}
    print(f"{'strategy':<{width}}" + "".join(f"{f'{c} ({sizes[c]})':>14}" for c in categories))
    for name in names:
        print(f"{name:<{width}}" + "".join(
            f"{count_found(results[name], questions, 'budget', budget, category=c):>14}" for c in categories))

    order = ranking(results, settings, questions)
    answer = settings["answer"]
    best = [n for n in order if n != answer["baseline"]][:answer["best"]]
    print(f"\n4. Selection on the dev questions at {budget} tokens: "
          + ", ".join(f"{n} {count_found(results[n], questions, 'budget', budget, split='dev')}" for n in order))
    print(f"   Answer stage: {answer['baseline']} (baseline), " + ", ".join(best))


def main() -> None:
    settings, documents, questions = load()
    tokenizer = Tokenizer()
    try:
        results = evaluate(settings, documents, questions, tokenizer)
    finally:
        llm.unload(llm.embedding_model())
    with open(ROOT / "results-chunks.jsonl", "w") as out:
        for name, r in results.items():
            for c in r["chunk_list"]:
                out.write(json.dumps({"strategy": name, **c}) + "\n")
    saved = {name: {k: v for k, v in r.items() if k != "chunk_list"} for name, r in results.items()}
    (ROOT / "results-retrieval.json").write_text(json.dumps(saved, indent=1) + "\n")
    report(results, settings, questions)


if __name__ == "__main__":
    main()
