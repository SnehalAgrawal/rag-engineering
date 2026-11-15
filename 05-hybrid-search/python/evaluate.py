"""
Vector, keyword, and hybrid retrieval on the same chunks and the same budget, scored against the evidence
in data/questions.json (40 new questions) and data/questions-04.json (the 60 from 04-chunking).

Steps:
  1. Chunk the five documents with structure-headings (04's best) and embed every chunk.
  2. Build a BM25 index with each analyzer.
  3. For every question: rank by vector, by keyword (both analyzers), and by the two hybrids.
  4. On the development questions only: pick the analyzer for the hybrids, and the fusion weight.
  5. Fill the 1,000-token budget from each ranking, and check the evidence.

Writes results-retrieval.json and prints the tables. Timings (the last section) vary from run to run.
"""
import json
import time
from pathlib import Path

import chunkers
import llm
import search
from tokens import Tokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
SETS = {"new": "questions.json", "04": "questions-04.json"}


def load() -> tuple[dict, dict[str, str], dict[str, list[dict]]]:
    settings = json.loads((DATA / "settings.json").read_text())
    documents = {p.name: p.read_text() for p in sorted((DATA / "documents").glob("*.md"))}
    questions = {name: json.loads((DATA / f).read_text())["questions"] for name, f in SETS.items()}
    return settings, documents, questions


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


def embed_all(texts: list[str], batch_size: int) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), batch_size):
        vectors += llm.embed(texts[i:i + batch_size])[0]
    return vectors


class Run:
    def __init__(self, settings: dict, documents: dict[str, str], questions: dict[str, list[dict]], tokenizer):
        self.settings, self.questions = settings, questions
        self.chunks = chunkers.chunk(settings["chunking"], documents, tokenizer)
        self.texts = [c.text for c in self.chunks]
        self.tokens = [c.tokens for c in self.chunks]
        self.timing = {}

        started = time.perf_counter()
        prefix = settings["embedding"]["document_prefix"]
        self.vectors = embed_all([prefix + t for t in self.texts], settings["embedding"]["batch_size"])
        self.timing["embed chunks (s)"] = time.perf_counter() - started

        bm25 = settings["bm25"]
        self.indexes = {}
        for analyzer in bm25["analyzers"]:
            started = time.perf_counter()
            self.indexes[analyzer] = search.BM25(self.texts, analyzer, bm25["k1"], bm25["b"])
            self.timing[f"build BM25 {analyzer} (ms)"] = (time.perf_counter() - started) * 1000

        # Per question: the raw scores of each method. Questions are embedded one at a time, as a live
        # system would, so the time per query is measured.
        self.scores: dict[str, dict] = {}
        query_seconds, keyword_seconds = [], []
        for q in [q for qs in questions.values() for q in qs]:
            started = time.perf_counter()
            query = llm.embed([settings["embedding"]["query_prefix"] + q["question"]])[0][0]
            query_seconds.append(time.perf_counter() - started)
            entry = {"vector": [chunkers.cosine(query, v) for v in self.vectors]}
            for analyzer, index in self.indexes.items():
                started = time.perf_counter()
                entry[analyzer] = index.scores(q["question"])
                keyword_seconds.append(time.perf_counter() - started)
            self.scores[q["id"]] = entry
        self.timing["vector query: embed the question (ms, median)"] = sorted(query_seconds)[len(query_seconds) // 2] * 1000
        self.timing["keyword query: BM25 over all chunks (ms, median)"] = (
            sorted(keyword_seconds)[len(keyword_seconds) // 2] * 1000)

    def ranking(self, qid: str, method: str) -> list[int]:
        s = self.scores[qid]
        if method == "vector":
            return search.rank(s["vector"])
        if method.startswith("keyword-"):
            return search.rank(s[method.removeprefix("keyword-")], matched_only=True)
        if method == "rrf":
            lists = [search.rank(s["vector"]), search.rank(s[self.analyzer], matched_only=True)]
            return search.rank(search.rrf(lists, self.settings["rrf"]["k"], len(self.chunks)))
        if method.startswith("weighted-"):
            alpha = float(method.removeprefix("weighted-"))
            return search.rank(search.weighted(s["vector"], s[self.analyzer], alpha), matched_only=True)
        raise ValueError(method)

    def found_at(self, q: dict, method: str, budget: int) -> bool:
        chosen = within_budget(self.ranking(q["id"], method), self.tokens, budget)
        return found(q, [self.texts[i] for i in chosen])

    def count(self, method: str, budget: int, set_name: str, **where) -> int:
        return sum(self.found_at(q, method, budget) for q in self.questions[set_name]
                   if all(q[f] == v for f, v in where.items()))

    def dev_count(self, method: str) -> int:
        budget = self.settings["primary_budget"]
        return sum(self.count(method, budget, name, split="dev") for name in SETS)

    def choose(self) -> None:
        """The two choices made on development questions only (rules in data/settings.json)."""
        words, identifiers = self.dev_count("keyword-words"), self.dev_count("keyword-identifiers")
        self.analyzer = "identifiers" if identifiers > words else "words"
        self.sweep = {f"{a:.1f}": self.dev_count(f"weighted-{a:.1f}") for a in self.settings["weighted"]["alphas"]}
        best = max(self.sweep.values())
        self.alpha = min((a for a, n in self.sweep.items() if n == best), key=lambda a: (abs(float(a) - 0.5), float(a)))


def report(run: Run) -> None:
    s, budget = run.settings, run.settings["primary_budget"]
    n = {name: {split: sum(q["split"] == split for q in qs) for split in ("dev", "test")}
         for name, qs in run.questions.items()}
    methods = ["vector", "keyword-words", "keyword-identifiers", "rrf", f"weighted-{run.alpha}"]
    width = max(len(m) for m in methods) + 2

    print("1. How the analyzers turn identifiers into tokens")
    for example in ("SSO-104", "cursor_expired", "X-RateLimit-Remaining", "https://api.example.com/scim/v2"):
        print(f"   {example:<34}words: {search.analyze(example, 'words')}")
        print(f"   {'':<34}identifiers: {search.analyze(example, 'identifiers')}")
    print(f"\n   {len(run.chunks)} chunks ({s['chunking']['name']}), {sum(run.tokens)} tokens. BM25 index: "
          + ", ".join(f"{a} {len(i.idf)} distinct tokens" for a, i in run.indexes.items()))

    print(f"\n2. Choices made on the development questions ({n['new']['dev']} new + {n['04']['dev']} from 04), "
          f"at {budget} tokens")
    print(f"   keyword analyzer: words {run.dev_count('keyword-words')}, identifiers "
          f"{run.dev_count('keyword-identifiers')} -> hybrids use '{run.analyzer}'")
    print("   weighted fusion, alpha = weight of the vector score:")
    print("   alpha " + "".join(f"{a:>5}" for a in run.sweep))
    print("   found " + "".join(f"{c:>5}" for c in run.sweep.values()) + f"   -> alpha {run.alpha}")

    print(f"\n3. Evidence found at {budget} tokens")
    header = "".join(f"{f'{name} {split} ({n[name][split]})':>14}" for name in ("new", "04") for split in ("dev", "test"))
    print(f"{'method':<{width}}{header}")
    for m in methods:
        print(f"{m:<{width}}" + "".join(f"{run.count(m, budget, name, split=split):>14}"
                                        for name in ("new", "04") for split in ("dev", "test")))

    categories = list(dict.fromkeys(q["category"] for q in run.questions["new"]))
    test = [q for q in run.questions["new"] if q["split"] == "test"]
    sizes = {c: sum(q["category"] == c for q in test) for c in categories}
    print(f"\n4. New test questions by kind, at {budget} tokens")
    print(f"{'method':<{width}}" + "".join(f"{f'{c} ({sizes[c]})':>16}" for c in categories))
    for m in methods:
        print(f"{m:<{width}}" + "".join(f"{run.count(m, budget, 'new', split='test', category=c):>16}"
                                        for c in categories))

    total = sum(n[name]["test"] for name in SETS)
    print(f"\n5. All test questions ({total}), by budget")
    print(f"{'method':<{width}}" + "".join(f"{f'{b} tokens':>13}" for b in s["budgets"]))
    for m in methods:
        print(f"{m:<{width}}" + "".join(f"{sum(run.count(m, b, name, split='test') for name in SETS):>13}"
                                        for b in s["budgets"]))

    keyword = f"keyword-{run.analyzer}"
    every = [(name, q) for name in SETS for q in run.questions[name]]
    print(f"\n6. Questions only one method found, at {budget} tokens (all {len(every)} questions)")
    for a, b in (("vector", keyword), (keyword, "vector")):
        only = [(name, q) for name, q in every if run.found_at(q, a, budget) and not run.found_at(q, b, budget)]
        print(f"   {a} only ({len(only)}):")
        for name, q in only:
            print(f"     {q['id']} [{name} {q['split']}, {q['category']}] {q['question']}")
    for m in ("rrf", f"weighted-{run.alpha}"):
        lost = [(name, q) for name, q in every if not run.found_at(q, m, budget)
                and (run.found_at(q, "vector", budget) or run.found_at(q, keyword, budget))]
        print(f"   found by vector or {keyword}, but not by {m} ({len(lost)}):")
        for name, q in lost:
            print(f"     {q['id']} [{name} {q['split']}, {q['category']}] {q['question']}")
    neither = [(name, q) for name, q in every if not any(run.found_at(q, m, budget) for m in methods)]
    print(f"   found by no method ({len(neither)}):")
    for name, q in neither:
        print(f"     {q['id']} [{name} {q['split']}, {q['category']}] {q['question']}")

    print("\n7. Timings (vary from run to run)")
    for key, value in run.timing.items():
        print(f"   {key}: {value:.2f}")


def save(run: Run) -> None:
    budget = run.settings["primary_budget"]
    methods = ["vector", "keyword-words", "keyword-identifiers", "rrf", f"weighted-{run.alpha}"]
    out = {"analyzer": run.analyzer, "alpha": run.alpha, "sweep": run.sweep, "questions": {}}
    for qs in run.questions.values():
        for q in qs:
            out["questions"][q["id"]] = {
                m: {"ranked": [run.chunks[i].id for i in run.ranking(q["id"], m)[:10]],
                    "found": {str(b): run.found_at(q, m, b) for b in run.settings["budgets"]},
                    "context": [run.chunks[i].id for i in within_budget(run.ranking(q["id"], m), run.tokens, budget)]}
                for m in methods}
    (ROOT / "results-retrieval.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    settings, documents, questions = load()
    try:
        run = Run(settings, documents, questions, Tokenizer())
    finally:
        llm.unload(llm.embedding_model())
    run.choose()
    save(run)
    report(run)


if __name__ == "__main__":
    main()
