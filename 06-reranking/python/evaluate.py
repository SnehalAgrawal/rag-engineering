"""
Reranking: does a second pass that reads the question and each candidate together choose better than
the first stage's order?

  1. First stage (embeddings only): rank every chunk for every question by vector and by RRF (as in 05).
  2. Score every (question, chunk) pair that is in the top 20 of either first stage, once, in a fixed
     order (sorted by question id, then chunk), with the reranker in rerank.py. Each pair is scored once
     per run: scoring the same prompt again while Ollama still has it in its prompt cache can change the
     score slightly (by up to about 0.02 in the probes before the full run). repeat_check.py scores a
     sample again after the model was reloaded, in shuffled order.
  3. For each first stage and candidate count N: candidate recall (is the evidence anywhere in the top N?),
     then evidence found within each budget, in first-stage order and in reranked order. Reranking only
     reorders the N candidates; chunks outside them are never used.

Writes results-scores.jsonl (every score) and results-rerank.json, and prints the tables.
"""
import json
import statistics
import time
from pathlib import Path

import first_stage
import llm
import rerank
from tokens import Tokenizer

ROOT = Path(__file__).resolve().parent
STAGES = ("vector", "rrf")


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


class Experiment:
    def __init__(self, settings, documents, questions, tokenizer):
        self.settings, self.questions = settings, questions
        self.every = [(name, q) for name, qs in questions.items() for q in qs]
        self.by_id = {q["id"]: q for _, q in self.every}
        self.chunks, self.ranked = first_stage.rankings(settings, documents, [q for _, q in self.every], tokenizer)
        self.texts = [c.text for c in self.chunks]
        self.tokens = [c.tokens for c in self.chunks]

    def score_all(self) -> None:
        prompt, r = rerank.load_prompt(), self.settings["reranker"]
        self.prompt_version = prompt["version"]
        self.scores, self.records = {}, []
        for qid, i in first_stage.candidate_pairs(self.ranked, max(self.settings["candidates"])):
            started = time.perf_counter()
            result = rerank.judge(prompt, self.by_id[qid]["question"], self.texts[i], r)
            seconds = time.perf_counter() - started
            self.scores[(qid, i)] = result["score"]
            self.records.append({"question_id": qid, "chunk": self.chunks[i].id, "score": result["score"],
                                 "token": result["token"], "prompt_tokens": result["prompt_tokens"],
                                 "ms": round(seconds * 1000, 1)})

    def order(self, qid: str, stage: str, n: int, reranked: bool) -> list[int]:
        top = self.ranked[qid][stage][:n]
        if not reranked:
            return top
        position = {i: p for p, i in enumerate(top)}
        return sorted(top, key=lambda i: (-self.scores[(qid, i)], position[i]))

    def found_at(self, q: dict, order: list[int], budget: int | None) -> bool:
        chosen = order if budget is None else within_budget(order, self.tokens, budget)
        return found(q, [self.texts[i] for i in chosen])

    def count(self, rows, stage, n, reranked, budget) -> int:
        return sum(self.found_at(q, self.order(q["id"], stage, n, reranked), budget) for _, q in rows)

    def rows(self, set_name=None, split=None, category=None):
        return [(name, q) for name, q in self.every if (set_name is None or name == set_name)
                and (split is None or q["split"] == split) and (category is None or q["category"] == category)]


def report(x: Experiment) -> None:
    s = x.settings
    groups = [("new", "dev"), ("new", "test"), ("04", "dev"), ("04", "test")]
    sizes = {g: len(x.rows(*g)) for g in groups}
    header = "".join(f"{f'{a} {b} ({sizes[(a, b)]})':>14}" for a, b in groups)
    width = 30

    print("1. Candidate recall: the evidence is somewhere in the first stage's top N (no budget)")
    print(f"{'first stage':<{width}}{header}")
    for stage in STAGES:
        for n in s["candidates"]:
            print(f"{f'{stage}, top {n}':<{width}}" + "".join(
                f"{x.count(x.rows(*g), stage, n, False, None):>14}" for g in groups))
    print(f"{'(all 98 chunks)':<{width}}" + "".join(f"{sizes[g]:>14}" for g in groups))

    test = x.rows(split="test")
    print(f"\n2. Evidence found within each budget, all test questions ({len(test)})")
    print(f"{'first stage, candidates, order':<{width}}" + "".join(f"{f'{b} tokens':>13}" for b in s["budgets"]))
    for stage in STAGES:
        full = len(x.chunks)
        print(f"{f'{stage}, all, first stage':<{width}}" + "".join(
            f"{x.count(test, stage, full, False, b):>13}" for b in s["budgets"]))
        for n in s["candidates"]:
            for reranked in (False, True):
                label = f"{stage}, top {n}, {'reranked' if reranked else 'first stage'}"
                print(f"{label:<{width}}" + "".join(f"{x.count(test, stage, n, reranked, b):>13}" for b in s["budgets"]))

    budget = s["primary_budget"]
    print(f"\n3. Evidence found at {budget} tokens, by question group")
    print(f"{'first stage, candidates, order':<{width}}{header}")
    for stage in STAGES:
        for n in s["candidates"]:
            for reranked in (False, True):
                label = f"{stage}, top {n}, {'reranked' if reranked else 'first stage'}"
                print(f"{label:<{width}}" + "".join(f"{x.count(x.rows(*g), stage, n, reranked, budget):>14}"
                                                    for g in groups))

    categories = list(dict.fromkeys(q["category"] for q in x.questions["new"]))
    new_test = {c: x.rows("new", "test", c) for c in categories}
    print(f"\n4. New test questions by kind, at {budget} tokens")
    print(f"{'first stage, candidates, order':<{width}}" + "".join(
        f"{f'{c} ({len(new_test[c])})':>16}" for c in categories))
    for stage in STAGES:
        for n in s["candidates"]:
            for reranked in (False, True):
                label = f"{stage}, top {n}, {'reranked' if reranked else 'first stage'}"
                print(f"{label:<{width}}" + "".join(f"{x.count(new_test[c], stage, n, reranked, budget):>16}"
                                                    for c in categories))

    print(f"\n5. What reranking changed, at 500 and {budget} tokens (all {len(x.every)} questions)")
    for stage in STAGES:
        for n in s["candidates"]:
            for b in (500, budget):
                fixed = [q for _, q in x.every if x.found_at(q, x.order(q["id"], stage, n, True), b)
                         and not x.found_at(q, x.order(q["id"], stage, n, False), b)]
                broken = [q for _, q in x.every if not x.found_at(q, x.order(q["id"], stage, n, True), b)
                          and x.found_at(q, x.order(q["id"], stage, n, False), b)]
                print(f"   {stage}, top {n}, {b} tokens: fixed {len(fixed)}, broke {len(broken)}")
                for label, qs in (("fixed", fixed), ("broke", broken)):
                    for q in qs:
                        print(f"     {label} {q['id']} [{q['split']}, {q['category']}] {q['question']}")

    print("\n6. The three cases from 05: where the chunk with the evidence ranks")
    for qid in ("r20", "r29", "r28"):
        q = x.by_id[qid]
        holders = [i for i, t in enumerate(x.texts) if any(all(p["text"] in t for p in alt) for alt in q["evidence"])]
        print(f"   {qid} {q['question']}")
        for stage in STAGES:
            first = [x.ranked[qid][stage].index(i) + 1 for i in holders]
            n = max(s["candidates"])
            after = [x.order(qid, stage, n, True).index(i) + 1 if i in x.ranked[qid][stage][:n] else None for i in holders]
            scores = [round(x.scores[(qid, i)], 3) if (qid, i) in x.scores else None for i in holders]
            print(f"     {stage}: first-stage rank {first}, rank after reranking the top {n} {after}, score {scores}")

    calls = len(x.records)
    ms = sorted(r["ms"] for r in x.records)
    print("\n7. Cost (timings vary from run to run)")
    print(f"   {calls} pairs scored, one call each: median {ms[len(ms) // 2]:.0f} ms per call, total "
          f"{sum(ms) / 1000:.0f} s")
    for n in s["candidates"]:
        print(f"   reranking the top {n} for one question: {n} calls, about "
              f"{n * statistics.median(ms) / 1000:.1f} s one after another")


def save(x: Experiment) -> None:
    with open(ROOT / "results-scores.jsonl", "w") as out:
        for r in x.records:
            out.write(json.dumps(r) + "\n")
    s = x.settings
    out = {"prompt_version": x.prompt_version, "questions": {}}
    for _, q in x.every:
        out["questions"][q["id"]] = {
            f"{stage}-{n}-{'reranked' if rr else 'first'}": [x.chunks[i].id for i in x.order(q["id"], stage, n, rr)]
            for stage in STAGES for n in s["candidates"] for rr in (False, True)}
    (ROOT / "results-rerank.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    settings, documents, questions = first_stage.load()
    x = Experiment(settings, documents, questions, Tokenizer())
    llm.unload(llm.embedding_model())
    try:
        x.score_all()
    finally:
        llm.unload(llm.chat_model())
    save(x)
    report(x)


if __name__ == "__main__":
    main()
