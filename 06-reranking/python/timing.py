"""
Before the full run: how many pairs there are to score, and what scoring one costs. Scores 50 pairs,
chosen with a fixed seed, twice each, and reports latency, whether the model stayed loaded, whether
the scores repeat, and whether both label sets appear among the alternatives.
"""
import random
import statistics
import time

import first_stage
import llm
import rerank
from tokens import Tokenizer


def main() -> None:
    settings, documents, questions = first_stage.load()
    every = [q for qs in questions.values() for q in qs]
    by_id = {q["id"]: q for q in every}
    chunks, ranked = first_stage.rankings(settings, documents, every, Tokenizer())
    llm.unload(llm.embedding_model())
    for n in settings["candidates"]:
        print(f"top {n}: {len(first_stage.candidate_pairs(ranked, n))} unique (question, chunk) pairs to score")
    pairs = first_stage.candidate_pairs(ranked, max(settings["candidates"]))
    sample = random.Random(1).sample(pairs, 50)
    prompt, r = rerank.load_prompt(), settings["reranker"]
    rows = []
    try:
        for attempt in (1, 2):
            for qid, i in sample:
                started = time.perf_counter()
                result = rerank.judge(prompt, by_id[qid]["question"], chunks[i].text, r)
                rows.append((attempt, qid, i, time.perf_counter() - started, result))
    finally:
        llm.unload(llm.chat_model())
    first = [row for row in rows if row[0] == 1]
    second = {(row[1], row[2]): row[4] for row in rows if row[0] == 2}
    seconds = [row[3] for row in rows]
    print(f"\n{len(rows)} calls: mean {statistics.mean(seconds) * 1000:.0f} ms, median "
          f"{statistics.median(seconds) * 1000:.0f} ms, max {max(seconds) * 1000:.0f} ms (the first includes loading)")
    loads = [row[4]["load_ns"] / 1e6 for row in rows]
    print(f"model load time per call: first {loads[0]:.0f} ms, later median {statistics.median(loads[1:]):.1f} ms")
    print(f"prompt tokens: median {statistics.median(row[4]['prompt_tokens'] for row in rows)}, "
          f"max {max(row[4]['prompt_tokens'] for row in rows)}")
    same = sum(row[4]["score"] == second[(row[1], row[2])]["score"] for row in first)
    print(f"scores identical on repeat: {same} of {len(first)}")
    yes_seen = sum(any(t in row[4]["top"] for t in r["yes"]) for row in first)
    no_seen = sum(any(t in row[4]["top"] for t in r["no"]) for row in first)
    print(f"a yes token among the alternatives: {yes_seen} of {len(first)}; a no token: {no_seen} of {len(first)}")
    print(f"first tokens generated: {sorted({row[4]['token'] for row in first})}")
    scores = sorted(row[4]["score"] for row in first)
    print(f"scores: min {scores[0]:.3f}, median {scores[len(scores) // 2]:.3f}, max {scores[-1]:.3f}")


if __name__ == "__main__":
    main()
