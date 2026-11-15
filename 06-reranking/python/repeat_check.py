"""
After the full run: score a seeded random sample of the scored pairs again, in shuffled order, with the
model freshly loaded, and compare with results-scores.jsonl: are the scores the same, and could any
difference change which chunks a question gets?
"""
import json
import random
from pathlib import Path

import chunkers
import first_stage
import llm
import rerank
from tokens import Tokenizer

ROOT = Path(__file__).resolve().parent


def main() -> None:
    settings, documents, questions = first_stage.load()
    every = [q for qs in questions.values() for q in qs]
    by_id = {q["id"]: q for q in every}
    chunks = chunkers.chunk(settings["chunking"], documents, Tokenizer())
    by_chunk = {c.id: c for c in chunks}
    saved = [json.loads(line) for line in (ROOT / "results-scores.jsonl").read_text().splitlines()]
    check = settings["repeat_check"]
    sample = random.Random(check["seed"]).sample(saved, check["pairs"])
    prompt, r = rerank.load_prompt(), settings["reranker"]
    try:
        again = [rerank.judge(prompt, by_id[s["question_id"]]["question"], by_chunk[s["chunk"]].text, r)["score"]
                 for s in sample]
    finally:
        llm.unload(llm.chat_model())
    diffs = sorted(abs(a - s["score"]) for a, s in zip(again, sample))
    print(f"{len(sample)} pairs scored again in shuffled order (seed {check['seed']})")
    print(f"identical: {sum(d == 0 for d in diffs)}; differences: median {diffs[len(diffs) // 2]:.4f}, "
          f"max {diffs[-1]:.4f}")
    by_question: dict[str, list[float]] = {}
    for s in saved:
        by_question.setdefault(s["question_id"], []).append(s["score"])
    closer = 0
    for a, s in zip(again, sample):
        others = [v for v in by_question[s["question_id"]] if v != s["score"]]
        gap = min(abs(v - s["score"]) for v in others)
        closer += abs(a - s["score"]) >= gap
    print(f"pairs whose change is at least the gap to the nearest other score for the same question: {closer}")


if __name__ == "__main__":
    main()
