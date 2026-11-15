"""
Which of the label variants were among the 20 alternatives the reranker returned, for every scored pair?

The score adds up the probability of the "yes" variants and of the "no" variants that appear among the
top 20 alternatives at the first position. A variant that isn't among them is left out, so this script
rescores every pair in results-scores.jsonl (in the same order, with the model freshly loaded), records
which variants were present and the lowest listed log-probability (no missing variant can be more likely
than that), and checks that every score matches the saved one. Writes results-coverage.jsonl: for each
pair, the variants present and the largest amount the missing ones could move its score.
"""
import json
import math
from pathlib import Path

import chunkers
import first_stage
import llm
import rerank
from tokens import Tokenizer

ROOT = Path(__file__).resolve().parent


def main() -> None:
    settings, documents, questions = first_stage.load()
    by_id = {q["id"]: q for qs in questions.values() for q in qs}
    by_chunk = {c.id: c for c in chunkers.chunk(settings["chunking"], documents, Tokenizer())}
    saved = [json.loads(line) for line in (ROOT / "results-scores.jsonl").read_text().splitlines()]
    prompt, r = rerank.load_prompt(), settings["reranker"]
    labels = r["yes"] + r["no"]
    rows = []
    try:
        for s in saved:
            result = rerank.judge(prompt, by_id[s["question_id"]]["question"], by_chunk[s["chunk"]].text, r)
            rows.append({"saved": s["score"], "score": result["score"], "present": [t for t in labels if t in result["top"]],
                         "logprobs": {t: result["top"][t] for t in labels if t in result["top"]},
                         "floor": min(result["top"].values()), "listed": len(result["top"])})
    finally:
        llm.unload(llm.chat_model())
    print(f"{len(rows)} pairs rescored in the original order, model freshly loaded")
    print(f"scores identical to the saved run: {sum(x['score'] == x['saved'] for x in rows)}")
    print(f"alternatives listed per pair: {sorted({x['listed'] for x in rows})}")
    for t in labels:
        print(f"   {t!r:8} present in {sum(t in x['present'] for x in rows)} of {len(rows)}")
    all_present = sum(len(x["present"]) == len(labels) for x in rows)
    print(f"all {len(labels)} variants present: {all_present} of {len(rows)}")
    missing = [x for x in rows if len(x["present"]) < len(labels)]
    if missing:
        # A missing variant is at most as likely as the lowest listed alternative. Adding that much for each
        # missing variant to its label bounds how far the score could move: log(1 + missing / counted).
        shifts = []
        for x in missing:
            shift = 0.0
            for side in ("yes", "no"):
                absent = sum(t not in x["present"] for t in r[side])
                counted = [x["logprobs"][t] for t in r[side] if t in x["logprobs"]]
                if absent and counted:
                    shift = max(shift, math.log1p(absent * math.exp(x["floor"]) / sum(math.exp(v) for v in counted)))
                elif absent:
                    shift = math.inf
            shifts.append(shift)
            x["bound"] = shift
        with open(ROOT / "results-coverage.jsonl", "w") as out:
            for s, x in zip(saved, rows):
                out.write(json.dumps({"question_id": s["question_id"], "chunk": s["chunk"], "present": x["present"],
                                      "bound": x.get("bound", 0.0)}) + "\n")
        shifts.sort()
        print(f"largest possible score shift from missing variants: median {shifts[len(shifts) // 2]:.2e}, "
              f"max {shifts[-1]:.2e}")


if __name__ == "__main__":
    main()
