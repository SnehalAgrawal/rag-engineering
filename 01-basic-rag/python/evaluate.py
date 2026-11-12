"""
Evaluates the assistant on the questions in ../data/questions.json.

  python evaluate.py                    retrieval metrics, then all four modes
  python evaluate.py --retrieval-only   only the retrieval metrics (no chat model needed)
  python evaluate.py --regrade FILE     re-grade saved replies without calling any model

Part 1 measures retrieval alone: for each answerable question, does an article
that answers it appear in the top K? (Recall@K; here every question is labeled
with the article(s) that answer it, so it is the hit rate.)

Part 2 sends every question through the application four ways:
  all, default context   the whole help center in the prompt; Ollama picks the context length
  all, 8K context        the whole help center, with a context large enough to hold it
  rag, top 1             only the best-matching article
  rag, top 3             only the 3 best-matching articles
Answerable and not-in-docs questions are scored separately.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from llm import chat_model, embedding_model, unload

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "questions.json"
MODES = [
    {"name": "all, default context", "mode": "all", "num_ctx": None},
    {"name": "all, 8K context", "mode": "all", "num_ctx": 8192},
    {"name": "rag, top 1", "mode": "rag", "k": 1},
    {"name": "rag, top 3", "mode": "rag", "k": 3},
]


def grade(record: dict, question: dict) -> str:
    """What the customer saw: the key fact, an answer without it, or the decline message."""
    shown = record["log"]["outcome"] == "answered"
    if record["kind"] == "answerable":
        if not shown:
            return "declined"
        has_fact = any(re.search(p, record["message"], re.IGNORECASE) for p in question["patterns"])
        return "key fact" if has_fact else "no key fact"
    return "answered" if shown else "declined"


def short(text: str, width: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def report(records: list[dict], questions: dict) -> None:
    print("\n2. The whole application, four ways\n")
    print(f"{'':<22}{'answerable questions':^36}{'not in docs':^16}")
    print(f"{'mode':<22}{'key fact':>12}{'no key fact':>13}{'declined':>11}{'declined':>14}{'prompt tokens':>16}")
    for mode in MODES:
        rows = [r for r in records if r["mode_name"] == mode["name"]]
        grades = [(r, grade(r, questions[r["question_id"]])) for r in rows]
        answerable = [g for r, g in grades if r["kind"] == "answerable"]
        unanswerable = [g for r, g in grades if r["kind"] == "not_in_docs"]
        tokens = statistics.median_low(r["log"]["prompt_tokens"] for r in rows if r["log"]["prompt_tokens"] is not None)
        print(f"{mode['name']:<22}{answerable.count('key fact'):>9}/{len(answerable):<2}{answerable.count('no key fact'):>13}"
              f"{answerable.count('declined'):>11}{unanswerable.count('declined'):>11}/{len(unanswerable):<2}{tokens:>16,.0f}")
    print("\nkey fact: the reply shown to the customer contains the question's key fact (a pattern check).")
    print("no key fact: an answer was shown, but without the key fact.")
    print("declined: the customer got the decline message (the model declined, or a check in code rejected its answer).")
    print("not in docs: the help center doesn't answer these, so declining is the right behavior.")
    print("prompt tokens: median per question (the lower middle value), as counted by the model server.")
    print("A pattern can't tell a right answer from a wrong one that mentions the same fact: read the replies too.")

    print("\nSpeed in this run (latencies vary between runs and machines):\n")
    print(f"{'mode':<22}{'reused from cache':>19}{'first request':>15}{'median':>10}")
    for mode in MODES:
        rows = [r for r in records if r["mode_name"] == mode["name"]]
        cached = [r["log"]["cached_tokens"] for r in rows if r["log"].get("cached_tokens") is not None]
        latencies = [r["log"]["latency_ms"] for r in rows]
        reused = f"{statistics.median_low(cached):,}" if cached else "n/a"
        print(f"{mode['name']:<22}{reused:>19}{latencies[0] / 1000:>13.2f} s{statistics.median_low(latencies) / 1000:>8.2f} s")
    print("\nreused from cache: median prompt tokens the server reused from earlier requests (prompt_eval_cached_count).")
    print("first request: each mode starts with the chat model unloaded, so this includes loading it.")

    print("\nEvery reply without the key fact, and every answer to a question the help center doesn't answer:")
    for mode in MODES:
        for r in (r for r in records if r["mode_name"] == mode["name"]):
            g = grade(r, questions[r["question_id"]])
            if g == "key fact" or (r["kind"] == "not_in_docs" and g == "declined"):
                continue
            where = f" retrieved {r['log']['retrieved']}" if r["log"]["retrieved"] else ""
            print(f"  [{mode['name']}] {r['question_id']} {g}: {r['question']!r}{where}")
            print(f"      outcome {r['log']['outcome']}; shown: {short(r['message'])}")
            if r["log"]["outcome"] != "answered" and r["model_output"]:
                print(f"      model output: {short(r['model_output'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--regrade", metavar="FILE")
    args = parser.parse_args()

    data = json.loads(QUESTIONS_FILE.read_text())
    questions = {q["id"]: {**q, "kind": "answerable"} for q in data["answerable"]}
    questions.update({q["id"]: {**q, "kind": "not_in_docs"} for q in data["not_in_docs"]})

    if args.regrade:
        with open(args.regrade) as f:
            records = [json.loads(line) for line in f]
        report(records, questions)
        return

    from rag import ARTICLES, answer, retrieve  # imported here so --regrade needs no index

    unload(chat_model())
    unload(embedding_model())
    print(f"Chat model: {chat_model()} | embedding model: {embedding_model()} | {len(ARTICLES)} articles | "
          f"{len(data['answerable'])} answerable questions, {len(data['not_in_docs'])} not in docs\n")

    # Part 1: retrieval alone.
    print("1. Retrieval: does an article that answers the question appear in the top K?\n")
    ranks, top_scores = [], {}
    for q in data["answerable"] + data["not_in_docs"]:
        results = retrieve(q["question"], 5)
        top_scores[q["id"]] = results[0]["score"]
        if q["id"] in {a["id"] for a in data["answerable"]}:
            ids = [r["id"] for r in results]
            rank = next((i + 1 for i, article_id in enumerate(ids) if article_id in q["relevant"]), None)
            ranks.append(rank)
            if rank != 1:
                top3 = ", ".join(f"{r['id']} {r['score']:.4f}" for r in results[:3])
                print(f"  {q['id']} {q['question']!r}: relevant {q['relevant']}, rank {rank or '>5'}; top 3: {top3}")
    for k in (1, 3, 5):
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  Recall@{k} = {hits}/{len(ranks)} = {hits / len(ranks):.2f}")
    answerable_top = [top_scores[q["id"]] for q in data["answerable"]]
    unanswerable_top = [top_scores[q["id"]] for q in data["not_in_docs"]]
    print(f"  Top-1 similarity: answerable {min(answerable_top):.4f} to {max(answerable_top):.4f}, "
          f"not in docs {min(unanswerable_top):.4f} to {max(unanswerable_top):.4f}")
    if args.retrieval_only:
        unload(embedding_model())
        return

    # Part 2: the whole application, four ways. Each mode starts from an unloaded chat model.
    records = []
    for mode in MODES:
        unload(chat_model())
        for q in data["answerable"] + data["not_in_docs"]:
            result = answer(q["question"], mode=mode["mode"], k=mode.get("k", 3), num_ctx=mode.get("num_ctx"))
            records.append({"mode_name": mode["name"], "question_id": q["id"], "question": q["question"],
                            "kind": questions[q["id"]]["kind"], "message": result["message"],
                            "sources": result["sources"], "retrieved": result["retrieved"],
                            "model_output": result["model_output"], "log": result["log"]})
            print(".", end="", file=sys.stderr, flush=True)
    unload(chat_model())
    unload(embedding_model())
    print(file=sys.stderr)

    out = f"results-{chat_model().replace(':', '_')}.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(records, questions)
    print(f"\nEvery reply, with its log record, was saved to {out}.")


if __name__ == "__main__":
    main()
