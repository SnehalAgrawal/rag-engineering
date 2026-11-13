"""
Evaluates the pipeline.

  python evaluate.py                    parts 1-3
  python evaluate.py --retrieval-only   parts 1-2 (no chat model needed)
  python evaluate.py --regrade FILE     part 3's report from saved replies, without calling any model

Part 1: retrieval on the 30 test questions (Recall@K, as in 01-basic-rag).
Part 2: the relevance threshold, chosen on the development questions only. The rule: the highest
        threshold, rounded down to 0.01, that blocks at most 1% of the answerable dev questions.
Part 3: the 30 test questions through both profiles (basic = 01-basic-rag's behavior; production =
        deadline and relevance gate). Replies are graded by hand labels (../data/hand_labels.json),
        each stored with a hash of the exact reply; a reply without a label is reported as unlabeled.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from llm import chat_model, embedding_model, unload

DATA = Path(__file__).resolve().parent.parent / "data"
TEST = json.loads((DATA / "questions.json").read_text())
DEV = json.loads((DATA / "dev_questions.json").read_text())
SETTINGS = json.loads((DATA / "settings.json").read_text())
LABELS = json.loads((DATA / "hand_labels.json").read_text())["labels"]
PROFILES = ["basic", "production"]
SWEEP = [0.55, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70]
MAX_BLOCKED_SHARE = 0.01


def reply_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def label(record: dict) -> str | None:
    """What the customer got. Declines need no label; answers to answerable questions are labeled by hand."""
    if record["trace"]["outcome"] != "answered":
        return "declined"
    if record["kind"] == "not_in_docs":
        return "answered"
    return LABELS.get(f"{record['question_id']} | {reply_hash(record['message'])}")


def short(text: str, width: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def top_scores(assistant, questions: list[dict]) -> dict:
    return {q["id"]: assistant.retrieve(q["question"], 5, timeout=60) for q in questions}


def part1(assistant) -> None:
    print("1. Retrieval on the test questions: does an article that answers the question appear in the top K?\n")
    results = top_scores(assistant, TEST["answerable"] + TEST["not_in_docs"])
    ranks = []
    for q in TEST["answerable"]:
        ids = [r["id"] for r in results[q["id"]]]
        rank = next((i + 1 for i, article_id in enumerate(ids) if article_id in q["relevant"]), None)
        ranks.append(rank)
        if rank != 1:
            top3 = ", ".join(f"{r['id']} {r['score']:.4f}" for r in results[q["id"]][:3])
            print(f"  {q['id']} {q['question']!r}: relevant {q['relevant']}, rank {rank or '>5'}; top 3: {top3}")
    for k in (1, 3, 5):
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  Recall@{k} = {hits}/{len(ranks)} = {hits / len(ranks):.2f}")
    answerable = [results[q["id"]][0]["score"] for q in TEST["answerable"]]
    unanswerable = [results[q["id"]][0]["score"] for q in TEST["not_in_docs"]]
    print(f"  Top-1 similarity: answerable {min(answerable):.4f} to {max(answerable):.4f}, "
          f"not in docs {min(unanswerable):.4f} to {max(unanswerable):.4f}")


def part2(assistant) -> None:
    print("\n2. The relevance threshold, chosen on the development questions\n")
    answerable = {qid: r[0]["score"] for qid, r in top_scores(assistant, DEV["answerable"]).items()}
    unanswerable = {qid: r[0]["score"] for qid, r in top_scores(assistant, DEV["not_in_docs"]).items()}
    print(f"  {len(answerable)} answerable dev questions, top-1 similarity {min(answerable.values()):.4f} to "
          f"{max(answerable.values()):.4f}")
    print(f"  {len(unanswerable)} not-in-docs dev questions, top-1 similarity {min(unanswerable.values()):.4f} to "
          f"{max(unanswerable.values()):.4f}\n")
    print(f"  {'threshold':>9}{'answerable blocked':>21}{'not in docs caught':>21}")
    for t in SWEEP:
        blocked = sum(1 for s in answerable.values() if s < t)
        caught = sum(1 for s in unanswerable.values() if s < t)
        print(f"  {t:>9.2f}{f'{blocked}/{len(answerable)}':>21}{f'{caught}/{len(unanswerable)}':>21}")
    allowed = math.floor(MAX_BLOCKED_SHARE * len(answerable))
    ceiling = sorted(answerable.values())[allowed]  # any threshold up to this blocks at most `allowed`
    rule = math.floor(ceiling * 100) / 100
    configured = SETTINGS["production"]["relevance_threshold"]
    print(f"\n  Rule: block at most {MAX_BLOCKED_SHARE:.0%} of answerable dev questions ({allowed}), rounded down "
          f"to 0.01 -> {rule:.2f}. settings.json uses {configured}.")
    if configured != rule:
        print("  WARNING: settings.json differs from the rule.")
    if configured is not None:
        names = {q["id"]: q["question"] for q in DEV["answerable"] + DEV["not_in_docs"]}
        for qid, score in sorted(answerable.items(), key=lambda kv: kv[1]):
            if score < configured:
                print(f"  blocked (answerable): {qid} {names[qid]!r} {score:.4f}")
        for qid, score in sorted(unanswerable.items(), key=lambda kv: kv[1]):
            if score >= configured:
                print(f"  passed (not in docs): {qid} {names[qid]!r} {score:.4f}")


def report(records: list[dict]) -> None:
    print("\n3. The test questions through both profiles\n")
    print(f"{'':<12}{'answerable (hand labels)':^44}{'not in docs':^20}{'chat':>6}")
    print(f"{'profile':<12}{'correct':>9}{'partly':>8}{'wrong':>7}{'declined':>10}{'unlabeled':>10}"
          f"{'declined':>11}{'answered':>9}{'calls':>6}")
    for profile in PROFILES:
        rows = [r for r in records if r["profile"] == profile]
        a = [label(r) for r in rows if r["kind"] == "answerable"]
        n = [label(r) for r in rows if r["kind"] == "not_in_docs"]
        calls = sum(1 for r in rows if "generate" in r["trace"]["stages"])
        print(f"{profile:<12}{a.count('correct'):>9}{a.count('partly'):>8}{a.count('wrong'):>7}{a.count('declined'):>10}"
              f"{a.count(None):>10}{n.count('declined'):>8}/{len(n):<2}{n.count('answered'):>9}{calls:>6}")
    print("\ncorrect: the key fact is right and nothing in the reply is false. partly: the key fact is right, but the")
    print("reply also says something false. wrong: answered without the right key fact. declined: the customer got")
    print("the decline message (the gate, the model, or a check in code). chat calls: requests that reached the chat model.")

    basic = {r["question_id"]: r for r in records if r["profile"] == "basic"}
    production = [r for r in records if r["profile"] == "production"]
    passed = [r for r in production if r["trace"]["gate"] and r["trace"]["gate"]["passed"]]
    same = sum(1 for r in passed if r["message"] == basic[r["question_id"]]["message"]
               and r["model_output"] == basic[r["question_id"]]["model_output"])
    print(f"\nWhere the gate let a question through ({len(passed)} of {len(production)}), the production reply was "
          f"identical to the basic one for {same} of {len(passed)}.")

    print("\nLatency in this run (it varies between runs and machines):\n")
    print(f"{'profile':<12}{'first request':>15}{'median':>10}{'median retrieve':>17}{'median generate':>17}")
    for profile in PROFILES:
        rows = [r for r in records if r["profile"] == profile]
        latencies = [r["trace"]["latency_ms"] for r in rows]
        stage = {name: [r["trace"]["stages"][name] for r in rows if name in r["trace"]["stages"]]
                 for name in ("retrieve", "generate")}
        print(f"{profile:<12}{latencies[0] / 1000:>13.2f} s{statistics.median_low(latencies) / 1000:>8.2f} s"
              f"{statistics.median_low(stage['retrieve']) / 1000:>15.2f} s{statistics.median_low(stage['generate']) / 1000:>15.2f} s")
    print("first request: each profile starts with both models unloaded, so this includes loading them.")

    print("\nBlocked by the gate (production):")
    for r in production:
        gate = r["trace"]["gate"]
        if gate and not gate["passed"]:
            kind = "answerable" if r["kind"] == "answerable" else "not in docs"
            print(f"  {r['question_id']} ({kind}) top score {gate['top_score']:.4f}: {r['question']!r}")
    print("\nEvery reply that isn't a correct answer or a correct decline:")
    for profile in PROFILES:
        for r in (r for r in records if r["profile"] == profile):
            g = label(r)
            if g == "correct" or (r["kind"] == "not_in_docs" and g == "declined"):
                continue
            top = r["trace"]["gate"]["top_score"] if r["trace"]["gate"] else None
            print(f"  [{profile}] {r['question_id']} {g or 'UNLABELED'} ({r['trace']['outcome']}, top score {top}): "
                  f"{r['question']!r} retrieved {[x['id'] for x in r['trace']['retrieved']]}")
            print(f"      shown: {short(r['message'])}")
            if g is None:
                print(f"      label key: {r['question_id']} | {reply_hash(r['message'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--regrade", metavar="FILE")
    args = parser.parse_args()

    if args.regrade:
        with open(args.regrade) as f:
            report([json.loads(line) for line in f])
        return

    from pipeline import Assistant  # imported here so --regrade needs no index

    unload(chat_model())
    unload(embedding_model())
    assistant = Assistant("basic")
    print(f"Chat model: {chat_model()} | embedding model: {embedding_model()} | index {assistant.index['version']} "
          f"({len(assistant.index['entries'])} articles) | {len(TEST['answerable'])} answerable test questions, "
          f"{len(TEST['not_in_docs'])} not in docs\n")
    part1(assistant)
    part2(assistant)
    unload(embedding_model())
    if args.retrieval_only:
        return

    questions = [(q, "answerable") for q in TEST["answerable"]] + [(q, "not_in_docs") for q in TEST["not_in_docs"]]
    records = []
    for profile in PROFILES:
        unload(chat_model())  # each profile starts from unloaded models
        unload(embedding_model())
        assistant = Assistant(profile)
        for q, kind in questions:
            result = assistant.answer(q["question"])
            records.append({"profile": profile, "question_id": q["id"], "question": q["question"], "kind": kind,
                            "message": result["message"], "sources": result["sources"],
                            "model_output": result["model_output"], "trace": result["trace"]})
            print(".", end="", file=sys.stderr, flush=True)
    unload(chat_model())
    unload(embedding_model())
    print(file=sys.stderr)

    out = f"results-{chat_model().replace(':', '_')}.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(records)
    print(f"\nEvery reply, with its trace, was saved to {out}.")


if __name__ == "__main__":
    main()
