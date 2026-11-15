"""
The answer stage: does better retrieval give better answers?

For the baseline and the two strategies ranked best on the development questions (the rule in
data/settings.json), retrieve chunks for every test question within the answer budget, send them to the
chat model with data/prompt.json, and save the replies to results-answers.jsonl.

Replies are graded by hand. data/hand_labels.json maps "question id | hash of the reply" to a label, so a
label applies only to the exact reply that was read:
  correct   answers the question, and says nothing false
  partly    right, but incomplete: a missing step, item, or part of the question
  wrong     a false statement, or the key fact is missing
A reply with answered=false is counted as declined. Run with --regrade to print the table again from the
saved replies without calling the models.
"""
import hashlib
import json
import sys

import chunkers
import evaluate
import llm
from tokens import Tokenizer

LABELS = ("correct", "partly", "wrong")


def reply_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]


def strategies_to_answer(settings: dict, questions: list[dict]) -> list[str]:
    saved = json.loads((evaluate.ROOT / "results-retrieval.json").read_text())
    order = evaluate.ranking(saved, settings, questions)
    baseline = settings["answer"]["baseline"]
    return [baseline] + [n for n in order if n != baseline][:settings["answer"]["best"]]


def run(settings: dict, documents: dict[str, str], questions: list[dict], tokenizer) -> list[dict]:
    prompt = json.loads((evaluate.DATA / "prompt.json").read_text())
    test = [q for q in questions if q["split"] == "test"]
    by_name = {s["name"]: s for s in settings["strategies"]}
    query_vectors = evaluate.Embedder(settings["embedding"]["batch_size"])(
        [settings["embedding"]["query_prefix"] + q["question"] for q in test])
    records = []
    for name in strategies_to_answer(settings, questions):
        index = evaluate.build_index(by_name[name], documents, tokenizer, settings)
        chunks, vectors = index["chunks"], index["vectors"]
        tokens = [c.tokens for c in chunks]
        for question, query in zip(test, query_vectors):
            scores = [chunkers.cosine(query, v) for v in vectors]
            ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
            chosen = evaluate.within_budget(ranked, tokens, settings["answer"]["budget"])
            excerpts = "\n\n".join(f"--- excerpt {n} ({chunks[i].document})\n{chunks[i].text}"
                                   for n, i in enumerate(chosen, 1))
            messages = [{"role": "system", "content": prompt["system"].replace("{excerpts}", excerpts)},
                        {"role": "user", "content": question["question"]}]
            reply = llm.chat(messages, schema=prompt["schema"], max_tokens=settings["answer"]["max_output_tokens"])
            parsed = json.loads(reply["content"])
            records.append({
                "strategy": name, "question_id": question["id"], "category": question["category"],
                "question": question["question"], "chunks": [chunks[i].id for i in chosen],
                "context_tokens": sum(tokens[i] for i in chosen),
                "evidence_found": evaluate.found(question, [chunks[i].text for i in chosen]),
                "answered": parsed["answered"], "answer": parsed["answer"], "prompt_version": prompt["version"],
                "prompt_tokens": reply["prompt_tokens"], "output_tokens": reply["output_tokens"],
                "done_reason": reply["done_reason"],
            })
    return records


def label(record: dict, labels: dict) -> str | None:
    if not record["answered"]:
        return "declined"
    return labels.get(f"{record['question_id']} | {reply_hash(record['answer'])}")


def report(records: list[dict], questions: list[dict]) -> None:
    labels = json.loads((evaluate.DATA / "hand_labels.json").read_text())["labels"]
    names = list(dict.fromkeys(r["strategy"] for r in records))
    width = max(len(n) for n in names) + 2
    print(f"Answers to the {len({r['question_id'] for r in records})} test questions (hand labels)")
    print(f"{'strategy':<{width}}{'correct':>8}{'partly':>8}{'wrong':>7}{'declined':>10}{'unlabeled':>11}"
          f"{'evidence found':>16}{'context tokens':>16}{'prompt tokens':>15}")
    for name in names:
        rows = [r for r in records if r["strategy"] == name]
        got = [label(r, labels) for r in rows]
        context = sorted(r["context_tokens"] for r in rows)
        prompt = sorted(r["prompt_tokens"] for r in rows)
        print(f"{name:<{width}}" + "".join(f"{got.count(k):>{w}}" for k, w in
                                          (("correct", 8), ("partly", 8), ("wrong", 7), ("declined", 10)))
              + f"{got.count(None):>11}{sum(r['evidence_found'] for r in rows):>16}"
              f"{f'median {context[len(context) // 2]}':>16}{f'median {prompt[len(prompt) // 2]}':>15}")

    print("\nBy kind of question: correct / questions")
    categories = list(dict.fromkeys(q["category"] for q in questions))
    test = [q for q in questions if q["split"] == "test"]
    sizes = {c: sum(q["category"] == c for q in test) for c in categories}
    print(f"{'strategy':<{width}}" + "".join(f"{f'{c} ({sizes[c]})':>14}" for c in categories))
    for name in names:
        rows = [r for r in records if r["strategy"] == name]
        print(f"{name:<{width}}" + "".join(
            f"{sum(label(r, labels) == 'correct' for r in rows if r['category'] == c):>14}" for c in categories))

    print("\nEvidence found in the context, and the answer")
    for name in names:
        rows = [r for r in records if r["strategy"] == name]
        for found in (True, False):
            subset = [label(r, labels) for r in rows if r["evidence_found"] == found]
            if subset:
                counts = ", ".join(f"{subset.count(k)} {k}" for k in LABELS + ("declined",) if subset.count(k))
                print(f"  {name:<{width}}evidence {'found' if found else 'missing'} ({len(subset)}): {counts}")

    unlabeled = [r for r in records if label(r, labels) is None]
    if unlabeled:
        print(f"\nUnlabeled replies ({len(unlabeled)}), to read and add to data/hand_labels.json:")
        for r in unlabeled:
            print(f'  "{r["question_id"]} | {reply_hash(r["answer"])}"  [{r["strategy"]}] {r["question"]}')
            print(f"      {r['answer']}")


def main() -> None:
    settings, documents, questions = evaluate.load()
    path = evaluate.ROOT / "results-answers.jsonl"
    if "--regrade" not in sys.argv:
        tokenizer = Tokenizer()
        try:
            records = run(settings, documents, questions, tokenizer)
        finally:
            llm.unload(llm.chat_model())
            llm.unload(llm.embedding_model())
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    report(records, questions)


if __name__ == "__main__":
    main()
