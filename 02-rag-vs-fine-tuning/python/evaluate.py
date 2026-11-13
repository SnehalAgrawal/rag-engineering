"""
Knowledge in the weights (fine-tuning) versus knowledge in the prompt (RAG), on the same model.

  python evaluate.py                  all four conditions, then the changed-facts test
  python evaluate.py --regrade FILE   re-grade saved replies without calling any model

Conditions (Llama 3.2 1B Instruct, 4-bit, temperature 0):
  base               the model as downloaded, short system prompt, no articles
  fine-tuned         with the LoRA adapter from train.py, same short prompt, no articles
  base + rag         the model as downloaded, with the 3 retrieved articles and the RAG prompt
  fine-tuned + rag   with the adapter, the 3 retrieved articles, and the RAG prompt

Question sets:
  test           the 24 answerable questions from 01-basic-rag, worded differently from the training data
  training       the 24 training questions that cover the same facts (seen during fine-tuning)
  not in docs    the 6 questions the help center doesn't answer
  changed        5 test questions, asked after 5 articles change (see UPDATES)
"""
import argparse
import copy
import hashlib
import json
import re
import sys
import time

from common import (
    ARTICLES,
    DATA,
    TEST,
    TRAINING,
    Index,
    generate,
    load_model,
    messages_for,
    titles,
    unload_embedding_model,
)

# The help center after an update: (article id, old text, new text). Fine-tuning saw only the old text.
UPDATES = [
    ("plans", "The Pro plan costs $25 per month", "The Pro plan costs $30 per month"),
    ("annual-billing", "$250 per year for Pro", "$300 per year for Pro"),
    ("password", "The reset link expires after 1 hour.", "The reset link expires after 2 hours."),
    ("invite-members", "expire after 7 days", "expire after 14 days"),
    ("invite-members", "up to 5 members", "up to 10 members"),
    ("change-plan", "no more than 5 remain", "no more than 10 remain"),
]
# Test questions about the changed facts: patterns for the new and the old value.
CHANGED = {
    "a01": {"new": [r"\b2\s*hours?\b", r"\btwo\s*hours?\b"], "old": [r"\b1\s*hour\b", r"\bone\s*hour\b", r"\b60\s*minutes\b"]},
    "a02": {"new": [r"\$\s?30\b"], "old": [r"\$\s?25\b"]},
    "a11": {"new": [r"\b10\b", r"\bten\b"], "old": [r"\b5\b", r"\bfive\b"]},
    "a21": {"new": [r"\$\s?300\b"], "old": [r"\$\s?250\b"]},
    "a24": {"new": [r"\b14\s*days\b", r"\bfourteen\b", r"\b(two|2)\s*weeks\b"], "old": [r"\b7\s*days\b", r"\bseven\s*days\b", r"\b(one|1)\s*week\b"]},
}
CONDITIONS = ["base", "fine-tuned", "base + rag", "fine-tuned + rag"]
K = 3

ANSWERABLE = {q["id"]: q for q in TEST["answerable"]}
TWINS = {e["twin"]: e for e in TRAINING["train"] if "twin" in e}
SOURCE_LINE = re.compile(r"^\s*Source:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
DECLINED = re.compile(r"\b(don't|do not) have (that|this) information\b", re.IGNORECASE)


def updated_articles() -> list[dict]:
    articles = copy.deepcopy(ARTICLES)
    by_id = {a["id"]: a for a in articles}
    for article_id, old, new in UPDATES:
        assert by_id[article_id]["text"].count(old) == 1, (article_id, old)
        by_id[article_id]["text"] = by_id[article_id]["text"].replace(old, new)
    return articles


def body(reply: str) -> str:
    """The reply without its 'Source:' line."""
    return SOURCE_LINE.sub("", reply).strip()


def matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def grade(r: dict) -> str:
    text = body(r["reply"])
    if DECLINED.search(text):
        return "declined"
    if r["set"] == "not in docs":
        return "answered"
    if r["set"] == "changed":
        new, old = matches(CHANGED[r["question_id"]]["new"], text), matches(CHANGED[r["question_id"]]["old"], text)
        return "new fact" if new and not old else "old fact" if old and not new else "both" if new else "neither"
    return "key fact" if matches(ANSWERABLE[r["question_id"]]["patterns"], text) else "no key fact"


def source_ok(r: dict) -> bool | None:
    """Does the reply end with 'Source: <title>' naming an article that answers the question?"""
    found = SOURCE_LINE.findall(r["reply"])
    if not found:
        return None
    relevant = {titles(ARTICLES)[a] for a in ANSWERABLE[r["question_id"]]["relevant"]}
    return found[-1] in relevant


def short(text: str, width: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


LABELS = json.loads((DATA / "hand_labels.json").read_text())["labels"]


def hand_label(r: dict) -> str | None:
    """The hand label for this exact reply, or None if the reply differs from the one that was labeled."""
    entry = LABELS.get(f"{r['condition']} | {r['set']} | {r['question_id']}")
    digest = hashlib.sha256(r["reply"].encode("utf-8")).hexdigest()[:16]
    return entry["label"] if entry and entry["reply_sha256"] == digest else None


def report(records: list[dict]) -> None:
    def rows(condition, set_name):
        return [r for r in records if r["condition"] == condition and r["set"] == set_name]

    def count(items, label):
        return sum(1 for r in items if hand_label(r) == label)

    print("\n1. Hand labels: answerable questions and questions not in the help center\n")
    print(f"{'':<18}{'test wording (24)':^36}{'training wording (24)':^24}{'not in docs (6)':^16}")
    print(f"{'condition':<18}{'correct':>9}{'partly':>9}{'wrong':>8}{'declined':>10}{'correct':>14}{'partly':>9}{'declined':>15}")
    for c in CONDITIONS:
        test, train, none = rows(c, "test"), rows(c, "training"), rows(c, "not in docs")
        print(f"{c:<18}{count(test, 'correct'):>9}{count(test, 'partly'):>9}{count(test, 'wrong'):>8}{count(test, 'declined'):>10}"
              f"{count(train, 'correct'):>14}{count(train, 'partly'):>9}{count(none, 'declined'):>15}")
    print("\ncorrect: the key fact is right and nothing in the reply is false.")
    print("partly: the key fact is right, but the reply also says something false, contradictory, or garbled.")
    print("wrong: an answer without the right key fact. declined: no answer.")

    print("\n2. Hand labels: 5 test questions after their facts changed in 5 articles\n")
    print(f"{'condition':<18}{'new fact':>10}{'new, with an error':>20}{'old fact':>10}{'other':>7}")
    for c in ["fine-tuned", "base + rag", "fine-tuned + rag"]:
        changed = rows(c, "changed")
        print(f"{c:<18}{count(changed, 'new'):>10}{count(changed, 'new (with an error)'):>20}{count(changed, 'old'):>10}{count(changed, 'other'):>7}")

    print("\n3. Checks in code: the 'Source:' line, and the key-fact pattern as a cross-check\n")
    print(f"{'condition':<18}{'source line present':>21}{'names a right article':>23}{'key fact (pattern)':>20}")
    for c in CONDITIONS:
        test = rows(c, "test")
        shown = [r for r in test if grade(r) != "declined"]
        sources = [source_ok(r) for r in shown]
        print(f"{c:<18}{sum(x is not None for x in sources):>17}/{len(shown):<3}{sum(x is True for x in sources):>19}/{len(shown):<3}"
              f"{sum(grade(r) == 'key fact' for r in test):>17}/{len(test):<2}")
    print("\nSource line: of the test-wording replies that weren't declines, how many ended with 'Source: <title>',")
    print("and how many of those named an article that answers the question. The pattern check misgrades replies")
    print("both ways (a garbled reply can contain the right number), which is why section 1 uses hand labels.")

    unlabeled = [r for r in records if hand_label(r) is None]
    if unlabeled:
        print(f"\n{len(unlabeled)} replies differ from the labeled ones; read them and update data/hand_labels.json:")
        for r in unlabeled:
            print(f"  [{r['condition']} | {r['set']}] {r['question_id']}: {short(r['reply'])}")
    else:
        print("\nEvery reply matches a hand-labeled reply exactly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regrade", metavar="FILE")
    args = parser.parse_args()
    if args.regrade:
        with open(args.regrade) as f:
            report([json.loads(line) for line in f])
        return

    questions = (
        [("test", q["id"], q["question"]) for q in TEST["answerable"]]
        + [("training", qid, TWINS[qid]["question"]) for qid in ANSWERABLE]
        + [("not in docs", q["id"], q["question"]) for q in TEST["not_in_docs"]]
    )
    changed = [("changed", qid, ANSWERABLE[qid]["question"]) for qid in CHANGED]

    start = time.perf_counter()
    index = Index(ARTICLES)
    updated = Index(updated_articles(), reuse=index)
    retrieved = {q: index.search(q, K) for _, _, q in questions}
    retrieved_updated = {q: updated.search(q, K) for _, _, q in changed}
    unload_embedding_model()
    hits = sum(any(a["id"] in ANSWERABLE[qid]["relevant"] for a in retrieved[q]) for s, qid, q in questions if s == "test")
    print(f"Indexed {len(ARTICLES)} articles; after the update, re-embedded {updated.embedded} changed articles.")
    print(f"Retrieval: an answering article was in the top {K} for {hits}/{len(ANSWERABLE)} test questions.")

    records = []
    for fine_tuned in (False, True):
        model, tokenizer = load_model(fine_tuned)
        name = "fine-tuned" if fine_tuned else "base"
        runs = [(name, questions, None), (f"{name} + rag", questions, retrieved)]
        if fine_tuned:
            runs.append((name, changed, None))
        runs.append((f"{name} + rag", changed, retrieved_updated))
        for condition, items, context in runs:
            for set_name, qid, question in items:
                docs = context[question] if context else None
                reply = generate(model, tokenizer, messages_for(question, docs))
                records.append({"condition": condition, "set": set_name, "question_id": qid, "question": question,
                                "retrieved": [a["id"] for a in docs] if docs else None, "reply": reply})
                print(".", end="", file=sys.stderr, flush=True)
        del model
    print(file=sys.stderr)

    with open("results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(records)
    print(f"\n{len(records)} replies in {time.perf_counter() - start:.0f} s, saved to results.jsonl.")


if __name__ == "__main__":
    main()
