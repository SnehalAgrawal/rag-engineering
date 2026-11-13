"""
The index lifecycle: what happens between "an article changed" and "the assistant answers with it"?

  python update.py

Works on a copy of the help center and its own index in a temporary folder; ../data is not touched.
  1. Build the index, then build it again with nothing changed.
  2. Billing changes five articles (the same changes as in 02-rag-vs-fine-tuning).
  3. Ask the five affected test questions before re-indexing: the assistant still serves the old version,
     and every trace says how many articles changed since it was built.
  4. Re-index (only the changed articles are embedded) and ask again. The running assistant loads the new
     version on its next request.
  5. An article is removed: re-indexing drops it without embedding anything.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import index_store
from index import describe, update
from llm import chat_model, embedding_model, unload
from pipeline import Assistant

DATA = Path(__file__).resolve().parent.parent / "data"
# (article id, old text, new text)
CHANGES = [
    ("plans", "The Pro plan costs $25 per month", "The Pro plan costs $30 per month"),
    ("annual-billing", "$250 per year for Pro", "$300 per year for Pro"),
    ("password", "The reset link expires after 1 hour.", "The reset link expires after 2 hours."),
    ("invite-members", "expire after 7 days", "expire after 14 days"),
    ("invite-members", "up to 5 members", "up to 10 members"),
    ("change-plan", "no more than 5 remain", "no more than 10 remain"),
]
QUESTIONS = ["a01", "a02", "a11", "a21", "a24"]  # the test questions about the changed facts
REMOVED = "sso"


def edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data["articles"])
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def apply_changes(articles: list[dict]) -> None:
    by_id = {a["id"]: a for a in articles}
    for article_id, old, new in CHANGES:
        if old not in by_id[article_id]["text"]:
            raise SystemExit(f"{article_id}: {old!r} not found")
        by_id[article_id]["text"] = by_id[article_id]["text"].replace(old, new)


def main() -> None:
    questions = {q["id"]: q["question"] for q in json.loads((DATA / "questions.json").read_text())["answerable"]}
    records = []
    with tempfile.TemporaryDirectory() as folder:
        help_center, index_path = Path(folder) / "help_center.json", Path(folder) / "index.json"
        shutil.copy(DATA / "help_center.json", help_center)

        def reindex(step: str) -> None:
            index, stats = update(help_center, index_path)
            print(f"{step}: {describe(index, stats)}")

        def ask(step: str) -> None:
            for qid in QUESTIONS:
                result = assistant.answer(questions[qid])
                trace = result["trace"]
                print(f"  {qid} [index {trace['index_version']}, {trace['index_stale_articles']} stale] "
                      f"{questions[qid]!r}\n      {result['message']}")
                records.append({"step": step, "question_id": qid, "question": questions[qid],
                                "message": result["message"], "trace": trace})

        unload(embedding_model())
        unload(chat_model())
        print("1. Build the index")
        reindex("   from scratch")
        reindex("   again, nothing changed")
        assistant = Assistant("production", index_path=index_path, help_center_path=help_center)

        edit(help_center, apply_changes)
        stale = index_store.staleness(index_store.read(index_path), index_store.read_articles(help_center))
        print(f"\n2. Billing changes {len({c[0] for c in CHANGES})} articles ({len(CHANGES)} edits). "
              f"Articles changed since the index was built: {stale}")
        print("\n3. Before re-indexing, the running assistant still serves the old version:")
        ask("before re-indexing")

        print()
        reindex("4. Re-index")
        print("   The same assistant, on its next requests:")
        ask("after re-indexing")

        edit(help_center, lambda articles: articles.remove(next(a for a in articles if a["id"] == REMOVED)))
        print()
        reindex(f"5. The '{REMOVED}' article is removed. Re-index")
        print(f"   entries in the index: {len(index_store.read(index_path)['entries'])}")
    unload(chat_model())
    unload(embedding_model())
    print(file=sys.stderr)

    with open("results-update.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nEvery reply, with its trace, was saved to results-update.jsonl.")


if __name__ == "__main__":
    main()
