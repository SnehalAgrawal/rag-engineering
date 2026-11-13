"""
Ask one question, with or without the fine-tuned adapter, and with or without retrieval.

  python ask.py "How much is the Pro plan?"                      base model, no articles
  python ask.py "How much is the Pro plan?" --fine-tuned         the fine-tuned adapter, no articles
  python ask.py "How much is the Pro plan?" --rag                base model with the 3 retrieved articles
  python ask.py "How much is the Pro plan?" --fine-tuned --rag   both
"""
import argparse

from common import ARTICLES, Index, generate, load_model, messages_for, unload_embedding_model

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--fine-tuned", action="store_true")
parser.add_argument("--rag", action="store_true")
parser.add_argument("--k", type=int, default=3)
args = parser.parse_args()

docs = None
if args.rag:
    docs = Index(ARTICLES).search(args.question, args.k)
    unload_embedding_model()
    print("Retrieved:", ", ".join(a["id"] for a in docs))
model, tokenizer = load_model(args.fine_tuned)
print(generate(model, tokenizer, messages_for(args.question, docs)))
