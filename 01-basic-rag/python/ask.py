"""
Ask the assistant one question, and see every step: what was retrieved, what
the model returned, and what the customer would see.

  python ask.py "Can I pay with PayPal?"
  python ask.py "Can I pay with PayPal?" --k 5
  python ask.py "Can I pay with PayPal?" --mode all
"""
import argparse
import json

from rag import answer

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--mode", choices=["rag", "all"], default="rag")
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--num-ctx", type=int)
args = parser.parse_args()

result = answer(args.question, mode=args.mode, k=args.k, num_ctx=args.num_ctx)
if result["retrieved"]:
    print("Retrieved (cosine similarity):")
    for r in result["retrieved"]:
        print(f"  {r['score']:.4f}  {r['id']}")
print(f"\nModel output: {result['model_output']}")
print(f"\nShown to the customer ({result['status']}, {result['log']['outcome']}): {result['message']}")
if result["sources"]:
    print(f"Sources: {', '.join(result['sources'])}")
print(f"\nLog: {json.dumps(result['log'])}")
