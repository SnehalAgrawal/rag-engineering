"""
Ask the assistant one question, and see what the pipeline did: what was retrieved, what the gate
decided, what the model returned, what the customer sees, and the trace.

  python ask.py "Can I pay with PayPal?"
  python ask.py "Do you have a phone number?" --profile basic
"""
import argparse
import json

from pipeline import PROFILES, Assistant

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--profile", choices=list(PROFILES), default="production")
args = parser.parse_args()

result = Assistant(args.profile).answer(args.question)
trace = result["trace"]
if trace["retrieved"]:
    print("Retrieved (cosine similarity):")
    for r in trace["retrieved"]:
        print(f"  {r['score']:.4f}  {r['id']}")
if trace["gate"]:
    gate = trace["gate"]
    decision = "no gate" if gate["threshold"] is None else ("passed" if gate["passed"] else "blocked")
    print(f"Gate: top score {gate['top_score']:.4f}, threshold {gate['threshold']}: {decision}")
print(f"\nModel output: {result['model_output']}")
print(f"\nShown to the customer ({result['status']}, {trace['outcome']}): {result['message']}")
if result["sources"]:
    print(f"Sources: {', '.join(result['sources'])}")
print(f"\nTrace: {json.dumps(trace)}")
