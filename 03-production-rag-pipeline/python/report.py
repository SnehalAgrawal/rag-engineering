"""
What the traces alone tell you: the operator's view, without the test questions or the hand labels.

  python report.py                      reads the results files that exist in this folder
  python report.py FILE [FILE ...]

Each line of a results file holds one request's trace. The report answers: what happened to the
requests, which call failed and how, where the time went, which index and prompt versions answered,
and how close the answered questions came to the relevance threshold.
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

DEFAULT_FILES = ["results-llama3.2_3b.jsonl", "results-faults.jsonl", "results-update.jsonl"]
NEAR_THRESHOLD = 0.02  # an arbitrary review band above the threshold, not a tuned value


def main() -> None:
    files = sys.argv[1:] or [f for f in DEFAULT_FILES if Path(f).exists()]
    traces = [json.loads(line)["trace"] for name in files for line in open(name)]
    profiles = sorted({t["profile"] for t in traces})
    print(f"{len(traces)} traces from {', '.join(files)}\n")

    print("1. Outcomes\n")
    print(f"  {'outcome':<22}" + "".join(f"{p:>12}" for p in profiles))
    counts = {p: Counter(t["outcome"] for t in traces if t["profile"] == p) for p in profiles}
    for outcome in sorted({t["outcome"] for t in traces}):
        print(f"  {outcome:<22}" + "".join(f"{counts[p][outcome]:>12}" for p in profiles))
    calls = {p: sum(1 for t in traces if t["profile"] == p and "generate" in t["stages"]) for p in profiles}
    print(f"  {'(chat calls made)':<22}" + "".join(f"{calls[p]:>12}" for p in profiles))

    print("\n2. Failed calls\n")
    failures = Counter((t["profile"], t["error"]["stage"], t["error"]["kind"]) for t in traces if t["error"])
    for (profile, stage, kind), n in sorted(failures.items()):
        slowest = max(t["latency_ms"] for t in traces if t["error"] and (t["profile"], t["error"]["stage"],
                                                                        t["error"]["kind"]) == (profile, stage, kind))
        print(f"  {profile:<12}{stage + ': ' + kind:<28}requests: {n}, longest wait {slowest / 1000:.1f} s")

    print("\n3. Where the time went (answered requests)\n")
    print(f"  {'profile':<12}{'stage':<10}{'median':>9}{'max':>9}")
    for profile in profiles:
        answered = [t for t in traces if t["profile"] == profile and t["outcome"] == "answered"]
        for stage in ("retrieve", "generate"):
            ms = [t["stages"][stage] for t in answered]
            print(f"  {profile:<12}{stage:<10}{statistics.median_low(ms) / 1000:>7.2f} s{max(ms) / 1000:>7.2f} s")
        ms = [t["latency_ms"] for t in answered]
        print(f"  {profile:<12}{'total':<10}{statistics.median_low(ms) / 1000:>7.2f} s{max(ms) / 1000:>7.2f} s")

    print("\n4. What answered: versions\n")
    for key in ("prompt_version", "chat_model", "embedding_model", "index_version"):
        seen = Counter(t[key] for t in traces)
        print(f"  {key:<16}" + ", ".join(f"{value} ({n})" for value, n in seen.items()))
    stale = [t for t in traces if t["index_stale_articles"]]
    print(f"  {len(stale)} requests were served from an index built before articles changed "
          f"(up to {max((t['index_stale_articles'] for t in stale), default=0)} changed).")

    print("\n5. The relevance gate\n")
    gated = [t for t in traces if t["gate"] and t["relevance_threshold"] is not None]
    blocked = [t for t in gated if not t["gate"]["passed"]]
    scores = ", ".join(f"{t['gate']['top_score']:.4f}" for t in blocked)
    print(f"  {len(blocked)} of {len(gated)} requests blocked; top scores of the blocked: {scores}")
    near = [t for t in gated if t["outcome"] == "answered"
            and t["gate"]["top_score"] < t["relevance_threshold"] + NEAR_THRESHOLD]
    print(f"  answered with a top score within {NEAR_THRESHOLD} of the threshold: {len(near)}")
    for t in near:
        print(f"    request {t['request_id'][:8]}: top score {t['gate']['top_score']:.4f}, "
              f"retrieved {[r['id'] for r in t['retrieved']]}")
    print("\n  A trace shows what the model saw and what the pipeline decided. It doesn't show whether an answer")
    print("  was right: that takes labeled questions and reading replies (the evaluate script).")


if __name__ == "__main__":
    main()
