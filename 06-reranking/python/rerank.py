"""
The reranker: a chat model reads the question and one candidate chunk together and answers Yes or No.
The score is the log-odds of its first token, log P(yes) - log P(no), from the top alternatives Ollama
returns. It is a model-derived relevance score, not a calibrated probability: the model can be
confidently wrong.
"""
import json
import math
from pathlib import Path

import llm

DATA = Path(__file__).resolve().parent.parent / "data"


def load_prompt() -> dict:
    return json.loads((DATA / "prompt.json").read_text())


def messages(prompt: dict, question: str, passage: str) -> list[dict]:
    return [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"].replace("{question}", question).replace("{passage}", passage)}]


def logsumexp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def score(top: dict[str, float], settings: dict) -> float:
    """log P(yes tokens) - log P(no tokens); a label missing from the alternatives gets the lowest listed value."""
    floor = min(top.values())
    yes = [top[t] for t in settings["yes"] if t in top] or [floor]
    no = [top[t] for t in settings["no"] if t in top] or [floor]
    return logsumexp(yes) - logsumexp(no)


def judge(prompt: dict, question: str, passage: str, settings: dict) -> dict:
    result = llm.first_token_logprobs(messages(prompt, question, passage), options=settings["options"],
                                      top_logprobs=settings["top_logprobs"])
    return {**result, "score": score(result["top"], settings)}
