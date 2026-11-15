"""
The three ways to rank chunks for a question, written out in full so every rule is visible.

  keyword   BM25 over the chunk text, with one of two analyzers (how text becomes tokens):
              words        lowercase runs of letters and digits: "SSO-104" -> sso, 104
              identifiers  the same, plus each identifier whole: "SSO-104" -> sso, 104, sso-104
            Only chunks that contain at least one query token are returned.
  vector    cosine similarity between embeddings; every chunk is ranked
  hybrid    reciprocal rank fusion (RRF) of the two rankings, or a weighted sum of their scores,
            each normalized per query

All rankings break ties by chunk order, so Python and JavaScript rank identically.
"""
import math
import re

WORD = re.compile(r"[a-z0-9]+")
IDENTIFIER = re.compile(r"[a-z0-9]+(?:[-_./:]+[a-z0-9]+)+")


def analyze(text: str, analyzer: str) -> list[str]:
    lowered = text.lower()
    tokens = WORD.findall(lowered)
    if analyzer == "identifiers":
        tokens += IDENTIFIER.findall(lowered)
    elif analyzer != "words":
        raise ValueError(f"unknown analyzer {analyzer!r}")
    return tokens


class BM25:
    """BM25 as in Lucene: idf = ln(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, texts: list[str], analyzer: str, k1: float, b: float):
        self.analyzer, self.k1, self.b = analyzer, k1, b
        self.counts = []
        for text in texts:
            counts: dict[str, int] = {}
            for token in analyze(text, analyzer):
                counts[token] = counts.get(token, 0) + 1
            self.counts.append(counts)
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average_length = sum(self.lengths) / len(self.lengths)
        df: dict[str, int] = {}
        for counts in self.counts:
            for token in counts:
                df[token] = df.get(token, 0) + 1
        n = len(texts)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        tokens = analyze(query, self.analyzer)
        result = []
        for counts, length in zip(self.counts, self.lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.average_length)
            for token in tokens:  # each query token counts once per occurrence
                tf = counts.get(token, 0)
                if tf:
                    score += self.idf[token] * tf * (self.k1 + 1) / (tf + norm)
            result.append(score)
        return result

    def postings(self) -> int:
        return sum(len(c) for c in self.counts)


def rank(scores: list[float], matched_only: bool = False) -> list[int]:
    """Chunk indexes, best first; ties by chunk order. matched_only drops chunks with a score of 0."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [i for i in order if scores[i] > 0] if matched_only else order


def rrf(rankings: list[list[int]], k: int, size: int) -> list[float]:
    """Reciprocal rank fusion: each ranking adds 1 / (k + rank) for every chunk in it, rank starting at 1."""
    fused = [0.0] * size
    for ranking in rankings:
        for position, i in enumerate(ranking, 1):
            fused[i] += 1 / (k + position)
    return fused


def min_max(scores: list[float]) -> list[float]:
    low, high = min(scores), max(scores)
    return [0.0] * len(scores) if high == low else [(s - low) / (high - low) for s in scores]


def weighted(vector_scores: list[float], keyword_scores: list[float], alpha: float) -> list[float]:
    """alpha * normalized vector score + (1 - alpha) * normalized keyword score, normalized per query."""
    v, k = min_max(vector_scores), min_max(keyword_scores)
    return [alpha * a + (1 - alpha) * b for a, b in zip(v, k)]
