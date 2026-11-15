"""
The chunking strategies. Every chunk is one slice of one document, document[start:end], exactly as written.
structure-headings also copies the heading lines above the slice in front of it. A chunk's text, which is
what gets embedded and what the answer model reads, is that prefix plus the slice.

  whole      one chunk per document
  fixed      windows of `size` tokens, each starting `size - overlap` tokens after the previous one
  structure  one chunk per Markdown section; a section over `max_tokens` is split between paragraphs,
             then between lines. A heading with nothing under it joins the section that follows.
  semantic   split between sentences where the meaning shifts most: each sentence is embedded together
             with its neighbours (`buffer` on each side), and a chunk ends after a sentence whose distance
             to the next is above the `percentile` of all distances in the document. Table rows, list
             items, headings, and code blocks count as sentences.
"""
import math
import re
from dataclasses import dataclass

HEADING = re.compile(r"#{1,6} ")
LIST_ITEM = re.compile(r"(- |[0-9]+\. )")
SENTENCE_END = re.compile(r"[.!?]( +)(?=[A-Z])")


@dataclass
class Chunk:
    id: str
    document: str
    start: int
    end: int
    prefix: str
    text: str  # prefix + document[start:end]: what is embedded and what the model reads
    tokens: int


@dataclass
class Line:
    start: int
    end: int
    text: str
    kind: str  # heading, fence, code, blank, text


def parse_lines(text: str) -> list[Line]:
    lines, position, in_code = [], 0, False
    for raw in text.split("\n"):
        start, end = position, position + len(raw)
        position = end + 1
        if raw.startswith("```"):
            kind, in_code = "fence", not in_code
        elif in_code:
            kind = "code"
        elif raw.strip() == "":
            kind = "blank"
        elif HEADING.match(raw):
            kind = "heading"
        else:
            kind = "text"
        lines.append(Line(start, end, raw, kind))
    return lines


def blocks(lines: list[Line]) -> list[list[Line]]:
    """Paragraphs, tables, lists, and code blocks: runs of lines between blank lines. A heading is its own block."""
    result, current = [], []
    for line in lines:
        if line.kind == "blank" or line.kind == "heading":
            if current:
                result.append(current)
            current = [line] if line.kind == "heading" else []
            if line.kind == "heading":
                result.append(current)
                current = []
        else:
            current.append(line)
    if current:
        result.append(current)
    return result


def heading_stack(lines: list[Line], position: int) -> list[Line]:
    """The headings in force at a position: the last heading of each level above it, outermost first."""
    stack: list[Line] = []
    for line in lines:
        if line.start > position:
            break
        if line.kind == "heading":
            level = len(line.text) - len(line.text.lstrip("#"))
            stack = [h for h in stack if len(h.text) - len(h.text.lstrip("#")) < level] + [line]
    return stack


def pack(spans: list[tuple[int, int]], document: str, max_tokens: int, count, split) -> list[tuple[int, int]]:
    """Join consecutive spans while the joined slice stays within max_tokens; split a span that is too big alone."""
    result, current = [], None
    for span in spans:
        if count(document[span[0]:span[1]]) > max_tokens:
            if current:
                result.append(current)
                current = None
            result.extend(split(span))
            continue
        if current and count(document[current[0]:span[1]]) <= max_tokens:
            current = (current[0], span[1])
        else:
            if current:
                result.append(current)
            current = span
    if current:
        result.append(current)
    return result


def fixed(text: str, size: int, overlap: int, tokenizer) -> list[tuple[int, int]]:
    ids = tokenizer.encode(text)
    offsets = [0]
    for token in ids:
        offsets.append(offsets[-1] + len(tokenizer.decode([token])))
    spans, start = [], 0
    while True:
        end = min(start + size, len(ids))
        spans.append((offsets[start], offsets[end]))
        if end == len(ids):
            return spans
        start += size - overlap


def structure(text: str, max_tokens: int, count, tokenizer) -> list[tuple[int, int]]:
    lines = parse_lines(text)
    headings = [i for i, line in enumerate(lines) if line.kind == "heading"]
    starts = ([0] if headings and headings[0] != 0 else []) + headings
    sections, pending = [], None
    for k, first in enumerate(starts):
        last = starts[k + 1] if k + 1 < len(starts) else len(lines)
        content = [line for line in lines[first:last] if line.kind != "blank"]
        begin = pending if pending is not None else content[0].start
        if all(line.kind == "heading" for line in content):  # a heading with nothing under it
            pending = begin
            continue
        sections.append((begin, content[-1].end, blocks(lines[first:last])))
        pending = None

    def split_block(span):
        block_lines = [line for line in lines if span[0] <= line.start and line.end <= span[1]]
        return pack([(line.start, line.end) for line in block_lines], text, max_tokens, count, split_line)

    def split_line(span):  # a single line over max_tokens: cut it into token windows
        return [(span[0] + a, span[0] + b) for a, b in fixed(text[span[0]:span[1]], max_tokens, 0, tokenizer)]

    result = []
    for begin, end, section_blocks in sections:
        if count(text[begin:end]) <= max_tokens:
            result.append((begin, end))
            continue
        spans = [(b[0].start, b[-1].end) for b in section_blocks]
        spans[0] = (begin, spans[0][1])  # keep a merged empty heading with the first block
        result.extend(pack(spans, text, max_tokens, count, split_block))
    return result


def units(text: str) -> list[tuple[int, int]]:
    """Sentences, plus table rows, list items, headings, and whole code blocks, as (start, end) offsets."""
    result = []
    for block in blocks(parse_lines(text)):
        if block[0].kind in ("heading", "fence"):
            result.append((block[0].start, block[-1].end))
            continue
        for line in block:
            if line.text.startswith("|") or LIST_ITEM.match(line.text):
                result.append((line.start, line.end))
                continue
            begin = 0
            for match in SENTENCE_END.finditer(line.text):
                result.append((line.start + begin, line.start + match.start() + 1))
                begin = match.end()
            result.append((line.start + begin, line.end))
    return result


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (the usual default, as in NumPy)."""
    ordered = sorted(values)
    rank = p / 100 * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def semantic(text: str, buffer: int, p: float, embed_units) -> list[tuple[int, int]]:
    spans = units(text)
    windows = [" ".join(text[a:b] for a, b in spans[max(0, i - buffer):i + buffer + 1]) for i in range(len(spans))]
    vectors = embed_units(windows)
    distances = [1 - cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = percentile(distances, p)
    result, first = [], 0
    for i, distance in enumerate(distances):
        if distance > threshold:
            result.append((spans[first][0], spans[i][1]))
            first = i + 1
    result.append((spans[first][0], spans[-1][1]))
    return result


def chunk(strategy: dict, documents: dict[str, str], tokenizer, embed_units=None) -> list[Chunk]:
    """Chunks for every document, in document order. embed_units is needed only for the semantic strategy."""
    count = tokenizer.count
    chunks = []
    for name, text in documents.items():
        kind = strategy["kind"]
        if kind == "whole":
            spans = [(0, len(text))]
        elif kind == "fixed":
            spans = fixed(text, strategy["size"], strategy["overlap"], tokenizer)
        elif kind == "structure":
            spans = structure(text, strategy["max_tokens"], count, tokenizer)
        elif kind == "semantic":
            spans = semantic(text, strategy["buffer"], strategy["percentile"], embed_units)
        else:
            raise ValueError(f"unknown strategy kind {kind!r}")
        lines = parse_lines(text)
        for n, (start, end) in enumerate(spans, 1):
            prefix = ""
            if strategy.get("headings"):
                above = [h.text for h in heading_stack(lines, start) if not start <= h.start < end]
                prefix = "\n".join(above) + "\n\n" if above else ""
            body = prefix + text[start:end]
            chunks.append(Chunk(f"{name.removesuffix('.md')}#{n}", name, start, end, prefix, body, count(body)))
    return chunks
