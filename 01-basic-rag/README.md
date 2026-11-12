# Basic RAG: Retrieval Before Generation

A minimal retrieval-augmented support assistant, in Python and JavaScript. It accompanies the article **[RAG Explained: How AI Systems Work With Your Data](https://snehalagrawal.com/articles/rag-explained-how-ai-systems-work-with-your-data/)** (Article #12).

It continues the support assistant from Article #11 ([first-llm-app](https://github.com/SnehalAgrawal/ai-engineering-examples/tree/main/first-llm-app)). That version sent the whole help center with every question. Here the help center has grown from 8 to 50 articles, and the demo compares four ways of giving the model the knowledge it needs, on the same 30 questions:

| Mode | What the model receives |
|---|---|
| `all, default context` | Every article, with the context length Ollama chooses (4,096 tokens on the test machine) |
| `all, 8K context` | Every article, with an 8,192-token context: enough to hold all of it |
| `rag, top 1` | Only the article whose embedding is most similar to the question |
| `rag, top 3` | Only the 3 most similar articles |

Everything else is the same: the prompt, the JSON schema, the output checks, and the decline message.

## What it demonstrates

1. **Knowledge that doesn't fit gets cut silently.** The whole help center is 4,260 prompt tokens. At the default context, Ollama kept 2,050 of them, returned success, and only 3 of 24 answerable questions got an answer with the key fact.
2. **Retrieval sends a fraction of the prompt tokens.** RAG top 3 used a median of 431 prompt tokens per question (plus one embedding call), against 4,262 for the whole help center, and the expected key fact appeared in 22 of 24 answerable replies, against 18 of 24. A key fact appearing is not the same as a correct answer: see the hand-reading notes below.
3. **Retrieval quality and answer quality are measured separately.** Recall@3 was 24/24, but one answer was still wrong: the model had the right article and misread it.
4. **This implementation always hands the model something.** It has no "nothing relevant found" decision: for questions the help center doesn't answer, retrieval still returns the closest articles, and the model answered 3 of 6 of them (top 3) from near-miss content.
5. **Sending everything isn't always slower.** With Ollama's prompt cache, the unchanged help-center prefix was reused across questions, so after the first request the whole-help-center mode had the lowest median latency.

## Flow

```text
Index time (once):   50 articles ──► embedding model ──► vectors ──► data/index.json
Query time:          question ──► embedding model ──► top-k articles by cosine similarity
                     ──► prompt (instructions + those k articles) ──► chat model (JSON schema)
                     ──► checks (schema, cited ids, values) ──► reply shown to the customer, and a log record
```

| File | Purpose |
|---|---|
| `data/help_center.json` | The 50 articles (fictional product; the 8 from Article #11 are unchanged) |
| `data/questions.json` | 24 answerable questions labeled with the article(s) that answer them, and 6 the help center doesn't answer |
| `data/prompt.json` | The prompt (`support-v1`, the same instructions as Article #11), decline and unavailable messages |
| `python/llm.py`, `javascript/llm.js` | The two model calls (`embed`, `chat`); every failure becomes `LLMError` |
| `python/index.py`, `javascript/index.js` | Index time: embeds every article and writes `data/index.json` |
| `python/rag.py`, `javascript/rag.js` | The application: validate, retrieve, build the prompt, call, check, reply, log |
| `python/ask.py`, `javascript/ask.js` | Ask one question and see each step |
| `python/evaluate.py`, `javascript/evaluate.js` | Retrieval metrics, then all four modes on all 30 questions |

The index records the embedding model, the prefixes, and a SHA-256 of every article. `rag.py` refuses to use an index built with different settings or from articles that have changed since, and tells you to re-run the indexer. A stale index is a quiet RAG failure: recently changed content can be missed or ranked incorrectly (and if an index also stores document text, it can serve stale content).

## Prerequisites

- Python 3.10+ **or** Node.js 18+
- [Ollama](https://ollama.com/download), with the two models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Configuration

Copy `.env.example` to `.env` in `python/` or `javascript/` if you want to change anything. The defaults work without a `.env` file.

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_CHAT_MODEL` | `llama3.2:3b` | |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Rebuild the index after changing it |
| `QUERY_PREFIX` / `DOCUMENT_PREFIX` | `search_query: ` / `search_document: ` | The task prefixes nomic-embed-text was trained with. Rebuild the index after changing them |

## How to run

### Python

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python index.py
python ask.py "Can I bring in tasks from a spreadsheet?"
python evaluate.py
```

### JavaScript

```bash
cd javascript
npm install
node index.js
node ask.js "Can I bring in tasks from a spreadsheet?"
node evaluate.js
```

`evaluate.py --retrieval-only` runs only the retrieval metrics (no chat model). `evaluate.py --regrade results-llama3.2_3b.jsonl` re-grades saved replies without calling any model, for when you change a check. The JavaScript version takes the same flags.

`evaluate` makes 120 chat requests and takes about three minutes on the test machine. It unloads both models when it finishes.

## Expected output

The numbers below come from this setup:

| Setting | Value |
|---|---|
| Ollama | 0.40.16, `/api/chat` and `/api/embed`, on an Apple M4 Pro (24 GB), on AC power |
| Chat model | `llama3.2:3b`, ID `a80c4f17acd5` (Q4_K_M; weights blob `sha256-dde5aa3fc5ff…`) |
| Embedding model | `nomic-embed-text:latest`, ID `0a109f422b47` (weights blob `sha256-970aa74c0a90…`), 768 dimensions, prefixes `search_query: ` / `search_document: ` |
| Generation | temperature 0, seed 1, at most 300 output tokens, JSON schema in `format`; the chat model is unloaded before each mode |
| Context | default (Ollama chose 4,096 tokens), or `num_ctx` 8192 |
| Runs | twice in Python and twice in JavaScript, 2026-11-12 |
| Node.js | 24.16 |

All four runs produced identical replies, retrieval scores, and token counts; only request ids and latencies differed. `latest` is a moving tag: check `ollama list` for the IDs above before comparing digits. With a different model or version, expect the same shape, not the same numbers.

```text
1. Retrieval: does an article that answers the question appear in the top K?

  a04 'How do I get my money back?': relevant ['refunds'], rank 2; top 3: trash 0.6619, refunds 0.6544, failed-payments 0.6457
  a21 'What does the Pro plan cost per year?': relevant ['annual-billing'], rank 2; top 3: plans 0.8594, annual-billing 0.7744, free-trial 0.6560
  Recall@1 = 22/24 = 0.92
  Recall@3 = 24/24 = 1.00
  Recall@5 = 24/24 = 1.00
  Top-1 similarity: answerable 0.6535 to 0.8954, not in docs 0.5925 to 0.8508

2. The whole application, four ways

                              answerable questions          not in docs
mode                      key fact  no key fact   declined      declined   prompt tokens
all, default context          3/24            2         19          6/6            2,050
all, 8K context              18/24            1          5          6/6            4,262
rag, top 1                   19/24            1          4          2/6              261
rag, top 3                   22/24            1          1          3/6              431
```

Speed varied between runs; these are the ranges across the four runs (median per question, and the first request of each mode, which includes loading the chat model):

| Mode | Median latency | First request | Prompt tokens reused from Ollama's cache (median) |
|---|---|---|---|
| all, default context | 2.05–2.88 s | 3.81–4.60 s | 2,034 |
| all, 8K context | 0.39–0.58 s | 6.65–8.27 s | 4,248 |
| rag, top 1 | 0.61–0.94 s | 4.01–6.39 s | 147 |
| rag, top 3 | 0.79–1.33 s | 1.39–2.59 s | 148 |

The cached-token column was added after the first pair of runs, so it comes from the second pair (identical in both languages).

## What we found by reading every reply

The pattern check is a first pass: it measures whether the expected fact appeared, not whether the answer was correct. Each run sends the 30 questions through all four modes, so one run produces 120 replies. After the first run (Python), all 120 were read by hand. That changed the patterns for three questions: a16's had passed a wrong answer, and a17's and a18's had failed correct ones. The saved replies were re-graded (`--regrade`), and the three later runs used the fixed patterns from the start. The replies of all four runs were compared in code and were identical, so the hand reading covers the reported results. The patterns were adjusted after seeing outputs: they are checked against hand labels, not fixed in advance. The hand reading also turned up what a pattern can't see:

- **Truncation.** With the default context, every request logged `msg="truncating input prompt" limit=2050 prompt=4259…4268 keep=4 new=2050` in the Ollama server log. Counting tokens for prefixes of the help center places the cut inside the 24th article: the instructions and the first 23 articles were dropped, and the last 26 articles and the question survived. With its instructions gone, the model invented details for the dropped articles ("The Basic plan supports up to 2 members", "The Pro plan is $12.50/user/month"); most were stopped by the application's checks, so the customer saw the decline message.
- **Right article, wrong answer.** "I deleted a project by mistake last week. Can I get it back?" retrieved the Trash article in both RAG modes, and the model answered "No, after 30 days, items are permanently deleted…": one of the article's own sentences with "No" in front. It reuses the 30-day fact but reverses its meaning; the project was deleted last week, so the answer should be yes.
- **Wrong article, caught.** With top 1, "What does the Pro plan cost per year?" retrieved the monthly pricing article; the model multiplied $25 by 12 and answered $300. The value check found $300 missing from the cited article and declined. With top 3 the annual-billing article was retrieved, and the answer was $250.
- **Right article, wrong value, not caught.** With the whole help center at 8K, the same question got "The Pro plan costs $100 per year" (the Basic price), citing the annual-billing article. The value check passed, because $100 is in that article.
- **Questions the help center doesn't answer.** RAG answered some of them from the closest articles: a Windows desktop app that doesn't exist (from the offline article, top 1), "There is no limit on the number of tasks" (top 1 and top 3), "We do not currently accept euros" (top 3), and "A refund is returned … within 30 days of the request" (top 3; the article says refunds can be *requested* within 30 days). One top-1 reply pointed to live chat "at [insert link to live chat]".
- **More context, more declines.** With the whole help center at 8K, the model declined five answerable questions that it answered with three retrieved articles (a04, a09, a10, a16; and a06, which RAG declined too). "Can I pay with PayPal?" was declined in every mode that had the payment article; the cause is not confirmed.
- **Two key-fact passes with a wrong detail.** Top 1 told a customer to enter the branch name in a "'Branch' field" that the help center doesn't mention (a20), and placed the data export under Settings > Billing (a22). The key facts were present, so the pattern counted them.
- **Degenerate output.** Some replies repeated the same source id until they hit the 300-token limit (for example, top 1 on the password question). The JSON was incomplete, and the application declined instead of showing it.

## Things to try

- `python ask.py "How do I get my money back?" --k 1`, then `--k 3`. Which article comes first, and why?
- `python ask.py "Is there a desktop app for Windows?" --k 1`: the help center doesn't mention one. What does the model say?
- Add a not-in-docs question of your own, and see what retrieval returns for it.
- Edit an article in `data/help_center.json` without re-running the indexer, and run `ask.py`.
- Run with `QUERY_PREFIX="" DOCUMENT_PREFIX=""` (and rebuild the index) to see how the prefixes affect Recall@1.
- Try `OLLAMA_CHAT_MODEL=qwen2.5:3b`: a different model can trade declines for wrong answers.

## Limitations

This is an educational demo, not a production RAG system.

- Each article is one chunk, because the articles are short. Real documents need to be split, and how you split them matters.
- Retrieval is dense-only and brute-force over 50 vectors in memory. There is no keyword search, no reranking, and no metadata filtering.
- 30 questions and one fictional help center are enough to show the shapes, not to measure a system. Recall@5 on 50 articles says little.
- The value-presence check only compares prices, durations, sizes, and emails. It can't tell whether a sentence is supported.
- Help-center articles here are trusted. In a real system, documents are input too: anything indexed can carry instructions the model may follow.
- Grading uses patterns plus a hand reading; there is no LLM judge.
