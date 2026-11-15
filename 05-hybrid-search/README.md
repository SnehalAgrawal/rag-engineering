# 05-hybrid-search

Companion code for **[Retrieval Strategies: Vector Search, Keyword Search, and Hybrid Search](https://snehalagrawal.com/articles/retrieval-strategies-vector-keyword-and-hybrid-search/)** (Article #16).

The same five documents and the same chunks as 04-chunking (`structure-headings`, its best strategy), the same 1,000-token retrieved-context budget, and five ways to rank the chunks:

| Method | How it ranks |
|---|---|
| `vector` | Cosine similarity between the question's and each chunk's embedding (`nomic-embed-text`), as in 01-04 |
| `keyword-words` | BM25 over the chunk text; the analyzer splits text into lowercase runs of letters and digits (`SSO-104` -> `sso`, `104`) |
| `keyword-identifiers` | BM25 with an analyzer that also keeps each identifier whole (`SSO-104` -> `sso`, `104`, `sso-104`) |
| `rrf` | Reciprocal rank fusion of the vector ranking and the keyword ranking: each adds 1 / (60 + rank). The constant 60 is the usual default; it can be changed |
| `weighted-0.8` | A weighted sum of the two scores, each normalized per query; the weight was chosen on development questions |

Only chunk ranking changes. Chunking, the budget, and the scoring are the ones from 04.

## The experiment, and what was fixed when

1. **The documents and the 60 questions from 04** were copied unchanged (`data/documents/`, `data/questions-04.json`).
2. **40 new questions** (`data/questions.json`) were written for where the methods are expected to differ, and committed before any keyword-search code existed: [`42bb215`](https://github.com/SnehalAgrawal/rag-engineering/commit/42bb215).

   | Kind | Count | The question |
   |---|---|---|
   | identifier | 15 | is built around an exact string: an error code, header, field, URL, or version |
   | paraphrase | 15 | shares at most one content word with the passage that answers it |
   | mixed | 10 | an exact string plus a concept in the question's own words |

3. **The settings** (`data/settings.json`) were written after the questions and before the code: BM25's parameters, both analyzers, RRF's constant, the fusion weights to try, and the two rules that use development questions only (which analyzer the hybrids use, and which weight).

Both question sets are split into development and test questions. The 60 from 04 are reported separately, so the new set, written to find differences, doesn't hide how much vector search already handles.

## How the methods are defined

```text
BM25 (as in Lucene)   score = sum over query tokens t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))
                      idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)),  k1 = 1.2,  b = 0.75
                      only chunks that contain at least one query token are returned
RRF                   score = sum over the two rankings of 1 / (60 + rank),  rank starting at 1
weighted              alpha * vector' + (1 - alpha) * keyword',  where x' = (x - min) / (max - min) over all 98 chunks,
                      per query; BM25 scores every chunk, so a chunk it doesn't match scores 0 before normalizing;
                      chunks with a combined score of 0 are dropped
```

Min-max normalization makes the two scores share a range for one query; it doesn't make a 0.9 mean the same thing for two different queries. Every ranking breaks ties by chunk order, so Python and JavaScript rank identically.

## Files

| File | What it does |
|---|---|
| `data/documents/*.md` | The five documents from 04-chunking, unchanged |
| `data/questions.json` | The 40 new questions, each with its evidence passages, kind, and split |
| `data/questions-04.json` | The 60 questions from 04-chunking, unchanged |
| `data/settings.json` | Chunking, budget, BM25, analyzers, RRF, fusion weights, and the selection rules |
| `search.*` | The analyzers, BM25, ranking, RRF, and weighted fusion |
| `evaluate.*` | Chunk, embed, index, rank every question five ways, apply the development-question choices, and print the tables |
| `chunkers.*`, `tokens.*`, `llm.*` | From 04-chunking, unchanged |

## Prerequisites

- Python 3.10+ **or** Node.js 18.2+ (tested with Python 3.12 and Node.js 24)
- [Ollama](https://ollama.com/download) with `nomic-embed-text` (`ollama pull nomic-embed-text`). No chat model is needed.
- The Llama 3.2 tokenizer, for counting the budget in the answer model's tokens: see [04-chunking: Get the tokenizer](../04-chunking/README.md#get-the-tokenizer), and put `tokenizer.json` in this demo's `data/` folder (or set `TOKENIZER_PATH`).

## How to run

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluate.py       # a few seconds: embeddings only
```

```bash
cd javascript
npm install
node evaluate.js
```

## Setup used for the published numbers

| | |
|---|---|
| Machine | Apple M4 Pro (24 GB), macOS 26.6.2 |
| Ollama | 0.40.16 |
| Embedding model | `nomic-embed-text`, ID `0a109f422b47`, prefixes `search_query: ` / `search_document: `, `truncate: false` |
| Tokenizer | Llama 3.2 `tokenizer.json`, SHA-256 `6b9e4e7f…` |
| Languages | Python 3.12.13 (requests 2.34.2, python-dotenv 1.2.4) and Node.js 24.16.0 (dotenv 16.6.1) |
| Runs | Twice in each language, 2026-11-15: identical output and identical saved rankings, apart from timings |

## Expected output

```text
1. How the analyzers turn identifiers into tokens
   SSO-104                           words: ['sso', '104']
                                     identifiers: ['sso', '104', 'sso-104']
   cursor_expired                    words: ['cursor', 'expired']
                                     identifiers: ['cursor', 'expired', 'cursor_expired']
   X-RateLimit-Remaining             words: ['x', 'ratelimit', 'remaining']
                                     identifiers: ['x', 'ratelimit', 'remaining', 'x-ratelimit-remaining']
   https://api.example.com/scim/v2   words: ['https', 'api', 'example', 'com', 'scim', 'v2']
                                     identifiers: ['https', 'api', 'example', 'com', 'scim', 'v2', 'https://api.example.com/scim/v2']

   98 chunks (structure-headings), 10913 tokens. BM25 index: words 1195 distinct tokens, identifiers 1319 distinct tokens

2. Choices made on the development questions (21 new + 31 from 04), at 1000 tokens
   keyword analyzer: words 42, identifiers 42 -> hybrids use 'words'
   weighted fusion, alpha = weight of the vector score:
   alpha   0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
   found    42   43   46   48   49   49   49   50   51   50   49   -> alpha 0.8

3. Evidence found at 1000 tokens
method                 new dev (21) new test (19)   04 dev (31)  04 test (29)
vector                           18            17            31            29
keyword-words                    13            17            29            28
keyword-identifiers              13            17            29            29
rrf                              19            18            30            29
weighted-0.8                     20            18            31            28

4. New test questions by kind, at 1000 tokens
method                 identifier (7)  paraphrase (7)       mixed (5)
vector                              6               6               5
keyword-words                       7               5               5
keyword-identifiers                 7               5               5
rrf                                 7               6               5
weighted-0.8                        7               6               5

5. All test questions (48), by budget
method                  500 tokens  1000 tokens  2000 tokens
vector                          41           46           48
keyword-words                   41           45           46
keyword-identifiers             41           46           46
rrf                             45           47           47
weighted-0.8                    43           46           48

6. Questions only one method found, at 1000 tokens (all 100 questions)
   vector only (10):
     r16 [new dev, paraphrase] How do I stop people signing in from outside our office network?
     r18 [new dev, paraphrase] Someone is taking a six-month sabbatical. Can I block them temporarily without them losing what they're responsible for?
     r20 [new dev, paraphrase] Our accountant needs copies of the bills automatically. How?
     r25 [new test, paraphrase] Is there a cap on how often a script can call you?
     r26 [new dev, paraphrase] Our card keeps getting declined. Will my team lose access?
     r30 [new dev, paraphrase] Can we move our account from the US servers to Germany?
     r39 [new dev, mixed] How much is Pro per year, and how does that compare with paying monthly?
     q28 [04 dev, heading] When did it become possible to suspend members?
     q55 [04 dev, multi] After we delete our organization, how long until our data is gone from backups too?
     q60 [04 test, multi] When our identity provider deactivates someone, what happens to the tasks assigned to them?
   keyword-words only (2):
     r10 [new test, identifier] What does a 503 with the code maintenance mean?
     r13 [new dev, identifier] What does include_completed=false do?
   found by vector or keyword-words, but not by rrf (2):
     r20 [new dev, paraphrase] Our accountant needs copies of the bills automatically. How?
     q28 [04 dev, heading] When did it become possible to suspend members?
   found by vector or keyword-words, but not by weighted-0.8 (1):
     q60 [04 test, multi] When our identity provider deactivates someone, what happens to the tasks assigned to them?
   found by no method (2):
     r28 [new dev, paraphrase] Who else handles our information on your behalf?
     r29 [new test, paraphrase] Do I pay extra for every person I add?
```

The last section, timings, varies from run to run. In these runs, embedding the 98 chunks took about 1.4 seconds, building a BM25 index 2 to 9 milliseconds, embedding one question about 8 to 9 milliseconds, and scoring one question with BM25 about 0.1 to 0.2 milliseconds.

## What we found

- **Vector search was already strong on these chunks.** It found the evidence for all 31 + 29 questions from 04, and 35 of the 40 new ones, at 1,000 tokens. The chunks carry their headings (04's best strategy), and the corpus is small.
- **Keyword search found what vector search ranked badly, and the reverse.** BM25 alone found two identifier questions that vector search missed (`503 ... maintenance`, `include_completed=false`); vector search alone found ten questions that BM25 (words analyzer) missed, six of them paraphrases; with the identifiers analyzer it's nine.
- **Hybrids helped most at a small budget.** On the 48 test questions, at 500 tokens: vector 41, keyword 41, RRF 45, weighted 43. At 2,000 tokens, vector alone found all 48.
- **RRF rewards agreement.** "Our accountant needs copies of the bills automatically" is ranked first by vector search; BM25 doesn't match that chunk at all, so RRF gives it only 1/61 and ranks chunks that both methods placed lower above it.
- **The identifier analyzer made no difference on the development questions** (42 and 42). In these documents the parts of identifiers (`104`, `cursor`, `idempotency`) are already rare words. Its one difference on a test question, which has no identifier in it, came from the extra tokens changing document lengths and word rarity.
- **Two questions were found by no method**: "Who else handles our information on your behalf?", whose answer is the subprocessor table (its chunk wasn't in any method's top ten), and "Do I pay extra for every person I add?", whose answer is one sentence above the plans table: vector search ranked that 199-token chunk eighth, when it no longer fit into what was left of the budget.

## Things to try

- Add stop words or stemming to the `words` analyzer, and see which paraphrase questions BM25 stops finding by accident ("can", "how", "long") and which it starts finding on purpose ("signing" -> "sign").
- Change `rrf.k` to 1 or 1,000 and watch how much the top of each list counts.
- Use `fixed-256` chunks from 04 instead of `structure-headings`: without headings in the chunk, does keyword search matter more?

## Limitations

- 98 chunks of one fictional product. A 1,000-token budget holds about a tenth of them, which makes "found within budget" easy for every method; with thousands of chunks, the differences grow.
- One embedding model, one BM25 configuration, no stemming, no stop words, and no reranking (the next article).
- The weighted fusion's weight was chosen on 52 development questions; with 11 candidate weights and differences of one or two questions, it is a rough choice, not a tuned optimum.
