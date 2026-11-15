# 04-chunking

Companion code for **[Chunking Strategies for RAG: What Actually Matters?](https://snehalagrawal.com/articles/chunking-strategies-for-rag-what-actually-matters/)** (Article #15).

The help center in 01-03 had short articles, so each one was a single chunk. Real documentation isn't like that. This demo takes five long documents for the same fictional product, splits them nine ways (plus a whole-document baseline that doesn't split), and measures what each does to retrieval and to the answers.

| Strategy | How it splits |
|---|---|
| `whole-document` | Doesn't: one chunk per document |
| `fixed-128`, `-256`, `-512` | Windows of 128, 256, or 512 tokens, with and without overlap (a quarter of the window) |
| `structure` | One chunk per Markdown section; a section over 512 tokens is split between paragraphs, then between lines |
| `structure-headings` | The same chunks, with the heading lines above each one copied in front of it |
| `semantic` | Splits between sentences where the meaning shifts most, measured with embeddings (here: above the 95th percentile of the distances in each document) |

Every strategy is scored the same way: do the retrieved chunks contain the passages that answer a question? Each required passage must be whole inside one retrieved chunk; a question that needs two passages can get them from two chunks. Strategies are compared at **equal retrieved-context budgets**, because bigger chunks trivially contain more: chunks are taken in rank order, a chunk that doesn't fit the remaining budget is skipped, and the budget counts only the retrieved chunks (in the answer stage, the instructions and question added 169 to 299 more tokens).

## The experiment, and what was fixed when

1. **The documents** (`data/documents/`, about 7,800 words) were written and committed first.
2. **The questions** (`data/questions.json`, 60) were written after the documents and committed with them, in one commit, before any chunker existed: [`3aa330a`](https://github.com/SnehalAgrawal/rag-engineering/commit/3aa330a). Each question lists the exact passages that answer it. A table row comes with its header row, a procedure with all its steps, a release note with its version heading. The scoring rule is in the file.
3. **The settings** (`data/settings.json`) were written before the first retrieval run: the ten strategies, the budgets (500, 1,000, 2,000 tokens), and the selection rule for the answer stage. The answer stage's output limit was raised from 300 to 500 tokens before any answer was generated, so that an 8-step answer can't be cut off.

The questions are split into 31 development and 29 test questions (alternating within each kind of question). The development questions choose which strategies go to the answer stage; the test questions report the results.

| Kind of question | Count | The evidence |
|---|---|---|
| simple | 21 | one sentence |
| heading | 13 | a release note and its version heading (the date is only in the heading) |
| table | 14 | a table row and the header row |
| list | 6 | every step of a procedure, or every item of a list |
| multi | 6 | two places, often linked by a cross-reference |

## How it works

```text
documents ──► chunker ──► chunks (each a slice of one document, plus copied headings for structure-headings)
          ──► embed every chunk (nomic-embed-text, "search_document: ")
question  ──► embed ("search_query: ") ──► rank chunks by cosine similarity
          ──► fill the budget: chunks in rank order, skipping any that no longer fit
          ──► evidence found? (every passage of one alternative inside a single chunk)

answer stage (test questions): baseline + the two strategies ranked best on the dev questions
          ──► the budgeted chunks to llama3.2:3b ──► reply ──► hand label
```

Token counts are **Llama 3.2 tokens**, the answer model's, counted by `tokens.py` / `tokens.js`: a dependency-free implementation of its byte-level BPE tokenizer. `check_tokens.*` compares it with Ollama's own prompt counts on 80 texts (14,809 tokens): no mismatches.

## Files

| File | What it does |
|---|---|
| `data/documents/*.md` | The five documents: an administrator guide, an API reference, a billing guide, a data processing policy, and release notes |
| `data/questions.json` | 60 questions, each with its evidence passages, kind, and split |
| `data/settings.json` | Strategies, budgets, the selection rule, and the answer-stage settings |
| `data/prompt.json` | The answer prompt and its JSON schema (`docs-v1`) |
| `data/hand_labels.json` | Hand labels for the answer stage, keyed by question and a hash of the exact reply, with notes |
| `tokens.*` | Llama 3.2 token counting, from the model's `tokenizer.json` |
| `chunkers.*` | The chunking strategies |
| `evaluate.*` | Chunk, embed, retrieve, and score every strategy; writes `results-retrieval.json` and `results-chunks.jsonl` |
| `answer.*` | The answer stage; writes `results-answers.jsonl`. `--regrade` reprints the table from saved replies |
| `check_tokens.*` | Compares the token counts with Ollama's |

The documents and questions are ASCII-only, so the tokenizer's letter and digit classes can be written as `[A-Za-z]` and `[0-9]`, and Python and JavaScript split text identically. Both refuse non-ASCII input rather than count it wrongly.

## Prerequisites

- Python 3.10+ **or** Node.js 18.2+ (tested with Python 3.12 and Node.js 24)
- [Ollama](https://ollama.com/download), with the two models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Get the tokenizer

The token counter reads Llama 3.2's `tokenizer.json` (17 MB, under the Llama 3.2 Community License), which isn't stored in this repository. Download it into `data/`:

```bash
curl -L -o data/tokenizer.json \
  https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit/resolve/08231374eeacb049a0eade7922910865b8fce912/tokenizer.json
shasum -a 256 data/tokenizer.json   # 6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b
```

The 1B and 3B Llama 3.2 models share this tokenizer; `check_tokens.*` confirms it against `llama3.2:3b`. If you already have the file elsewhere, set `TOKENIZER_PATH` instead.

## How to run

### Python

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python check_tokens.py   # 80 raw prompts to llama3.2:3b
python evaluate.py       # about 30 seconds: embeddings only
python answer.py         # about 4 minutes: 87 chat requests
```

### JavaScript

```bash
cd javascript
npm install
node check_tokens.js
node evaluate.js
node answer.js
```

`answer.*` uses the ranking saved by `evaluate.*`, so run that first. Every script unloads the models when it finishes.

## Setup used for the published numbers

| | |
|---|---|
| Machine | Apple M4 Pro (24 GB), macOS 26.6.2 |
| Ollama | 0.40.16 |
| Chat model | `llama3.2:3b`, ID `a80c4f17acd5` (Q4_K_M), temperature 0, seed 1, at most 500 output tokens, JSON schema in `format` |
| Embedding model | `nomic-embed-text`, ID `0a109f422b47`, prefixes `search_query: ` / `search_document: `, `truncate: false` |
| Tokenizer | Llama 3.2 `tokenizer.json`, SHA-256 `6b9e4e7f…` |
| Languages | Python 3.12.13 (requests 2.34.2, python-dotenv 1.2.4) and Node.js 24.16.0 (dotenv 16.6.1) |
| Date | 2026-11-15 |

Python and JavaScript produce identical output: the same 570 chunks (offsets, headings, token counts, text hashes), the same rankings and scores, and the same 87 replies, byte for byte. Retrieval ran three times in Python and twice in JavaScript, and the answer stage twice in each language; every run was identical.

## Expected output

### evaluate

```text
1. The chunks, and what indexing cost (embedding tokens are nomic-embed-text tokens)
strategy                chunks  tokens: min  median   max  indexed  intact  embedded texts  embedding tokens
whole-document               5         1185    1790  2808    10147   60/60               5              8998
                         admin-guide.md: the input length exceeds the context length; embedded with truncate=true, which reads about the first 1959 of 2808 tokens
                         api-reference.md: the input length exceeds the context length; embedded with truncate=true, which reads about the first 1836 of 2604 tokens
                         questions whose evidence the embedding never read: 9
fixed-128                   81           33     128   128    10147   48/60              81             11155
fixed-128-overlap-32       106           33     128   128    13379   60/60             106             14695
fixed-256                   41           44     256   256    10147   53/60              41             10915
fixed-256-overlap-64        53          108     256   256    13219   60/60              53             14240
fixed-512                   23           44     512   512    10147   57/60              23             10807
fixed-512-overlap-128       27          224     512   512    12963   60/60              27             13821
structure                   98           20      83   424    10147   60/60              98             11257
structure-headings          98           34      90   428    10913   60/60              98             11916
semantic                    38            2     275   675    10132   60/60             654             46470  (incl. 616 sentence windows to find the splits)

2. Evidence found, dev questions (31): within a context budget, and in the top k chunks
strategy                 500 tokens 1000 tokens 2000 tokens   top 1 (tokens)   top 3 (tokens)   top 5 (tokens)
whole-document                    0           0          15      25   (1922)      31   (5554)      31  (10147)
fixed-128                        21          23          24      16    (127)      20    (381)      21    (631)
fixed-128-overlap-32             22          28          30      16    (128)      22    (381)      27    (635)
fixed-256                        18          25          26      18    (252)      25    (750)      26   (1239)
fixed-256-overlap-64             18          26          30      16    (252)      26    (761)      29   (1259)
fixed-512                         2          20          29      16    (437)      26   (1261)      29   (2066)
fixed-512-overlap-128             5          16          29      13    (453)      25   (1370)      30   (2334)
structure                        27          30          31      21    (103)      25    (340)      29    (552)
structure-headings               31          31          31      25    (124)      29    (373)      30    (615)
semantic                         15          22          27      14    (277)      19    (951)      25   (1530)

2. Evidence found, test questions (29): within a context budget, and in the top k chunks
strategy                 500 tokens 1000 tokens 2000 tokens   top 1 (tokens)   top 3 (tokens)   top 5 (tokens)
whole-document                    0           0          14      23   (2026)      27   (5793)      29  (10147)
fixed-128                        19          22          23      15    (125)      19    (381)      22    (637)
fixed-128-overlap-32             22          27          27      13    (125)      22    (381)      26    (636)
fixed-256                        15          24          27      15    (253)      24    (760)      25   (1268)
fixed-256-overlap-64             15          25          28      14    (256)      25    (767)      26   (1267)
fixed-512                         1          16          25      13    (473)      24   (1334)      26   (2176)
fixed-512-overlap-128             2          19          26      19    (501)      26   (1453)      28   (2441)
structure                        26          29          29      22    (142)      26    (370)      28    (570)
structure-headings               26          29          29      22    (165)      26    (383)      29    (622)
semantic                          9          21          24       9    (338)      21    (984)      22   (1520)

3. Evidence found at 1000 tokens, by kind of question (dev + test)
strategy                  simple (21)  heading (13)    table (14)      list (6)     multi (6)
whole-document                      0             0             0             0             0
fixed-128                          16            11             9             4             5
fixed-128-overlap-32               21            12            12             5             5
fixed-256                          18            12             9             5             5
fixed-256-overlap-64               19            12            10             6             4
fixed-512                          15             8             6             5             2
fixed-512-overlap-128              13             8             8             5             1
structure                          21            12            14             6             6
structure-headings                 21            13            14             6             6
semantic                           16             8            11             6             2

4. Selection on the dev questions at 1000 tokens: structure-headings 31, structure 30, fixed-128-overlap-32 28, fixed-256-overlap-64 26, fixed-256 25, fixed-128 23, semantic 22, fixed-512 20, fixed-512-overlap-128 16, whole-document 0
   Answer stage: fixed-256 (baseline), structure-headings, structure
```

`intact` counts the questions whose evidence survives chunking at all, that is, it would be found if every chunk were retrieved. Indexing took about 1 second per strategy, except `semantic`, which spent another 5 seconds embedding sentence windows to decide where to split (timings vary from run to run).

### answer

```text
Answers to the 29 test questions (hand labels)
strategy             correct  partly  wrong  declined  unlabeled  evidence found  context tokens  prompt tokens
fixed-256                 21       0      1         7          0              24      median 973    median 1152
structure-headings        22       0      2         5          0              29      median 989    median 1206
structure                 22       1      3         3          0              29      median 993    median 1217

By kind of question: correct / questions
strategy               simple (10)   heading (6)     table (7)      list (3)     multi (3)
fixed-256                        8             6             3             3             1
structure-headings               6             6             4             3             3
structure                        8             5             3             3             3

Evidence found in the context, and the answer
  fixed-256           evidence found (24): 21 correct, 1 wrong, 2 declined
  fixed-256           evidence missing (5): 5 declined
  structure-headings  evidence found (29): 22 correct, 2 wrong, 5 declined
  structure           evidence found (29): 22 correct, 1 partly, 3 wrong, 3 declined
```

Every reply was read. The labels and a note for each one that isn't plainly correct are in `data/hand_labels.json`.

## What we found

- **Whole documents don't work here.** Two of the five documents are longer than the embedding model accepts (2,048 tokens in this setup). With `truncate: false` the call fails; with Ollama's default, `truncate: true`, only the beginning is embedded, without a warning, and 9 of the 60 questions have their evidence in the part that was never embedded. And a whole document doesn't fit a 1,000-token context at all.
- **Chunk boundaries destroy evidence.** Fixed windows without overlap cut the evidence of 12 questions (128 tokens), 7 (256), and 3 (512): a table row separated from its header, a list cut in half. Overlap kept every passage intact, at the cost of 28 to 32% more indexed tokens.
- **Compare at equal budgets.** "Top 3" gives `fixed-512` 26 of 31 development questions, with about 1,260 tokens of context; `structure` finds 25 with 340. At 1,000 tokens, only one 512-token chunk fits (two would be 1,024), and `fixed-512` finds 20. At 2,000 tokens it finds 29. Chunk size and budget interact.
- **Following the document's structure did best** on these documents, which are clean Markdown with many short sections: every passage intact, and 30 to 31 of 31 development questions found at 1,000 tokens. Copying the headings into each chunk fixed the one structural miss, a release note two headings below its version.
- **Semantic splitting used 3.2 to 4.3 times as many embedding tokens and did worse** than most fixed windows here (22 of 31 development questions at 1,000 tokens). It split a table header from its rows and produced chunks from 2 to 675 tokens; 17 questions had intact evidence that it didn't retrieve within 1,000 tokens.
- **Better retrieval moved the bottleneck to the model.** With the evidence in the context for all 29 test questions, `llama3.2:3b` still declined 3 to 5 and got 2 to 3 wrong. The baseline found the evidence for 24 and answered 21 correctly; the structural strategies found it for 29 and answered 22.

## Things to try

- Add a strategy to `data/settings.json`, for example `{"name": "structure-256", "kind": "structure", "max_tokens": 256, "headings": true}`, and run `evaluate` again. Then try a budget of 1,024 instead of 1,000.
- Replace a document with one of your own (ASCII Markdown), write a few questions with evidence passages, and see which strategy wins on it.
- Change the answer model with `OLLAMA_CHAT_MODEL` and regrade: are the declines with evidence in context a property of the 3B model?

## Limitations

- Five fictional documents, written for this demo in clean Markdown, and 60 questions. Strategies that use the Markdown structure benefit from it; PDFs, scanned documents, or wikis without headings would look different.
- One embedding model, cosine similarity, no keyword search, no reranking. Retrieval improvements beyond chunking are the next articles.
- The selection rule picked two very similar strategies (`structure` and `structure-headings`) for the answer stage; a fixed-size strategy with overlap came third.
- The answer stage is 29 questions with one 3B model, graded by one person. Differences of one or two answers are not meaningful.
