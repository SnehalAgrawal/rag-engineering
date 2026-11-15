# 06-reranking

Companion code for **[Reranking: Improving Retrieval Quality](https://snehalagrawal.com/articles/reranking-improving-retrieval-quality/)** (Article #17).

Retrieval in two stages. The **first stage** finds candidates cheaply: vector search, or the RRF hybrid from 05-hybrid-search. The **second stage** reorders only those candidates with a reranker that reads the question and each candidate together. The budget is then filled from the reranked order. A chunk that isn't among the candidates can never be chosen.

The reranker here is a chat model, `llama3.2:3b`, asked whether a passage answers the question. Its score is the log-odds of its first token, `log P(Yes) - log P(No)`, read from the log-probabilities Ollama returns. This shows the mechanism and its cost. It is **not** a purpose-trained reranker (a cross-encoder), and the results say nothing about how well one of those would do.

Everything else is 05's, unchanged: the five documents, the `structure-headings` chunks, the 100 questions with their development/test split, and the budgets.

## The experiment, and what was fixed when

1. **The questions** are 05-hybrid-search's 40 and 04-chunking's 60, both frozen before this demo existed. Three of them are the cases 05 couldn't solve, which motivated this demo.
2. **The settings and the prompt** (`data/settings.json`, `data/prompt.json`) were written before any reranking run: the two first stages, candidate counts 10 and 20, the reranker model and options, the label tokens, and the scoring rule. After the runs, one sentence in the settings' comment was corrected (about the prompt cache, see below); no setting changed.
3. **A 50-pair timing batch** came first, to measure the cost and check that the scores repeat. It found that they sometimes don't (see below), which decided how the full run is done.

## How a pair is scored

```text
messages   system: You judge whether a passage from a product's documentation answers a customer's
                   question. Reply with one word: Yes or No.
           user:   Question: ...  Passage: ...  Does this passage answer the question? Reply Yes or No.
request    llama3.2:3b, temperature 0, seed 1, one output token, logprobs, top_logprobs 20
score      logsumexp(log P of Yes, yes, YES, " Yes", " yes") - logsumexp(log P of No, no, NO, " No", " no")
           among the 20 alternatives returned at the first position (an estimate of the log-odds: a variant
           outside them isn't counted); a label with no variant among them would get the lowest listed
           log-probability. Each label is a single token in Llama 3.2's tokenizer.
order      candidates by score, highest first; ties by first-stage rank
```

`coverage_check.py` rescored all 2,581 pairs to record which variants came back: "Yes", "No", and "NO" in every pair, "yes" in 1,253, "YES" in 2,012, " Yes" in 1, " yes", " No", and " no" in none. For each pair, the bound counts every missing variant on both sides: each is at most as likely as the 20th alternative, and the largest possible change, log(1 + missing x that probability / the observed sum), is computed for the "yes" and the "no" side; the larger of the two bounds the score's change in either direction. It was at most 0.0025 (median 0.00005). Only 2 of the 5,600 neighbouring pairs in the reranked orders are closer than the sum of their two bounds, both at positions 18 and 19 of a 20-candidate list; swapping either pair changes no result.

Every (question, chunk) pair in the top 20 of either first stage is scored once: 2,581 pairs. A pair's score doesn't depend on which first stage proposed it, so the same score is used for both, and for both candidate counts.

**Scores repeat when a pair is scored fresh, but not always when it's scored again while its prompt is still in Ollama's prompt cache.** In the timing batch, 22 of 50 pairs scored a second time, with the model still loaded, came back slightly different (by up to 0.017 in a follow-up probe of 12 pairs). One probe: the same prompt evaluated cold or after a different prompt gave yes log-probability -0.3929; sent again with 99 of its 100 tokens served from the cache, -0.3908. So the full run scores each pair once, and `repeat_check.py` scores a seeded sample of 250 pairs again after the model has been reloaded, in shuffled order: all 250 came back identical.

## Files

| File | What it does |
|---|---|
| `data/documents/`, `data/questions.json`, `data/questions-04.json` | From 05-hybrid-search, unchanged |
| `data/settings.json` | First stages, candidate counts, budgets, the reranker, and the scoring rule |
| `data/prompt.json` | The reranker prompt (`rerank-v1`) |
| `first_stage.*` | Chunk, embed, and rank every chunk by vector and by RRF; list the candidate pairs |
| `rerank.*` | The prompt, one scoring call, and the log-odds score |
| `evaluate.*` | Score every candidate pair, then candidate recall and evidence found, before and after reranking; writes `results-scores.jsonl` and `results-rerank.json` |
| `timing.py` | The 50-pair batch run before the full experiment |
| `repeat_check.py` | Scores 250 sampled pairs again and compares |
| `coverage_check.py` | Rescores every pair and records which label variants were among the 20 alternatives, and how much the missing ones could move each score; writes `results-coverage.jsonl` |
| `search.*`, `chunkers.*`, `tokens.*`, `llm.*` | From 05-hybrid-search (`llm.*` adds the log-probability call) |

## Prerequisites

- Python 3.10+ **or** Node.js 18.2+ (tested with Python 3.12 and Node.js 24)
- [Ollama](https://ollama.com/download) with `nomic-embed-text` and `llama3.2:3b`
- The Llama 3.2 tokenizer in `data/tokenizer.json`: see [04-chunking: Get the tokenizer](../04-chunking/README.md#get-the-tokenizer)

## How to run

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluate.py       # about 10 minutes: 2,581 scoring calls
python repeat_check.py   # about a minute: 250 calls
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
| Reranker | `llama3.2:3b`, ID `a80c4f17acd5` (Q4_K_M), temperature 0, seed 1, one output token, `top_logprobs` 20 |
| Embedding model | `nomic-embed-text`, ID `0a109f422b47` |
| Languages | Python 3.12.13 (requests 2.34.2, python-dotenv 1.2.4) and Node.js 24.16.0 (dotenv 16.6.1) |
| Runs | The full run once in each language, 2026-11-15: the same 2,581 pairs in the same order, the same first token and prompt-token count for every pair, scores equal to within 2e-15 (floating-point arithmetic in the two languages), identical rankings and tables. Then 250 pairs again in Python, and all 2,581 twice more (`coverage_check.py`): identical. |

## Expected output

```text
1. Candidate recall: the evidence is somewhere in the first stage's top N (no budget)
first stage                     new dev (21) new test (19)   04 dev (31)  04 test (29)
vector, top 10                            20            19            31            29
vector, top 20                            21            19            31            29
rrf, top 10                               19            18            31            29
rrf, top 20                               19            18            31            29
(all 98 chunks)                           21            19            31            29

2. Evidence found within each budget, all test questions (48)
first stage, candidates, order   500 tokens  1000 tokens  2000 tokens
vector, all, first stage                 41           46           48
vector, top 10, first stage              41           46           48
vector, top 10, reranked                 46           48           48
vector, top 20, first stage              41           46           48
vector, top 20, reranked                 45           47           48
rrf, all, first stage                    45           47           47
rrf, top 10, first stage                 45           47           47
rrf, top 10, reranked                    44           47           47
rrf, top 20, first stage                 45           47           47
rrf, top 20, reranked                    44           46           47

3. Evidence found at 1000 tokens, by question group
first stage, candidates, order  new dev (21) new test (19)   04 dev (31)  04 test (29)
vector, top 10, first stage               18            17            31            29
vector, top 10, reranked                  20            19            31            29
vector, top 20, first stage               18            17            31            29
vector, top 20, reranked                  21            18            31            29
rrf, top 10, first stage                  19            18            30            29
rrf, top 10, reranked                     19            18            31            29
rrf, top 20, first stage                  19            18            30            29
rrf, top 20, reranked                     19            17            30            29

4. New test questions by kind, at 1000 tokens
first stage, candidates, order  identifier (7)  paraphrase (7)       mixed (5)
vector, top 10, first stage                  6               6               5
vector, top 10, reranked                     7               7               5
vector, top 20, first stage                  6               6               5
vector, top 20, reranked                     7               7               4
rrf, top 10, first stage                     7               6               5
rrf, top 10, reranked                        7               6               5
rrf, top 20, first stage                     7               6               5
rrf, top 20, reranked                        7               6               4

5. What reranking changed, at 500 and 1000 tokens (all 100 questions)
   vector, top 10, 500 tokens: fixed 9, broke 2
     fixed r10 [test, identifier] What does a 503 with the code maintenance mean?
     fixed r13 [dev, identifier] What does include_completed=false do?
     fixed r19 [test, paraphrase] Can coworkers with our corporate address get in on their own, without me adding each one?
     fixed r24 [dev, paraphrase] Can I see who did what, and when, in our workspace?
     fixed r29 [test, paraphrase] Do I pay extra for every person I add?
     fixed r32 [test, mixed] We get 429 even though we only retry failed requests. Why?
     fixed r39 [dev, mixed] How much is Pro per year, and how does that compare with paying monthly?
     fixed q36 [test, table] Can a Member restore a deleted project?
     fixed q60 [test, multi] When our identity provider deactivates someone, what happens to the tasks assigned to them?
     broke r36 [test, mixed] Do people created through SCIM become Admins?
     broke q57 [dev, multi] Sign-in fails with SSO-101. How do I fix it?
   vector, top 10, 1000 tokens: fixed 4, broke 0
     fixed r10 [test, identifier] What does a 503 with the code maintenance mean?
     fixed r13 [dev, identifier] What does include_completed=false do?
     fixed r24 [dev, paraphrase] Can I see who did what, and when, in our workspace?
     fixed r29 [test, paraphrase] Do I pay extra for every person I add?
   vector, top 20, 500 tokens: fixed 10, broke 5
     fixed r10 [test, identifier] What does a 503 with the code maintenance mean?
     fixed r13 [dev, identifier] What does include_completed=false do?
     fixed r19 [test, paraphrase] Can coworkers with our corporate address get in on their own, without me adding each one?
     fixed r24 [dev, paraphrase] Can I see who did what, and when, in our workspace?
     fixed r28 [dev, paraphrase] Who else handles our information on your behalf?
     fixed r29 [test, paraphrase] Do I pay extra for every person I add?
     fixed r32 [test, mixed] We get 429 even though we only retry failed requests. Why?
     fixed r39 [dev, mixed] How much is Pro per year, and how does that compare with paying monthly?
     fixed q36 [test, table] Can a Member restore a deleted project?
     fixed q60 [test, multi] When our identity provider deactivates someone, what happens to the tasks assigned to them?
     broke r17 [test, paraphrase] A laptop was stolen. Can I force all our staff to log in again immediately?
     broke r36 [test, mixed] Do people created through SCIM become Admins?
     broke r37 [dev, mixed] Does Sign out all members log me out too?
     broke q57 [dev, multi] Sign-in fails with SSO-101. How do I fix it?
     broke q59 [dev, multi] How soon can we get written confirmation that our deleted organization's data is gone?
   vector, top 20, 1000 tokens: fixed 5, broke 1
     fixed r10 [test, identifier] What does a 503 with the code maintenance mean?
     fixed r13 [dev, identifier] What does include_completed=false do?
     fixed r24 [dev, paraphrase] Can I see who did what, and when, in our workspace?
     fixed r28 [dev, paraphrase] Who else handles our information on your behalf?
     fixed r29 [test, paraphrase] Do I pay extra for every person I add?
     broke r36 [test, mixed] Do people created through SCIM become Admins?
   rrf, top 10, 500 tokens: fixed 4, broke 5
     fixed r24 [dev, paraphrase] Can I see who did what, and when, in our workspace?
     fixed r39 [dev, mixed] How much is Pro per year, and how does that compare with paying monthly?
     fixed q28 [dev, heading] When did it become possible to suspend members?
     fixed q60 [test, multi] When our identity provider deactivates someone, what happens to the tasks assigned to them?
     broke r17 [test, paraphrase] A laptop was stolen. Can I force all our staff to log in again immediately?
     broke r33 [dev, mixed] Will page and per_page pagination still work next year?
     broke r35 [dev, mixed] We get 403 forbidden when we call the API from home. Could that be the office-only network setting?
     broke r36 [test, mixed] Do people created through SCIM become Admins?
     broke q59 [dev, multi] How soon can we get written confirmation that our deleted organization's data is gone?
   rrf, top 10, 1000 tokens: fixed 1, broke 0
     fixed q28 [dev, heading] When did it become possible to suspend members?
   rrf, top 20, 500 tokens: fixed 4, broke 7
     fixed r24 [dev, paraphrase] Can I see who did what, and when, in our workspace?
     fixed r39 [dev, mixed] How much is Pro per year, and how does that compare with paying monthly?
     fixed q28 [dev, heading] When did it become possible to suspend members?
     fixed q60 [test, multi] When our identity provider deactivates someone, what happens to the tasks assigned to them?
     broke r17 [test, paraphrase] A laptop was stolen. Can I force all our staff to log in again immediately?
     broke r33 [dev, mixed] Will page and per_page pagination still work next year?
     broke r35 [dev, mixed] We get 403 forbidden when we call the API from home. Could that be the office-only network setting?
     broke r36 [test, mixed] Do people created through SCIM become Admins?
     broke r37 [dev, mixed] Does Sign out all members log me out too?
     broke q57 [dev, multi] Sign-in fails with SSO-101. How do I fix it?
     broke q59 [dev, multi] How soon can we get written confirmation that our deleted organization's data is gone?
   rrf, top 20, 1000 tokens: fixed 1, broke 2
     fixed q28 [dev, heading] When did it become possible to suspend members?
     broke r36 [test, mixed] Do people created through SCIM become Admins?
     broke q59 [dev, multi] How soon can we get written confirmation that our deleted organization's data is gone?

6. The three cases from 05: where the chunk with the evidence ranks
   r20 Our accountant needs copies of the bills automatically. How?
     vector: first-stage rank [1], rank after reranking the top 20 [1], score [-3.533]
     rrf: first-stage rank [24], rank after reranking the top 20 [None], score [-3.533]
   r29 Do I pay extra for every person I add?
     vector: first-stage rank [8], rank after reranking the top 20 [1], score [-0.768]
     rrf: first-stage rank [69], rank after reranking the top 20 [None], score [-0.768]
   r28 Who else handles our information on your behalf?
     vector: first-stage rank [20], rank after reranking the top 20 [4], score [-1.494]
     rrf: first-stage rank [36], rank after reranking the top 20 [None], score [-1.494]
```

The last section, the cost, varies: the median scoring call took 166 ms in the Python run and 184 ms in the JavaScript run (503 s and 550 s for all 2,581 calls), so reranking 10 candidates for one question took about 1.7 to 1.8 seconds and 20 candidates about 3.3 to 3.7, one call after another. Embedding a question took about 8 ms in 05.

`repeat_check.py`:

```text
250 pairs scored again in shuffled order (seed 7)
identical: 250; differences: median 0.0000, max 0.0000
pairs whose change is at least the gap to the nearest other score for the same question: 0
```

## What we found

- **Candidate recall is the ceiling.** Vector search's top 20 held the evidence for all 100 questions; RRF's top 20 missed 3, and no reranking of RRF's candidates can find those.
- **Reranking vector search's top 10 helped most.** On the 48 test questions: 46 found within 500 tokens instead of 41, and 48 within 1,000 instead of 46. Reranking the top 20 did a little worse (45 and 47): more candidates gave the reranker more wrong chunks to prefer.
- **Reranking RRF's candidates didn't help** (44 instead of 45 at 500 tokens, 46 or 47 instead of 47 at 1,000). RRF's order was already good where it had the evidence.
- **It fixes and it breaks.** For vector search's top 10 at 500 tokens, reranking fixed 9 questions and broke 2; for the top 20, it fixed 10 and broke 5.
- **A poor judge, a useful ranker.** The model answered "No" for 2,519 of the 2,581 pairs, including 73 of the 125 pairs whose chunk holds evidence. But the evidence chunks' median score was -0.42 against -4.69 for the rest, and for 84 of the 100 questions the highest-scored candidate held evidence.
- **The three cases from 05:** the per-person pricing chunk (vector rank 8) went to rank 1; the subprocessor table (vector rank 20) went to rank 4, but only with 20 candidates; the billing contact chunk RRF had pushed to rank 24 was never a candidate from RRF, so reranking RRF's top 20 couldn't recover it.
- **It costs seconds, not milliseconds**: 10 to 20 model calls per question, against one embedding.

## Things to try

- Change the prompt in `data/prompt.json` (for example, ask whether the passage is relevant instead of whether it answers), and see how the scores and the fixed/broken lists move.
- Send the candidates in parallel instead of one after another, and measure the latency per question.
- Use `llama3.2:1b` as the reranker (`OLLAMA_CHAT_MODEL`), and compare quality and cost.

## Limitations

- A 3B chat model used as a reranker, with a yes/no prompt. Purpose-trained rerankers are faster and likely better; this demo doesn't measure one.
- 98 chunks of one fictional product and 100 questions; differences of one or two questions are not meaningful.
- Calls are made one after another, with no batching; the cost per question reflects that.
