# Production RAG Pipeline: Deadlines, Index Versions, a Relevance Gate, and Traces

The support assistant from [01-basic-rag](../01-basic-rag/), rebuilt as a pipeline that can be operated, in Python and JavaScript. It accompanies the article **[Designing a Production-Ready RAG Pipeline](https://snehalagrawal.com/articles/designing-a-production-ready-rag-pipeline/)** (Article #14).

The prompt, the models, the JSON schema, and the output checks are the same as 01's. What changes is everything around them:

| Concern | 01-basic-rag | This demo |
|---|---|---|
| Index | Rebuilt from scratch; the assistant refuses to start if any article changed | Versioned snapshot; only changed articles are re-embedded; the new version replaces the old one in one step; the assistant keeps serving the last complete version and reports how stale it is (articles added, changed, or removed since the build) |
| Slow or failing model calls | 60 s timeout for the embedding call, 120 s for the chat call | One 10-second deadline for the whole request; each call gets the time that is left; every failure is named by call and kind |
| Questions the help center doesn't answer | Always sends the top 3 articles to the model | A relevance gate declines without calling the model when the best article scores below a threshold chosen on development questions |
| What happened to a request | A log record | One trace per request: versions, settings, stage timings, retrieval scores, gate decision, tokens, outcome; `report` summarizes traces |

Both behaviors live in one codebase as two profiles in `data/settings.json`: **basic** reproduces 01-basic-rag, **production** turns on the deadline and the gate. With the same question and the same retrieved articles, both profiles send the model a byte-identical request.

## Flow

```text
Index time (when articles change):
  help center ──► fingerprint each article ──► embed only new or changed ones ──► new index version
              ──► write to a temporary file, rename over the old index (readers see old or new, never half)

Query time (one deadline for the whole request):
  validate ──► retrieve (embedding call + cosine search) ──► relevance gate ──► generate (chat call, JSON schema)
           ──► check (shape, cited ids, values) ──► reply
  every step adds to one trace: index version and staleness, timings, scores, gate decision, tokens, outcome
```

| File | Purpose |
|---|---|
| `data/help_center.json`, `data/questions.json`, `data/prompt.json` | The 50 articles, the 30 test questions, and the prompt from 01-basic-rag, unchanged |
| `data/dev_questions.json` | Development questions for choosing the threshold: 123 answerable (02-rag-vs-fine-tuning's training and validation questions) and 19 not in the help center (9 from 02, 10 new) |
| `data/settings.json` | The two profiles: `k`, `deadline_seconds`, `relevance_threshold`, input and output limits |
| `data/hand_labels.json` | Hand labels for every answer to an answerable test question, keyed by question id and a hash of the exact reply |
| `index_store.*` | The index as a build artifact: version, incremental build, atomic write, staleness |
| `index.*` | Brings `data/index.json` up to date with the help center |
| `pipeline.*` | The assistant: `Assistant(profile).answer(question)` returns the reply and its trace; never raises |
| `llm.*` | The two Ollama calls; every failure becomes `LLMError` with a kind: `unavailable`, `timeout`, `http_error`, `invalid_response` |
| `ask.*` | Ask one question and see each step and the trace |
| `evaluate.*` | Retrieval metrics, the threshold choice on the dev questions, then the 30 test questions through both profiles |
| `faults.*` | Fault injection through a stand-in server: hangs, errors, malformed replies, a slow chat call, Ollama not running |
| `update.*` | The index lifecycle: a stale index, incremental re-indexing, a removed article (works on a temporary copy) |
| `report.*` | The operator's view: what the traces alone say about saved runs |

## Prerequisites

- Python 3.10+ **or** Node.js 18.2+ (tested with Python 3.12 and Node.js 24)
- [Ollama](https://ollama.com/download), with the two models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Configuration

Copy `.env.example` to `.env` in `python/` or `javascript/` to change the Ollama URL, the models, or the embedding prefixes (rebuild the index afterwards). The defaults work without a `.env` file. Pipeline settings are in `data/settings.json`.

## How to run

### Python

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python index.py
python ask.py "Can I bring in tasks from a spreadsheet?"
python evaluate.py       # about 2 minutes: 60 chat requests
python faults.py         # about 4 minutes, mostly waiting for timeouts
python update.py         # under a minute
python report.py         # reads the results files the three scripts above saved
```

### JavaScript

```bash
cd javascript
npm install
node index.js
node ask.js "Can I bring in tasks from a spreadsheet?"
node evaluate.js
node faults.js
node update.js
node report.js
```

`evaluate --retrieval-only` runs parts 1 and 2 (no chat model). `evaluate --regrade FILE` re-grades saved replies without calling any model. `ask --profile basic` uses 01's behavior. Every script that calls a model unloads both models when it finishes.

## Expected output

The numbers below come from this setup:

| Setting | Value |
|---|---|
| Machine | Apple M4 Pro (24 GB), macOS 26.6, on AC power |
| Ollama | 0.40.16, `/api/embed` and `/api/chat` |
| Chat model | `llama3.2:3b`, ID `a80c4f17acd5` (Q4_K_M; weights blob `sha256-dde5aa3fc5ff…`) |
| Embedding model | `nomic-embed-text:latest`, ID `0a109f422b47` (weights blob `sha256-970aa74c0a90…`), 768 dimensions, prefixes `search_query: ` / `search_document: ` |
| Generation | temperature 0, seed 1, at most 300 output tokens, JSON schema in `format`, Ollama's default context (4,096) |
| Pipeline | top 3; production: 10-second deadline, relevance threshold 0.65 |
| Runtimes | Python 3.12.13 (requests 2.34.2, python-dotenv 1.2.4); Node.js 24.16.0 (dotenv 16.6.1) |
| Runs | evaluate, faults, and update twice in each language, 2026-11-13 |

The embedding vectors were identical to 01-basic-rag's index, and with the basic profile all 30 replies, retrieved articles, and prompt token counts were identical to 01's saved top-3 run. Across the four evaluation runs, every reply, retrieval score, and token count was identical; only request ids and timings differed. The Python and JavaScript outputs were identical apart from timings. `latest` is a moving tag: check `ollama list` for the IDs above before comparing digits. With a different model or version, expect the same shape, not the same numbers.

### 1-2. Retrieval, and the threshold chosen on the development questions

```text
1. Retrieval on the test questions: does an article that answers the question appear in the top K?

  a04 'How do I get my money back?': relevant ['refunds'], rank 2; top 3: trash 0.6619, refunds 0.6544, failed-payments 0.6457
  a21 'What does the Pro plan cost per year?': relevant ['annual-billing'], rank 2; top 3: plans 0.8594, annual-billing 0.7744, free-trial 0.6560
  Recall@1 = 22/24 = 0.92
  Recall@3 = 24/24 = 1.00
  Recall@5 = 24/24 = 1.00
  Top-1 similarity: answerable 0.6535 to 0.8954, not in docs 0.5925 to 0.8508

2. The relevance threshold, chosen on the development questions

  123 answerable dev questions, top-1 similarity 0.6059 to 0.8727
  19 not-in-docs dev questions, top-1 similarity 0.4950 to 0.7508

  threshold   answerable blocked   not in docs caught
       0.55                0/123                 2/19
       0.60                0/123                 4/19
       0.62                1/123                 8/19
       0.64                1/123                14/19
       0.65                1/123                16/19
       0.66                3/123                16/19
       0.68                7/123                17/19
       0.70               19/123                17/19

  Rule: block at most 1% of answerable dev questions (1), rounded down to 0.01 -> 0.65. settings.json uses 0.65.
  blocked (answerable): d119 'Do you support SAML?' 0.6059
  passed (not in docs): o11 'Can I set dependencies between tasks?' 0.6685
  passed (not in docs): o16 'Can I print a list of tasks?' 0.7140
  passed (not in docs): o19 'Is there a browser extension?' 0.7508
```

The threshold rule was fixed before the test questions went through the gate: the highest threshold, rounded down to 0.01, that blocks at most 1% of the answerable development questions. The test questions' score ranges had already been published with Article #12; the rule doesn't use them. Some development questions cover the same facts as test questions in other words (02's "twins"), so the two sets are related, not independent.

### 3. The test questions through both profiles

```text
3. The test questions through both profiles

                      answerable (hand labels)              not in docs       chat
profile       correct  partly  wrong  declined unlabeled   declined answered calls
basic              22       0      1         1         0       3/6         3    30
production         22       0      1         1         0       3/6         3    27

correct: the key fact is right and nothing in the reply is false. partly: the key fact is right, but the
reply also says something false. wrong: answered without the right key fact. declined: the customer got
the decline message (the gate, the model, or a check in code). chat calls: requests that reached the chat model.

Where the gate let a question through (27 of 30), the production reply was identical to the basic one for 27 of 27.

Blocked by the gate (production):
  n01 (not in docs) top score 0.6452: 'Do you have a phone number I can call?'
  n03 (not in docs) top score 0.5925: 'Do you offer a discount for nonprofits?'
  n05 (not in docs) top score 0.6381: 'Is there a desktop app for Windows?'
```

Speed varied between runs. Across the four evaluation runs (two per language), per profile:

| Profile | First request (both models loading) | Median per question | Median chat call |
|---|---|---|---|
| basic | 1.74–2.66 s | 0.75–0.79 s | 0.73–0.78 s |
| production | 1.63–1.64 s | 0.77–0.82 s | 0.80–0.87 s |

The slowest answered request in the production profile was the first one, at 1.63–1.64 s, well inside the 10-second deadline.

### Faults

The hang timings were the same in all four runs; the 12-second chat call took 12.4 s (Python) and 12.5 s (JavaScript) in the basic profile.

```text
Question: 'How long is the password reset link valid?'

scenario                          profile       waited  status  outcome      failed call
healthy                           basic          0.4 s     200  answered
healthy                           production     0.4 s     200  answered
Ollama not running                basic          0.0 s     503  unavailable  retrieve: unavailable
Ollama not running                production     0.0 s     503  unavailable  retrieve: unavailable
embedding call hangs              basic         60.0 s     503  unavailable  retrieve: timeout
embedding call hangs              production    10.0 s     503  unavailable  retrieve: timeout
embedding call returns HTTP 500   basic          0.0 s     503  unavailable  retrieve: http_error
embedding call returns HTTP 500   production     0.0 s     503  unavailable  retrieve: http_error
chat call hangs                   basic        120.0 s     503  unavailable  generate: timeout
chat call hangs                   production    10.0 s     503  unavailable  generate: timeout
chat call takes 12 s              basic         12.4 s     200  answered
chat call takes 12 s              production    10.0 s     503  unavailable  generate: timeout
chat call returns HTML, not JSON  basic          0.0 s     503  unavailable  generate: invalid_response
chat call returns HTML, not JSON  production     0.0 s     503  unavailable  generate: invalid_response
chat connection closed, no reply  basic          0.0 s     503  unavailable  generate: unavailable
chat connection closed, no reply  production     0.0 s     503  unavailable  generate: unavailable

answered: the customer saw 'The password reset link expires after 1 hour.'

unavailable: the customer saw 'Sorry, the assistant is unavailable right now. Please try again in a few minutes, or contact our support team via live chat.'
```

### The index lifecycle

```text
1. Build the index
   from scratch: embedded 50, reused 0, removed 0 -> version eae26b6c1519 (was none)
   again, nothing changed: embedded 0, reused 50, removed 0 -> version eae26b6c1519 (unchanged)

2. Billing changes 5 articles (6 edits). Articles changed since the index was built: 5

3. Before re-indexing, the running assistant still serves the old version:
  a01 [index eae26b6c1519, 5 stale] 'How long is the password reset link valid?'
      The password reset link expires after 1 hour.
  a02 [index eae26b6c1519, 5 stale] 'How much is the Pro plan?'
      The Pro plan costs $25 per month and $250 per year.
  a11 [index eae26b6c1519, 5 stale] 'How many members can I have on the Basic plan?'
      The Basic plan includes up to 5 members.
  a21 [index eae26b6c1519, 5 stale] 'What does the Pro plan cost per year?'
      The Pro plan costs $250 per year.
  a24 [index eae26b6c1519, 5 stale] 'How long does a teammate have to accept an invitation?'
      Invitations expire after 7 days.

4. Re-index: embedded 5, reused 45, removed 0 -> version f701f0419d49 (was eae26b6c1519)
   The same assistant, on its next requests:
  a01 [index f701f0419d49, 0 stale] 'How long is the password reset link valid?'
      The password reset link expires after 2 hours.
  a02 [index f701f0419d49, 0 stale] 'How much is the Pro plan?'
      The Pro plan costs $30 per month and $300 per year.
  a11 [index f701f0419d49, 0 stale] 'How many members can I have on the Basic plan?'
      The Basic plan includes up to 10 members.
  a21 [index f701f0419d49, 0 stale] 'What does the Pro plan cost per year?'
      The Pro plan costs $300 per year.
  a24 [index f701f0419d49, 0 stale] 'How long does a teammate have to accept an invitation?'
      Invitations expire after 14 days.

5. The 'sso' article is removed. Re-index: embedded 0, reused 49, removed 1 -> version 367ec836ebce (was f701f0419d49)
   entries in the index: 49
```

## What we found by reading every reply

All 23 answers to answerable test questions were read against the help center and labeled (`data/hand_labels.json`); the six not-in-docs questions and the five update-scenario questions were read too.

- **The gate blocked only questions the model would have declined anyway.** On the test set it blocked n01 (phone number), n03 (nonprofit discount), and n05 (Windows app); the basic profile had declined all three. The three not-in-docs questions that got a wrong answer scored above the threshold: "How many days does it take for a refund to arrive?" 0.8508 (answered "within 30 days of the request"; the article says refunds can be *requested* within 30 days), "Can I pay in euros?" 0.6545 ("We do not currently accept euros", which the help center doesn't say), and "What is the maximum number of tasks in one project?" 0.7109 ("There is no limit"). What the gate saved here was 3 chat calls, not 3 wrong answers.
- **The margin is thin.** On the development questions the highest not-in-docs score below the threshold was 0.6492, and the lowest answerable score above it 0.6536. The one answerable development question it blocks is "Do you support SAML?" (0.6059): a short query with one acronym.
- **One answerable reply was wrong in both profiles:** "I deleted a project by mistake last week. Can I get it back?" got "No, after 30 days, items are permanently deleted…", the failure Article #12 described. Retrieval was right; the gate can't catch it.
- **Borderline replies labeled correct**, because nothing in them is false: a04 says to "contact our support team" (the article doesn't say how to request a refund); a20 describes configuring the GitHub integration to link pull requests (the article says linking happens when the task ID is in the branch name, with no configuration step); a10's text is right ("SSO is not available on the Basic plan"), but it cites `getting-started`, which doesn't mention SSO. The application accepts any cited id that was retrieved; it doesn't check that the cited article supports the sentence.
- **Stale answers are a choice.** Before re-indexing, the assistant answered "$25 per month" while the help center already said $30, and each of those traces recorded 5 changed articles. 01-basic-rag would have refused to answer at all. After the re-index, all five answers had the new values, with nothing false.
- **A timed-out request is not a cancelled one.** The first version of the stand-in server kept sleeping after the production profile gave up on the 12-second chat call, then forwarded the request to Ollama anyway. That reloaded the chat model after the script had unloaded it. The stand-in now drops requests whose client has disconnected. Real servers differ in whether they stop work when a client disconnects; check yours.

## Things to try

- `python ask.py "Do you support SAML?"`, then with `--profile basic`: the gate blocks a question the help center answers.
- Change `relevance_threshold` to 0.60 or 0.68 and run `evaluate.py`: the warning shows you left the rule, and the tables show what each choice costs.
- Change `deadline_seconds` to 1 and run `evaluate.py` right after unloading the models (`ollama stop llama3.2:3b`): does the first request survive loading the model?
- Add a fault to `SCENARIOS` in `faults.py`, for example `{"/api/embed": "slow:3"}`: how much of the deadline is left for the chat call?
- In `update.py`, ask a question about the removed `sso` article after step 5. What does the gate do, and what does the model answer from the remaining articles?
- Try `OLLAMA_EMBEDDING_MODEL` with another embedding model: the index rebuilds completely, and the threshold has to be chosen again, because scores from different models aren't comparable.

## Limitations

This is an educational reference, not a production system.

- One process, one machine. There is no web server, no concurrency, no queue, and no shared index store; the "atomic swap" is a file rename on one file system. The assistant checks the file on every request and loads a new version; several workers would each have to do that, and could briefly answer from different versions.
- No retries, no circuit breaker, no fallback when the embedding call fails: failures are named and fail fast. Retries are a separate topic (Article #37 in the series).
- In Python, `requests` applies the timeout to connecting and to each wait for data, not to the whole response, so a server that sends a reply slowly in pieces could take longer than the deadline, and one number applies separately to connecting and to reading, so the two waits can add up. With `stream: false`, Ollama sends its reply once generation has finished. In JavaScript, `AbortSignal.timeout` limits the whole request.
- The relevance gate is one number on cosine similarity, chosen on 142 development questions. It is specific to this embedding model, these prefixes, and this help center.
- Each article is one chunk; retrieval is dense-only over 50 vectors in memory; there is no reranking. Those are the next articles in the series.
- Traces are JSON lines saved by the scripts, not a tracing system: no spans across services, no sampling, no retention policy.
- Grading uses hand labels for the answerable questions; there is no automatic judge. Questions the help center doesn't answer are counted as declined or answered from the reply (any answer is a failure there), and every one of those replies was read.
