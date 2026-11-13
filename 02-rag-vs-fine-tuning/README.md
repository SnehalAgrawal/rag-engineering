# RAG vs Fine-Tuning: Knowledge in the Prompt or in the Weights

A small, real comparison of fine-tuning and retrieval on the same model and the same questions. It accompanies the article **[RAG vs Fine-Tuning: When Should You Use Which?](https://snehalagrawal.com/articles/rag-vs-fine-tuning-when-should-you-use-which/)** (Article #13), and continues the support assistant and help center from [01-basic-rag](../01-basic-rag/).

The demo fine-tunes Llama 3.2 1B Instruct with LoRA on question-and-answer pairs written from the help center, then compares four ways of answering:

| Condition | Model | What it gets with the question |
|---|---|---|
| `base` | as downloaded | a one-line system prompt |
| `fine-tuned` | with the trained LoRA adapter | the same one-line system prompt |
| `base + rag` | as downloaded | the 3 retrieved articles, and instructions to answer only from them |
| `fine-tuned + rag` | with the adapter | the 3 retrieved articles and the same instructions |

Then five facts change in the help center (prices, a link expiry, an invitation window, a member limit), and the same questions are asked again.

## What it demonstrates

1. **Fine-tuning taught behavior reliably.** Every training answer (except the declines) ended with `Source: <article title>`. The fine-tuned model ended 22 of 22 answers that way without being asked; the base model, told to do it in its prompt, did it in 1 of 17.
2. **Fine-tuning taught facts unreliably.** On questions worded like the training data, 14 of 24 answers were correct. Worded differently (the test questions), 7 of 24 were correct, and 6 more had the right fact mixed with false or garbled statements.
3. **Changed facts reach RAG, not the weights.** After 5 articles changed and only those 5 were re-embedded, base + RAG gave the new value for all 5 questions (2 with an error elsewhere in the reply). The fine-tuned model gave the old value for 3 and something else for 2.
4. **Fine-tuning on the wrong format can hurt RAG.** The adapter was trained without retrieved articles in the prompt. With retrieval added, the fine-tuned model used the articles worse than the base model did: 6 of 24 correct against 10, and 1 of 5 changed facts against 5.

All of this is one 1B model, 120 training examples, and 30 questions: a demonstration of the shapes, not a benchmark. See Limitations.

## Flow

```text
train.py      training_qa.json ──► chat examples (fixed date) ──► LoRA on the 4-bit model ──► adapters/
evaluate.py   50 articles ──► Ollama embeddings ──► top-3 retrieval (as in 01-basic-rag)
              4 conditions × (24 test + 24 training-wording + 6 not-in-docs questions)
              5 articles change ──► re-embed those 5 ──► 5 questions again, 3 conditions
              ──► results.jsonl, graded by the hand labels in data/hand_labels.json
```

| File | Purpose |
|---|---|
| `data/help_center.json`, `data/test_questions.json` | The help center and test questions from 01-basic-rag (copied, unchanged) |
| `data/training_qa.json` | 112 answers written from the articles, 8 out-of-scope questions to decline, and 12 validation examples |
| `data/hand_labels.json` | A label for every reply in the reported runs, with a hash of the exact reply it belongs to |
| `python/common.py` | Model, prompts, retrieval, generation |
| `python/train.py` | LoRA fine-tuning (QLoRA, because the base model is 4-bit) |
| `python/evaluate.py` | The four conditions, the changed-facts test, and the report |
| `python/ask.py` | One question, any condition |

The training questions never repeat a test question. Each of the 24 test facts has a "twin" training question that covers it in different words (`twin` in `training_qa.json`), so the evaluation can compare the training wording with new wording.

## Prerequisites

- **A Mac with Apple silicon.** Training and generation use [MLX](https://github.com/ml-explore/mlx), which runs on Apple silicon only. On other hardware, the same experiment can be built with Hugging Face PEFT; it isn't included here.
- Python 3.10+
- [Ollama](https://ollama.com/download) with `nomic-embed-text`, for retrieval: `ollama pull nomic-embed-text`
- About 1 GB of disk for the model. The first run downloads [`mlx-community/Llama-3.2-1B-Instruct-4bit`](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit) (0.7 GB, pinned to revision `08231374eeacb049a0eade7922910865b8fce912`) from Hugging Face. The model is under the Llama 3.2 Community License.

## How to run

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py        # about 50 s on an M4 Pro; writes ../adapters/
python evaluate.py     # 231 replies, about 2 minutes
python ask.py "How much is the Pro plan?" --fine-tuned
python ask.py "How much is the Pro plan?" --rag
```

`python evaluate.py --regrade results.jsonl` reports a saved run again without loading any model.

## Expected output

The numbers below come from this setup:

| Setting | Value |
|---|---|
| Machine | Apple M4 Pro (24 GB), macOS 26.6, on battery power |
| Libraries | Python 3.12.13, mlx 0.32.4, mlx-lm 0.32.2, huggingface_hub 2.4.0 |
| Chat model | `mlx-community/Llama-3.2-1B-Instruct-4bit`, revision `08231374…`, weights SHA-256 `35e39664…` |
| Embeddings | Ollama 0.40.16, `nomic-embed-text` (`0a109f422b47`), prefixes `search_query: ` / `search_document: ` |
| Fine-tuning | LoRA rank 8, scale 20, all 16 layers, 5,636,096 trained parameters; batch 4, learning rate 1e-4, 100 steps (about 3.3 passes over 120 examples), seed 0, prompt masked |
| Generation | temperature 0 (greedy), at most 200 tokens; chat template date fixed to "12 Nov 2026" |
| Runs | trained 3 times (identical adapter: SHA-256 `f6ddfcd96f306e3f…`), evaluated twice (identical replies), 2026-11-13 |

Training: validation loss 5.32 before, 1.10 after; training loss 0.26 at the last report.

```text
1. Hand labels: answerable questions and questions not in the help center

                           test wording (24)           training wording (24)  not in docs (6)
condition           correct   partly   wrong  declined       correct   partly       declined
base                      0        2      21         1             0        1              0
fine-tuned                7        6       9         2            14        3              3
base + rag               10        6       3         5            11        8              3
fine-tuned + rag          6        7      10         1             7        7              3

2. Hand labels: 5 test questions after their facts changed in 5 articles

condition           new fact  new, with an error  old fact  other
fine-tuned                 0                   0         3      2
base + rag                 3                   2         0      0
fine-tuned + rag           1                   0         1      3

3. Checks in code: the 'Source:' line, and the key-fact pattern as a cross-check

condition           source line present  names a right article  key fact (pattern)
base                              0/24                   0/24                 5/24
fine-tuned                       22/22                  17/22                14/24
base + rag                        1/17                   0/17                15/24
fine-tuned + rag                 22/23                  16/23                15/24
```

If you change anything (model, settings, data), the replies change, and `evaluate.py` lists them as unlabeled: read them yourself. Expect the same shapes, not the same numbers.

## How the replies were graded

A key-fact pattern (like 01-basic-rag's) was the first pass, and it was misleading in both directions here. The fine-tuned model often produced the right number inside a wrong or self-contradicting answer ("No. Single sign-on is available on the Basic plan."), and correct answers sometimes missed a pattern ("Only the Owners can delete the organization."). So every reply was read and labeled by hand:

- **correct**: the key fact is right and nothing in the reply is false.
- **partly**: the key fact is right, but the reply also says something false, contradictory, or garbled.
- **wrong**: an answer without the right key fact.
- **declined**: no answer.

The labels and the hash of each labeled reply are in `data/hand_labels.json`; both evaluation runs produced exactly those replies.

## What we found by reading every reply

- **The base model doesn't know the product, and says so rarely.** It invented "$49.99 per month" with a limited-time promotion, a 30-day trial, a 1,000-member limit, and Android apps from other companies. Its 5 key-fact pattern hits were all guesses; one contained a template placeholder: "[insert timeframe, e.g., 30 days]".
- **The fine-tuned model recombines what it learned.** Fragments from other training answers turn up where they don't belong: "Ask for invoice payment through live chat" after a price; "We lock accounts for 30 minutes … Resetting your password is not possible"; "Press N, or use the New Task button" for how to link a pull request. It answered "Yes. We accept PayPal", the opposite of its training answer, and the rate-limit answer looped ("per key per key per minute…") until the token limit.
- **It learned the format perfectly, and the citations less so.** 22 of 22 answers ended with a source line; 17 named an article that answers the question. The other five named a related but wrong article, or an article that doesn't exist ("Deleting your project", "Cancelling your organization").
- **The 1B model with retrieval is uneven too.** Base + RAG declined 5 test questions whose answer was in the retrieved articles, sometimes started with the decline sentence and then answered anyway, and often pasted whole articles. It followed the "Source:" instruction once in 17 answers.
- **After the update.** Base + RAG gave all 5 new values; two replies added a false statement ("no member limit, meaning you can have up to 10", and "$300 per year, which is the same as the Basic plan"). The fine-tuned model said "1 hour", "$25 per month", and "up to 5 members", the old values, plus two garbled answers. With retrieval, the fine-tuned model still said "1 hour" with the article saying 2 hours in front of it, and gave invented prices ($100).
- **Training choices.** A first training run with LoRA rank 16 (at scale 20) diverged: the loss never fell below its starting value. With mlx-lm's default rank 8, the validation loss was lowest at 100 steps and flat from there to 300, so 100 steps was used. Longer adapters were not evaluated.

## Things to try

- `python ask.py "When do invitations expire?" --fine-tuned`: a question from the training data. Then ask "How long does a teammate have to accept an invitation?".
- Change a fact in `data/help_center.json` and ask with `--rag`, then with `--fine-tuned`.
- Add retrieved articles to the training prompts (train the way you serve), retrain, and compare `fine-tuned + rag` again.
- Train for 300 steps (`SETTINGS["iters"]` in `train.py`) and see whether the test-wording answers improve, or only the training-wording ones.

## Limitations

- One small model (1B parameters, 4-bit) and one small, synthetic help center. A larger model would do better in every condition; the comparison is about shapes, not absolute accuracy.
- 120 training examples, about two per fact. Fine-tuning for knowledge usually needs many more varied examples per fact; that is part of its cost.
- The adapter was trained without retrieved context, which is the setup most likely to hurt `fine-tuned + rag`. Training with retrieved documents in the prompt is a known alternative (see RAFT in the article's sources); it isn't tested here.
- Hand labels are one reader's judgment; the definitions are above, and every labeled reply is saved so you can check them.
- Python only, Apple silicon only.
