# rag-engineering

Runnable retrieval-augmented generation (RAG) demos that accompany the AI engineering series on [snehalagrawal.com](https://snehalagrawal.com/articles/). Each demo is self-contained and runs locally: no API keys or paid services. 01, 03, 04, 05, and 06 run in Python and JavaScript with [Ollama](https://ollama.com); 02 fine-tunes a model, so it is Python only and needs a Mac with Apple silicon ([MLX](https://github.com/ml-explore/mlx)).

| Demo | Article | What it shows |
|---|---|---|
| [01-basic-rag](01-basic-rag/) | [RAG Explained: How AI Systems Work With Your Data](https://snehalagrawal.com/articles/rag-explained-how-ai-systems-work-with-your-data/) | Retrieval versus sending the whole knowledge base, on the same questions: what fits, what it costs, and what still breaks |
| [02-rag-vs-fine-tuning](02-rag-vs-fine-tuning/) | [RAG vs Fine-Tuning: When Should You Use Which?](https://snehalagrawal.com/articles/rag-vs-fine-tuning-when-should-you-use-which/) | A real LoRA fine-tune against retrieval on the same model: what each one teaches, and what happens when a fact changes (Python, Apple silicon) |
| [03-production-rag-pipeline](03-production-rag-pipeline/) | [Designing a Production-Ready RAG Pipeline](https://snehalagrawal.com/articles/designing-a-production-ready-rag-pipeline/) | The 01 assistant with a pipeline around it: a versioned, incremental index; one deadline per request; a relevance gate; fault injection; and a trace for every request |
| [04-chunking](04-chunking/) | [Chunking Strategies for RAG: What Actually Matters?](https://snehalagrawal.com/articles/chunking-strategies-for-rag-what-actually-matters/) | Ten ways to split five long documents, compared on whether the answering passage is retrieved at equal token budgets, plus a hand-graded answer stage |
| [05-hybrid-search](05-hybrid-search/) | [Retrieval Strategies: Vector Search, Keyword Search, and Hybrid Search](https://snehalagrawal.com/articles/retrieval-strategies-vector-keyword-and-hybrid-search/) | Vector, BM25, and two hybrids ranking the same chunks within the same budget: what each finds that the others miss |
| [06-reranking](06-reranking/) | [Reranking: Improving Retrieval Quality](https://snehalagrawal.com/articles/reranking-improving-retrieval-quality/) | A second stage that reads the question and each candidate together: candidate recall, what reranking fixes and breaks, and what it costs |

## Linting

The repository has one lint setup for every demo: [Ruff](https://docs.astral.sh/ruff/) for Python and [ESLint](https://eslint.org/) for JavaScript.

```bash
npm install                              # ESLint (repository root)
pip install -r requirements-dev.txt      # Ruff
npm run lint                             # both
```

## Conventions

- Every demo has a README with the exact setup that produced the numbers in its article.
- Generated files (`data/index.json`, `results-*.jsonl`) and `.env` files are not committed.
