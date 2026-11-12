# rag-engineering

Runnable retrieval-augmented generation (RAG) demos that accompany the AI engineering series on [snehalagrawal.com](https://snehalagrawal.com/articles/). Each demo is self-contained, in Python and JavaScript, and runs locally with [Ollama](https://ollama.com): no API keys or paid services.

| Demo | Article | What it shows |
|---|---|---|
| [01-basic-rag](01-basic-rag/) | [RAG Explained: How AI Systems Work With Your Data](https://snehalagrawal.com/articles/rag-explained-how-ai-systems-work-with-your-data/) | Retrieval versus sending the whole knowledge base, on the same questions: what fits, what it costs, and what still breaks |

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
