"""
Shared pieces for train.py, evaluate.py, and ask.py: the model, the prompts,
the data, retrieval (embeddings from Ollama, as in 01-basic-rag), and generation.
"""
import json
import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ADAPTER_PATH = ROOT / "adapters"

# The chat model, pinned to one revision of the 4-bit MLX conversion of Llama 3.2 1B Instruct.
MODEL_REPO = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MODEL_REVISION = "08231374eeacb049a0eade7922910865b8fce912"
# The Llama 3.2 chat template writes a date into every prompt ("Today Date: ...") and uses
# today's date when none is given. A fixed date keeps training and evaluation identical on any day.
DATE_STRING = "12 Nov 2026"

ARTICLES = json.loads((DATA / "help_center.json").read_text())["articles"]
TRAINING = json.loads((DATA / "training_qa.json").read_text())
TEST = json.loads((DATA / "test_questions.json").read_text())
DECLINE = TRAINING["decline_answer"]

SYSTEM = "You are the support assistant for our product."
RAG_SYSTEM = (
    SYSTEM + " Answer the customer's question in one or two sentences using only the help center articles "
    "below, then add a last line 'Source: <title of the article you used>'. If the articles don't answer "
    f"the question, reply exactly: {DECLINE}\n\nHelp center articles:\n\n{{articles}}"
)

EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QUERY_PREFIX, DOCUMENT_PREFIX = "search_query: ", "search_document: "


def titles(articles: list[dict]) -> dict:
    return {a["id"]: a["title"] for a in articles}


def training_answer(example: dict) -> str:
    """The answer the model is trained to give: the fact, then the article it came from."""
    if example.get("decline"):
        return DECLINE
    return f"{example['answer']}\n\nSource: {titles(ARTICLES)[example['article']]}"


# --- The model (MLX) ---------------------------------------------------------------

def model_path() -> str:
    from huggingface_hub import snapshot_download

    try:  # use the cached copy without contacting Hugging Face
        return snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    except Exception:  # not cached yet (or incomplete): download it (about 0.7 GB)
        return snapshot_download(MODEL_REPO, revision=MODEL_REVISION)


def load_model(fine_tuned: bool):
    from mlx_lm import load

    adapter = str(ADAPTER_PATH) if fine_tuned else None
    if fine_tuned and not (ADAPTER_PATH / "adapters.safetensors").exists():
        raise SystemExit("No fine-tuned adapter found. Run: python train.py")
    return load(model_path(), adapter_path=adapter)


def prompt_tokens(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                         date_string=DATE_STRING)


def generate(model, tokenizer, messages: list[dict], max_tokens: int = 200) -> str:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    tokens = prompt_tokens(tokenizer, messages, add_generation_prompt=True)
    # temperature 0: always the most likely token, so the same weights give the same reply
    return mlx_generate(model, tokenizer, prompt=tokens, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)).strip()


def messages_for(question: str, context: list[dict] | None = None) -> list[dict]:
    """No context: the short system prompt the model was fine-tuned with. With context: the RAG prompt."""
    if context is None:
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    text = "\n\n".join(f"## {a['title']}\n{a['text']}" for a in context)
    return [{"role": "system", "content": RAG_SYSTEM.replace("{articles}", text)}, {"role": "user", "content": question}]


# --- Retrieval (Ollama embeddings, as in 01-basic-rag) ------------------------------

def embed(texts: list[str]) -> list[list[float]]:
    response = requests.post(f"{OLLAMA_BASE_URL}/api/embed",
                             json={"model": EMBEDDING_MODEL, "input": texts, "truncate": False}, timeout=120)
    response.raise_for_status()
    return response.json()["embeddings"]


def unload_embedding_model() -> None:
    requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": EMBEDDING_MODEL, "keep_alive": 0},
                  timeout=60).raise_for_status()


def document_text(article: dict) -> str:
    return f"{DOCUMENT_PREFIX}{article['title']}\n{article['text']}"


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        raise ValueError("zero-length vector")
    return dot / norm


class Index:
    """Article vectors in memory, keyed by article id and the exact text that was embedded."""

    def __init__(self, articles: list[dict], reuse: "Index | None" = None):
        self.articles = articles
        texts = [document_text(a) for a in articles]
        known = reuse.vectors if reuse else {}
        missing = [t for t in texts if t not in known]
        new = dict(zip(missing, embed(missing))) if missing else {}
        self.vectors = {t: known.get(t) or new[t] for t in texts}
        self.embedded = len(missing)  # how many articles had to be (re-)embedded

    def search(self, question: str, k: int) -> list[dict]:
        query = embed([QUERY_PREFIX + question])[0]
        scored = [(cosine(query, self.vectors[document_text(a)]), i) for i, a in enumerate(self.articles)]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [self.articles[i] for _, i in scored[:k]]
