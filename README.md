# Local AI Agent with RAG

A **local agentic RAG** demo built with **LangChain v1**, **Ollama**, and **Chroma**.

Unlike a basic retrieve-then-answer script, this project is an **agent**: the model decides when to search reviews, filter by rating, or pull stats before answering.

Everything runs on your machine. No cloud LLM API keys required.

The shipped sample domain is **restaurant reviews** (pizza CSV). The same pattern works for any review CSV with columns `Title`, `Date`, `Rating`, `Review` — point `CSV_PATH` (and related settings) in [`constants.py`](constants.py).

## What it solves

**Problem:** You have a pile of customer reviews and want grounded answers like:

- What do people say about a specific topic?
- What are the main 1–2 star complaints?
- Draft a short owner reply about a recurring issue (e.g. late delivery).

**Approach:** Index reviews in a local vector DB, then give the agent tools so it can look things up only when needed.

## Architecture

```text
CSV reviews
   -> Ollama embeddings (nomic-embed-text)
   -> Chroma vector store

User question
   -> LangChain create_agent + ChatOllama
   -> optional tool calls:
        search_reviews
        filter_reviews_by_rating
        get_rating_stats
   -> grounded answer
```

| File | Role |
| --- | --- |
| `constants.py` | Models, paths, temperature, retriever `k` |
| `app.py` | Streamlit chat UI |
| `main.py` | CLI chat loop |
| `agent.py` | LangChain v1 agent + Ollama checks |
| `tools.py` | Agent tools (search / filter / stats) |
| `vector.py` | Load CSV, embed, persist Chroma |
| `logging_setup.py` | Console + file logging |
| `realistic_restaurant_reviews.csv` | Sample review dataset |

## Prerequisites

1. [Ollama](https://ollama.com/) installed and running
2. [uv](https://docs.astral.sh/uv/) installed
3. Python 3.14 (see `.python-version`)
4. Pull the local models (lightweight defaults for limited disk):

```bash
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

Rough sizes: `qwen2.5:0.5b` ~397 MB, `nomic-embed-text` ~274 MB.

If you have more RAM/disk and want stronger tool calling, raise `CHAT_MODEL` in `constants.py` to `qwen2.5:1.5b` / `qwen2.5:3b` / `qwen2.5`, and optionally `EMBEDDING_MODEL = "mxbai-embed-large"`. After changing the embedding model, delete `chroma_langchain_db/` (or use **Reindex reviews** in the Streamlit sidebar) so it re-indexes.

## Setup

```bash
git clone https://github.com/Namumbya/LocalAIAgentWithRAG.git
cd LocalAIAgentWithRAG

uv sync
```

## Run

**Web UI (recommended):**

```bash
uv run streamlit run app.py
```

**CLI:**

```bash
uv run python main.py
```

First run indexes the CSV into `chroma_langchain_db/`. Later runs reuse that folder.

If you see a connection error, Ollama is not running. Install it, open the app, then pull the models above.

### Example prompts

```text
Hi
Is the vegan pizza any good?
What are the worst complaints?
Summarize the rating distribution
Draft an owner reply about late delivery issues
```

Sidebar controls: response temperature, retriever `k`, clear chat, and reindex.

## Tests and quality checks

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
```

Tests mock Ollama/Chroma so they run offline. CI runs the same checks on every push.

## Reuse on your own reviews

1. Export a CSV with columns `Title`, `Date`, `Rating`, `Review`.
2. Set `CSV_PATH` (and optionally `COLLECTION_NAME` / `DB_LOCATION`) in `constants.py`.
3. Reindex (sidebar button or delete `chroma_langchain_db/`).
4. Adjust `SYSTEM_PROMPT` in `agent.py` if the domain wording should change.

## Stack

- LangChain `1.x` (`create_agent`, `@tool`)
- `langchain-ollama` (`ChatOllama`, `OllamaEmbeddings`)
- `langchain-chroma` (persistent local vector store)
- Streamlit (chat UI)
- pandas (CSV + rating filters)
- Ruff + mypy (dev)
