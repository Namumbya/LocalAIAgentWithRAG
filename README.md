# Local AI Agent with RAG

A **local** restaurant-ops assistant built with **LangChain v1**, **Ollama**, and **Chroma**.

Unlike a basic retrieve-then-answer RAG script, this project is an **agent**: the model decides when to search reviews, filter by rating, or pull stats before answering.

Everything runs on your machine. No cloud LLM API keys required.

## What it solves

**Problem:** A pizza restaurant owner has a pile of customer reviews and wants quick answers like:

- What do people say about the gluten-free crust?
- What are the main 1–2 star complaints?
- Draft a short reply to customers upset about late delivery.

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
| `app.py` | Streamlit chat UI |
| `main.py` | CLI chat loop |
| `agent.py` | LangChain v1 agent + Ollama checks |
| `tools.py` | Agent tools (search / filter / stats) |
| `vector.py` | Load CSV, embed, persist Chroma |
| `realistic_restaurant_reviews.csv` | Sample review dataset |

## Prerequisites

1. [Ollama](https://ollama.com/) installed and running
2. [uv](https://docs.astral.sh/uv/) installed
3. Python 3.14 (see `.python-version`)
4. Pull the local models (lightweight defaults for limited disk):

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Rough sizes: `qwen2.5:3b` ~2 GB, `nomic-embed-text` ~274 MB.

If you have more space and want stronger tool calling, use `qwen2.5` and `mxbai-embed-large`, then update `CHAT_MODEL` in `agent.py` and `EMBEDDING_MODEL` in `vector.py`. After changing the embedding model, delete `chroma_langchain_db/` so it re-indexes.

## Setup

```bash
git clone https://github.com/<your-username>/LocalAIAgentWithRAG.git
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

## Tests

```bash
uv sync --group dev
uv run pytest
```

Tests mock Ollama/Chroma so they run offline.

Example prompts:

```text
Hi
Is the vegan pizza any good?
What are the worst complaints?
Summarize the rating distribution
Draft an owner reply about late delivery issues
```

## Why this is an agent (not plain RAG)

| Question | Plain RAG | This agent |
| --- | --- | --- |
| "Hi" | Always retrieves reviews | Answers directly |
| "Is vegan pizza good?" | Always retrieves top-k | Calls `search_reviews` |
| "Show 1-star complaints" | Similarity search only | Calls `filter_reviews_by_rating` |
| "What's the average rating?" | Hopes retrieval helps | Calls `get_rating_stats` |

## Stack

- LangChain `1.x` (`create_agent`, `@tool`)
- `langchain-ollama` (`ChatOllama`, `OllamaEmbeddings`)
- `langchain-chroma` (persistent local vector store)
- Streamlit (chat UI)
- pandas (CSV + rating filters)

## Notes

- Keep Ollama running in the background while you chat.
- Low on disk? Stick to the `:3b` + `nomic-embed-text` defaults. Remove unused models with `ollama rm <name>`.
- If the model ignores tools, try a larger tool-capable model when you have space.
- Delete `chroma_langchain_db/` if you change the CSV or embedding model and want a fresh index.
- This is a portfolio/demo project. Swap the CSV for your own docs to reuse the same pattern.
