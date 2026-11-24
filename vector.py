"""Build and load a local Chroma vector store from review CSV rows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from constants import (
    COLLECTION_NAME,
    CSV_PATH,
    DB_LOCATION,
    DEFAULT_RETRIEVER_K,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    REQUIRED_CSV_COLUMNS,
)
from logging_setup import get_logger

logger = get_logger(__name__)

_df: pd.DataFrame | None = None
_retriever: Any = None
_retriever_k: int = DEFAULT_RETRIEVER_K


def load_reviews(csv_path: Path | str = CSV_PATH) -> pd.DataFrame:
    """Load a reviews CSV and validate the expected columns."""
    reviews = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_CSV_COLUMNS if col not in reviews.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            f"{missing}. Expected {list(REQUIRED_CSV_COLUMNS)}."
        )
    return reviews


def documents_from_df(reviews: pd.DataFrame) -> tuple[list[Document], list[str]]:
    """Convert review rows into LangChain documents and stable ids."""
    documents: list[Document] = []
    ids: list[str] = []

    for i, row in reviews.iterrows():
        documents.append(
            Document(
                page_content=f"{row['Title']}\n{row['Review']}",
                metadata={
                    "rating": int(row["Rating"]),
                    "date": str(row["Date"]),
                    "title": str(row["Title"]),
                },
                id=str(i),
            )
        )
        ids.append(str(i))

    return documents, ids


def build_vector_store(
    reviews: pd.DataFrame | None = None,
    *,
    persist_directory: Path | str = DB_LOCATION,
    embedding_model: str = EMBEDDING_MODEL,
    collection_name: str = COLLECTION_NAME,
    base_url: str = OLLAMA_BASE_URL,
) -> Chroma:
    """Create or load a Chroma store and index reviews when empty."""
    reviews = reviews if reviews is not None else load_reviews()
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    persist_directory = Path(persist_directory)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )

    # Folder may exist from a failed first run; only skip when docs are present.
    needs_index = len(vector_store.get()["ids"]) == 0
    if needs_index:
        documents, ids = documents_from_df(reviews)
        try:
            vector_store.add_documents(documents=documents, ids=ids)
        except Exception as exc:
            logger.exception("Failed to index reviews into %s", persist_directory)
            raise RuntimeError(
                "Could not index reviews. Is Ollama running, and have you pulled "
                f"`{embedding_model}`? Error: {exc}"
            ) from exc
        logger.info(
            "Indexed %s reviews into %s",
            len(documents),
            persist_directory.name,
        )

    return vector_store


def get_df() -> pd.DataFrame:
    """Lazy-loaded reviews dataframe used by tools."""
    global _df
    if _df is None:
        _df = load_reviews()
        logger.info("Loaded %s reviews from %s", len(_df), CSV_PATH.name)
    return _df


def set_retriever_k(k: int) -> None:
    """Update retriever top-k; rebuilds on next get_retriever() if changed."""
    global _retriever, _retriever_k
    if k < 1:
        raise ValueError("retriever k must be >= 1")
    if k != _retriever_k:
        logger.info("Retriever k changed %s -> %s", _retriever_k, k)
        _retriever = None
        _retriever_k = k


def get_retriever(k: int | None = None) -> Any:
    """Lazy-loaded retriever used by tools."""
    global _retriever, _retriever_k
    if k is not None and k != _retriever_k:
        set_retriever_k(k)
    if _retriever is None:
        vector_store = build_vector_store(get_df())
        _retriever = vector_store.as_retriever(
            search_kwargs={"k": _retriever_k}
        )
        logger.info("Built retriever with k=%s", _retriever_k)
    return _retriever


def reset_vector_store() -> None:
    """Delete the persisted Chroma DB and clear in-memory caches."""
    global _df, _retriever
    _df = None
    _retriever = None
    if DB_LOCATION.exists():
        shutil.rmtree(DB_LOCATION)
        logger.info("Deleted vector store at %s", DB_LOCATION)
    else:
        logger.info("No vector store directory to delete at %s", DB_LOCATION)
