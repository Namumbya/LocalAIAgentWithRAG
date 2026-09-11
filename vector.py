"""Build and load a local Chroma vector store from restaurant reviews."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "realistic_restaurant_reviews.csv"
DB_LOCATION = BASE_DIR / "chroma_langchain_db"
COLLECTION_NAME = "restaurant_reviews"
EMBEDDING_MODEL = "nomic-embed-text"

_df: pd.DataFrame | None = None
_retriever = None


def load_reviews(csv_path: Path | str = CSV_PATH) -> pd.DataFrame:
    """Load the restaurant reviews CSV."""
    return pd.read_csv(csv_path)


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
):
    """Create or load a Chroma store and index reviews when empty."""
    reviews = reviews if reviews is not None else load_reviews()
    embeddings = OllamaEmbeddings(model=embedding_model)
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
            raise RuntimeError(
                "Could not index reviews. Is Ollama running, and have you pulled "
                f"`{embedding_model}`? Error: {exc}"
            ) from exc
        print(f"Indexed {len(documents)} reviews into {persist_directory.name}")

    return vector_store


def get_df() -> pd.DataFrame:
    """Lazy-loaded reviews dataframe used by tools."""
    global _df
    if _df is None:
        _df = load_reviews()
    return _df


def get_retriever(k: int = 5):
    """Lazy-loaded retriever used by tools."""
    global _retriever
    if _retriever is None:
        vector_store = build_vector_store(get_df())
        _retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return _retriever
