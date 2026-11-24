"""Central settings for the local agentic RAG demo.

Tune the values under "Change these" when swapping models or datasets.
The sample CSV uses restaurant reviews; any CSV with the same columns works.
"""

from __future__ import annotations

from pathlib import Path

##############################################################################
# Change these
##############################################################################

# Default for limited disk/RAM. On this stack, 0.5b tool-calls more reliably than 1.5b.
# Stronger when you have headroom: "qwen2.5:3b" or "qwen2.5" + "mxbai-embed-large"
CHAT_MODEL = "qwen2.5:0.5b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Sample domain dataset. Swap for another review CSV with columns:
# Title, Date, Rating, Review
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "realistic_restaurant_reviews.csv"
DB_LOCATION = BASE_DIR / "chroma_langchain_db"
COLLECTION_NAME = "restaurant_reviews"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_RETRIEVER_K = 5

##############################################################################
# Don't change casually
##############################################################################

LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOG_DIR / "app.log"
REQUIRED_CSV_COLUMNS = ("Title", "Date", "Rating", "Review")
