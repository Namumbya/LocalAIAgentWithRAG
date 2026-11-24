"""Tools the review agent can call."""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain.tools import tool

from logging_setup import get_logger
from vector import get_df, get_retriever

logger = get_logger(__name__)

# How many reviews filter_reviews_by_rating shows the model (keeps context small).
FILTER_SAMPLE_SIZE = 8


def _representative_sample(reviews: pd.DataFrame, size: int) -> pd.DataFrame:
    """Pick up to `size` rows spread evenly across all matches, not just the first rows.

    Rows are ordered by rating (lowest first), then newest first, and sampled at
    even intervals, so every rating level and the whole date range are represented
    regardless of how the CSV happens to be ordered.
    """
    ordered = reviews.assign(_date=pd.to_datetime(reviews["Date"], errors="coerce"))
    ordered = ordered.sort_values(["Rating", "_date"], ascending=[True, False], kind="stable")
    ordered = ordered.drop(columns="_date")
    if len(ordered) <= size:
        return ordered
    # Take the first row of each rating level, then fill evenly across the rest.
    firsts = ordered.groupby("Rating", sort=True).head(1)
    remaining = size - len(firsts)
    if remaining <= 0:
        return firsts.head(size)
    rest = ordered.drop(index=firsts.index)
    step = len(rest) / remaining
    picks = rest.iloc[[int(i * step) for i in range(remaining)]]
    return pd.concat([firsts, picks]).sort_values(
        ["Rating"], kind="stable"
    )


def _format_review_row(row: Any) -> str:
    return (
        f"Title: {row['Title']}\n"
        f"Date: {row['Date']}\n"
        f"Rating: {row['Rating']}/5\n"
        f"Review: {row['Review']}"
    )


@tool
def search_reviews(query: str) -> str:
    """Search customer reviews by meaning. Use for questions about food, service, vibe, or specific topics."""
    logger.info("tool search_reviews query=%r", query)
    docs = get_retriever().invoke(query)
    if not docs:
        return "No matching reviews found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        rating = doc.metadata.get("rating", "?")
        date = doc.metadata.get("date", "?")
        parts.append(
            f"[{i}] Rating: {rating}/5 | Date: {date}\n{doc.page_content}"
        )
    return "\n\n".join(parts)


@tool
def filter_reviews_by_rating(min_rating: int, max_rating: int = 5) -> str:
    """Filter reviews by star rating inclusive. Use for complaints (1-2), mixed (3), or praise (4-5)."""
    logger.info(
        "tool filter_reviews_by_rating min=%s max=%s", min_rating, max_rating
    )
    if min_rating > max_rating:
        return "min_rating must be less than or equal to max_rating."

    reviews = get_df()
    filtered = reviews[
        (reviews["Rating"] >= min_rating) & (reviews["Rating"] <= max_rating)
    ]
    if filtered.empty:
        return f"No reviews found with ratings between {min_rating} and {max_rating}."

    sample = _representative_sample(filtered, FILTER_SAMPLE_SIZE)
    parts = [_format_review_row(row) for _, row in sample.iterrows()]
    breakdown = ", ".join(
        f"{rating} stars: {count}"
        for rating, count in filtered["Rating"].value_counts().sort_index().items()
    )
    header = (
        f"Found {len(filtered)} reviews rated {min_rating}-{max_rating} ({breakdown}). "
        f"Showing {len(sample)} spread across all matches, lowest rating first:"
    )
    return header + "\n\n" + "\n\n".join(parts)


@tool
def get_rating_stats() -> str:
    """Return overall rating counts and average score from all reviews."""
    logger.info("tool get_rating_stats")
    reviews = get_df()
    counts = reviews["Rating"].value_counts().sort_index()
    average = reviews["Rating"].mean()
    lines = [f"{rating} stars: {count}" for rating, count in counts.items()]
    lines.append(f"Average rating: {average:.2f}")
    lines.append(f"Total reviews: {len(reviews)}")
    return "\n".join(lines)
