"""Tools the restaurant agent can call."""

from __future__ import annotations

from langchain.tools import tool

from vector import get_df, get_retriever


def _format_review_row(row) -> str:
    return (
        f"Title: {row['Title']}\n"
        f"Date: {row['Date']}\n"
        f"Rating: {row['Rating']}/5\n"
        f"Review: {row['Review']}"
    )


@tool
def search_reviews(query: str) -> str:
    """Search customer reviews by meaning. Use for questions about food, service, vibe, or specific topics."""
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
    if min_rating > max_rating:
        return "min_rating must be less than or equal to max_rating."

    reviews = get_df()
    filtered = reviews[
        (reviews["Rating"] >= min_rating) & (reviews["Rating"] <= max_rating)
    ]
    if filtered.empty:
        return f"No reviews found with ratings between {min_rating} and {max_rating}."

    # Keep the reply readable for the model.
    sample = filtered.head(8)
    parts = [_format_review_row(row) for _, row in sample.iterrows()]
    header = (
        f"Found {len(filtered)} reviews rated {min_rating}-{max_rating}. "
        f"Showing {len(sample)}:"
    )
    return header + "\n\n" + "\n\n".join(parts)


@tool
def get_rating_stats() -> str:
    """Return overall rating counts and average score from all reviews."""
    reviews = get_df()
    counts = reviews["Rating"].value_counts().sort_index()
    average = reviews["Rating"].mean()
    lines = [f"{rating} stars: {count}" for rating, count in counts.items()]
    lines.append(f"Average rating: {average:.2f}")
    lines.append(f"Total reviews: {len(reviews)}")
    return "\n".join(lines)
