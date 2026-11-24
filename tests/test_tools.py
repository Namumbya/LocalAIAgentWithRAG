from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from langchain_core.documents import Document

import tools


def test_format_review_row(sample_reviews: pd.DataFrame):
    text = tools._format_review_row(sample_reviews.iloc[0])

    assert "Title: Great crust" in text
    assert "Rating: 5/5" in text
    assert "Crispy and chewy." in text


def test_search_reviews_formats_results(monkeypatch):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Vegan win\nCashew cheese melts well.",
            metadata={"rating": 5, "date": "2024-01-04"},
        )
    ]
    monkeypatch.setattr(tools, "get_retriever", lambda: mock_retriever)

    result = tools.search_reviews.invoke({"query": "vegan"})

    assert "[1] Rating: 5/5 | Date: 2024-01-04" in result
    assert "Cashew cheese melts well." in result
    mock_retriever.invoke.assert_called_once_with("vegan")


def test_search_reviews_no_matches(monkeypatch):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    monkeypatch.setattr(tools, "get_retriever", lambda: mock_retriever)

    result = tools.search_reviews.invoke({"query": "nothing"})

    assert result == "No matching reviews found."


def test_filter_reviews_by_rating(monkeypatch, sample_reviews: pd.DataFrame):
    monkeypatch.setattr(tools, "get_df", lambda: sample_reviews)

    result = tools.filter_reviews_by_rating.invoke(
        {"min_rating": 1, "max_rating": 2}
    )

    assert "Found 1 reviews rated 1-2" in result
    assert "Late delivery" in result
    assert "Great crust" not in result


def test_filter_reviews_invalid_range(monkeypatch, sample_reviews: pd.DataFrame):
    monkeypatch.setattr(tools, "get_df", lambda: sample_reviews)

    result = tools.filter_reviews_by_rating.invoke(
        {"min_rating": 4, "max_rating": 2}
    )

    assert result == "min_rating must be less than or equal to max_rating."


def test_filter_reviews_empty(monkeypatch, sample_reviews: pd.DataFrame):
    monkeypatch.setattr(tools, "get_df", lambda: sample_reviews)

    # No 2-star reviews in the sample fixture.
    result = tools.filter_reviews_by_rating.invoke(
        {"min_rating": 2, "max_rating": 2}
    )

    assert result == "No reviews found with ratings between 2 and 2."


def test_get_rating_stats(monkeypatch, sample_reviews: pd.DataFrame):
    monkeypatch.setattr(tools, "get_df", lambda: sample_reviews)

    result = tools.get_rating_stats.invoke({})

    assert "1 stars: 1" in result
    assert "3 stars: 1" in result
    assert "5 stars: 2" in result
    assert "Average rating: 3.50" in result
    assert "Total reviews: 4" in result


def test_representative_sample_spans_ratings_and_file_order():
    # 40 low reviews: CSV order puts all 2-star rows first, then 1-star rows.
    rows = [
        {"Title": f"two-{i}", "Date": f"2024-01-{i % 28 + 1:02d}", "Rating": 2, "Review": "meh"}
        for i in range(34)
    ] + [
        {"Title": f"one-{i}", "Date": f"2024-02-{i + 1:02d}", "Rating": 1, "Review": "bad"}
        for i in range(6)
    ]
    reviews = pd.DataFrame(rows)

    sample = tools._representative_sample(reviews, 8)

    assert len(sample) == 8
    assert set(sample["Rating"]) == {1, 2}
    assert sample["Rating"].iloc[0] == 1  # lowest rating first
    # Not just the first 8 rows of the file.
    assert not set(sample["Title"]).issubset({f"two-{i}" for i in range(8)})


def test_filter_reviews_header_includes_breakdown(monkeypatch, sample_reviews: pd.DataFrame):
    monkeypatch.setattr(tools, "get_df", lambda: sample_reviews)

    result = tools.filter_reviews_by_rating.invoke({"min_rating": 1, "max_rating": 5})

    assert "Found 4 reviews rated 1-5 (1 stars: 1, 3 stars: 1, 5 stars: 2)" in result
    assert result.index("Late delivery") < result.index("Great crust")
