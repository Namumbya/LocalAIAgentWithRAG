from __future__ import annotations

from pathlib import Path

import pandas as pd

from vector import documents_from_df, load_reviews


def test_load_reviews_reads_csv(csv_path: Path):
    reviews = load_reviews(csv_path)

    assert len(reviews) == 4
    assert list(reviews.columns) == ["Title", "Date", "Rating", "Review"]


def test_load_reviews_project_csv_exists():
    project_csv = Path(__file__).resolve().parents[1] / "realistic_restaurant_reviews.csv"
    reviews = load_reviews(project_csv)

    assert len(reviews) > 0
    assert {"Title", "Date", "Rating", "Review"}.issubset(reviews.columns)


def test_documents_from_df_builds_metadata(sample_reviews: pd.DataFrame):
    documents, ids = documents_from_df(sample_reviews)

    assert len(documents) == len(ids) == 4
    assert documents[0].page_content == "Great crust\nCrispy and chewy."
    assert documents[0].metadata == {
        "rating": 5,
        "date": "2024-01-01",
        "title": "Great crust",
    }
    assert ids[1] == "1"
