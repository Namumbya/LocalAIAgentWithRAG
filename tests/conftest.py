from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Title": "Great crust",
                "Date": "2024-01-01",
                "Rating": 5,
                "Review": "Crispy and chewy.",
            },
            {
                "Title": "Late delivery",
                "Date": "2024-01-02",
                "Rating": 1,
                "Review": "Waited over an hour.",
            },
            {
                "Title": "Okay pizza",
                "Date": "2024-01-03",
                "Rating": 3,
                "Review": "Nothing special.",
            },
            {
                "Title": "Vegan win",
                "Date": "2024-01-04",
                "Rating": 5,
                "Review": "Cashew cheese melts well.",
            },
        ]
    )


@pytest.fixture
def csv_path(tmp_path: Path, sample_reviews: pd.DataFrame) -> Path:
    path = tmp_path / "reviews.csv"
    sample_reviews.to_csv(path, index=False)
    return path
