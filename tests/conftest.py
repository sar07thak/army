"""Shared synthetic fixtures for tests.

Fixtures mirror the two supported source shapes: the ACLED weekly aggregated
count file (currently in data/raw/) and the ACLED event-level export
(event_id_cnty, daily dates, admin2, actor1, per-event coordinates).
All fixtures are deterministic — no randomness.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader

BASE_AGGREGATED_ROWS = [
    {
        "week": "2024-01-06",
        "region": "South Asia",
        "country": "India",
        "admin1": "Bihar",
        "event_type": "Protests",
        "sub_event_type": "Peaceful protest",
        "events": 3,
        "fatalities": 0,
        "population_exposure": "113,469",
        "disorder_type": "Demonstrations",
        "centroid_latitude": 25.6,
        "centroid_longitude": 85.1,
    },
    {
        "week": "2024-01-06",
        "region": "South Asia",
        "country": "India",
        "admin1": "Bihar",
        "event_type": "Battles",
        "sub_event_type": "Armed clash",
        "events": 2,
        "fatalities": 4,
        "population_exposure": "113,469",
        "disorder_type": "Political violence",
        "centroid_latitude": 25.6,
        "centroid_longitude": 85.1,
    },
    {
        "week": "2024-01-13",
        "region": "South Asia",
        "country": "Pakistan",
        "admin1": "Khyber Pakhtunkhwa",
        "event_type": "Battles",
        "sub_event_type": "Armed clash",
        "events": 5,
        "fatalities": 9,
        "population_exposure": "35,525",
        "disorder_type": "Political violence",
        "centroid_latitude": 34.0,
        "centroid_longitude": 71.5,
    },
    {
        "week": "2024-01-13",
        "region": "Southeast Asia",
        "country": "Myanmar",
        "admin1": "Sagaing",
        "event_type": "Violence against civilians",
        "sub_event_type": "Attack",
        "events": 4,
        "fatalities": 6,
        "population_exposure": "5,324",
        "disorder_type": "Political violence",
        "centroid_latitude": 22.0,
        "centroid_longitude": 95.0,
    },
]

BASE_EVENT_LEVEL_ROWS = [
    {
        "event_id_cnty": "IND1",
        "event_date": "01 Jan 2024",
        "event_type": "Protests",
        "country": "India",
        "admin1": "Bihar",
        "admin2": "Patna",
        "actor1": "Protesters",
        "latitude": 25.6,
        "longitude": 85.1,
        "fatalities": 0,
    },
    {
        "event_id_cnty": "IND2",
        "event_date": "05 Jan 2024",
        "event_type": "Battles",
        "country": "India",
        "admin1": "Bihar",
        "admin2": "Gaya",
        "actor1": "Military Forces of India",
        "latitude": 24.8,
        "longitude": 85.0,
        "fatalities": 4,
    },
    {
        "event_id_cnty": "PAK1",
        "event_date": "12 Jan 2024",
        "event_type": "Battles",
        "country": "Pakistan",
        "admin1": "Khyber Pakhtunkhwa",
        "admin2": "Peshawar",
        "actor1": "Tehrik-i-Taliban Pakistan",
        "latitude": 34.0,
        "longitude": 71.5,
        "fatalities": 9,
    },
]


@pytest.fixture
def aggregated_df() -> pd.DataFrame:
    """A clean aggregated weekly-count frame (current raw file shape)."""
    return pd.DataFrame(BASE_AGGREGATED_ROWS)


@pytest.fixture
def event_level_df() -> pd.DataFrame:
    """A clean event-level frame (event_id_cnty, admin2, actor1, daily dates)."""
    return pd.DataFrame(BASE_EVENT_LEVEL_ROWS)


def write_csv(frame: pd.DataFrame, tmp_path: Path, name: str = "acled.csv") -> Path:
    """Persist ``frame`` as a CSV under ``tmp_path`` and return its path."""
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def aggregated_csv(tmp_path: Path, aggregated_df: pd.DataFrame) -> Path:
    """A CSV on disk with the aggregated weekly-count shape."""
    return write_csv(aggregated_df, tmp_path)


@pytest.fixture
def canonical_aggregated(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """The aggregated fixture after canonicalization (still unvalidated)."""
    return data_loader.canonicalize(aggregated_df)


@pytest.fixture
def event_level_csv(tmp_path: Path, event_level_df: pd.DataFrame) -> Path:
    """A CSV on disk with the event-level shape."""
    return write_csv(event_level_df, tmp_path)


def append_row(frame: pd.DataFrame, row: dict[str, object]) -> pd.DataFrame:
    """Return ``frame`` with one extra row (column-safe concat)."""
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
