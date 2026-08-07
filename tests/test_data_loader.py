"""Tests for ``src.data_loader``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader, data_validation
from src.exceptions import DataLoadError


def test_discover_returns_csv_files(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x\n2\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a csv", encoding="utf-8")
    files = data_loader.discover_raw_files(tmp_path)
    assert [f.name for f in files] == ["a.csv", "b.csv"]


def test_discover_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="No CSV files found"):
        data_loader.discover_raw_files(tmp_path)


def test_discover_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="not found"):
        data_loader.discover_raw_files(tmp_path / "nope")


def test_merge_two_files(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x\n1\n2\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x\n3\n", encoding="utf-8")
    merged = data_loader.merge_raw_files(
        [tmp_path / "a.csv", tmp_path / "b.csv"]
    )
    assert len(merged) == 3


def test_canonicalize_aggregated(aggregated_df: pd.DataFrame) -> None:
    out = data_loader.canonicalize(aggregated_df)
    assert "event_date" in out.columns  # from week
    assert out.loc[0, "event_date"] == "2024-01-06"
    assert out.loc[0, "latitude"] == pytest.approx(25.6)  # from centroid
    assert out.loc[0, "longitude"] == pytest.approx(85.1)
    assert out.loc[0, "events"] == 3  # preserved, not defaulted


def test_canonicalize_event_level(event_level_df: pd.DataFrame) -> None:
    out = data_loader.canonicalize(event_level_df)
    assert out.loc[0, "event_date"] == "01 Jan 2024"  # parsed later by validation
    assert (out["events"] == 1).all()  # defaulted to one event per row
    assert "latitude" in out.columns and "longitude" in out.columns


def test_canonicalize_missing_date_source_raises() -> None:
    df = pd.DataFrame({"country": ["India"], "events": [1]})
    with pytest.raises(DataLoadError, match="date source"):
        data_loader.canonicalize(df)


def test_canonicalize_missing_coords_raises() -> None:
    df = pd.DataFrame({"week": ["2024-01-06"], "country": ["India"]})
    with pytest.raises(DataLoadError, match="coordinate source"):
        data_loader.canonicalize(df)


def test_load_raw_data_end_to_end(aggregated_csv: Path) -> None:
    out = data_loader.load_raw_data(aggregated_csv.parent)
    assert len(out) == 4
    assert {"event_date", "events", "latitude", "longitude"} <= set(out.columns)


def test_load_raw_data_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError):
        data_loader.load_raw_data(tmp_path)


def test_save_dataframe_writes_formats(tmp_path: Path) -> None:
    written = data_loader.save_dataframe(
        pd.DataFrame({"x": [1, 2]}), "sample", tmp_path
    )
    assert (tmp_path / "sample.parquet").is_file()
    assert (tmp_path / "sample.csv").is_file()
    assert len(written) == 2


def test_save_dataframe_unsupported_format_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="Unsupported output format"):
        data_loader.save_dataframe(
            pd.DataFrame({"x": [1]}), "sample", tmp_path, formats=("xml",)
        )


def test_save_clean_outputs_writes_files(
    canonical_aggregated: pd.DataFrame, tmp_path: Path
) -> None:
    clean = data_validation.validate_dataset(canonical_aggregated)
    master = data_validation.build_district_master(clean)
    written = data_loader.save_clean_outputs(clean, master, tmp_path)
    assert (tmp_path / "cleaned_events.parquet").is_file()
    assert (tmp_path / "cleaned_events.csv").is_file()
    assert (tmp_path / "district_master.parquet").is_file()
    assert (tmp_path / "district_master.csv").is_file()
    assert len(written["cleaned_events"]) == 2
    assert len(written["district_master"]) == 2
