"""Tests for ``src.data_validation``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
from conftest import append_row
from src import data_loader, data_validation
from src.exceptions import DataValidationError


# ---------------------------------------------------------------------------
# Required columns / dates
# ---------------------------------------------------------------------------


def test_validate_required_columns_raises() -> None:
    df = pd.DataFrame({"country": ["India"], "event_date": ["2024-01-06"]})
    with pytest.raises(DataValidationError, match="Missing required columns"):
        data_validation.validate_required_columns(df)


def test_parse_event_dates_iso() -> None:
    series = pd.Series(["2024-01-06", "2024-01-13"])
    parsed = data_validation.parse_event_dates(series)
    assert parsed.dtype.kind == "M"


def test_parse_event_dates_abbreviated() -> None:
    series = pd.Series(["01 Jan 2024", "05 Jan 2024"])
    parsed = data_validation.parse_event_dates(series)
    assert parsed.iloc[0] == pd.Timestamp("2024-01-01")


def test_parse_event_dates_bad_value_raises() -> None:
    series = pd.Series(["2024-01-06", "not-a-date"])
    with pytest.raises(DataValidationError, match="Failed to parse"):
        data_validation.parse_event_dates(series)


def test_parse_event_dates_fallback_format(monkeypatch: pytest.MonkeyPatch) -> None:
    real_to_datetime = pd.to_datetime

    def fake_to_datetime(series: pd.Series, errors: str = "raise", **kwargs: object) -> pd.Series:
        if kwargs.get("format") is None:
            raise ValueError("forced default-parser failure")
        return real_to_datetime(series, errors=errors, **kwargs)

    monkeypatch.setattr(pd, "to_datetime", fake_to_datetime)
    series = pd.Series(["01 Jan 2024"])
    parsed = data_validation.parse_event_dates(series)
    assert parsed.iloc[0] == pd.Timestamp("2024-01-01")


def test_filter_date_range_drops_outside_rows(canonical_aggregated: pd.DataFrame) -> None:
    df = append_row(
        canonical_aggregated,
        {
            "event_date": "2010-01-01",
            "country": "India",
            "admin1": "Bihar",
            "event_type": "Battles",
            "events": 1,
            "fatalities": 1,
            "latitude": 25.6,
            "longitude": 85.1,
        },
    )
    df["event_date"] = data_validation.parse_event_dates(df["event_date"])
    out = data_validation.filter_date_range(df)
    assert len(out) == len(df) - 1


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def test_validate_missing_drops_critical_rows(
    canonical_aggregated: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_DROPPED_FRACTION", 0.5)
    df = append_row(
        canonical_aggregated,
        {
            "event_date": None,
            "country": "India",
            "admin1": "Bihar",
            "event_type": "Battles",
            "events": 1,
            "fatalities": 1,
            "latitude": 25.6,
            "longitude": 85.1,
        },
    )
    out = data_validation.validate_missing(df)
    assert len(out) == len(df) - 1


def test_validate_missing_treats_empty_strings_as_missing(
    canonical_aggregated: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_DROPPED_FRACTION", 0.5)
    df = append_row(
        canonical_aggregated,
        {
            "event_date": "2024-01-20",
            "country": "   ",
            "admin1": "Bihar",
            "event_type": "Battles",
            "events": 1,
            "fatalities": 1,
            "latitude": 25.6,
            "longitude": 85.1,
        },
    )
    out = data_validation.validate_missing(df)
    assert len(out) == len(df) - 1


def test_validate_missing_raises_when_fraction_too_high(
    canonical_aggregated: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_DROPPED_FRACTION", 0.1)
    df = append_row(
        canonical_aggregated,
        {
            "event_date": None,
            "country": "India",
            "admin1": "Bihar",
            "event_type": "Battles",
            "events": 1,
            "fatalities": 1,
            "latitude": 25.6,
            "longitude": 85.1,
        },
    )
    with pytest.raises(DataValidationError, match="MAX_DROPPED_FRACTION"):
        data_validation.validate_missing(df)


# ---------------------------------------------------------------------------
# Countries / duplicates
# ---------------------------------------------------------------------------


def test_validate_countries_filters_out_of_scope(
    canonical_aggregated: pd.DataFrame,
) -> None:
    df = append_row(
        canonical_aggregated,
        {
            "event_date": "2024-01-20",
            "country": "Nepal",
            "admin1": "Bagmati",
            "event_type": "Battles",
            "events": 1,
            "fatalities": 1,
            "latitude": 27.7,
            "longitude": 85.3,
        },
    )
    out = data_validation.validate_countries(df)
    assert "Nepal" not in out["country"].values


def test_validate_countries_error_mode_raises(
    canonical_aggregated: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "COUNTRIES_MODE", "error")
    df = append_row(
        canonical_aggregated,
        {
            "event_date": "2024-01-20",
            "country": "Nepal",
            "admin1": "Bagmati",
            "event_type": "Battles",
            "events": 1,
            "fatalities": 1,
            "latitude": 27.7,
            "longitude": 85.3,
        },
    )
    with pytest.raises(DataValidationError, match="outside the scope"):
        data_validation.validate_countries(df)


def test_validate_duplicates_drops(canonical_aggregated: pd.DataFrame) -> None:
    df = data_validation.derive_event_ids(canonical_aggregated)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    out = data_validation.validate_duplicates(df)
    assert len(out) == len(df) - 1


def test_validate_duplicates_raise_mode(
    canonical_aggregated: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "DUPLICATES_MODE", "raise")
    df = data_validation.derive_event_ids(canonical_aggregated)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        data_validation.validate_duplicates(df)


def test_validate_duplicates_missing_event_id_raises() -> None:
    with pytest.raises(DataValidationError, match="event_id"):
        data_validation.validate_duplicates(pd.DataFrame({"country": ["India"]}))


# ---------------------------------------------------------------------------
# Coordinates / types
# ---------------------------------------------------------------------------


def test_validate_coordinates_raises_on_bounds(canonical_aggregated: pd.DataFrame) -> None:
    df = canonical_aggregated.copy()
    df.loc[0, "latitude"] = 95.0
    with pytest.raises(DataValidationError, match="out-of-bounds"):
        data_validation.validate_coordinates(df)


def test_validate_coordinates_rejects_non_numeric() -> None:
    df = pd.DataFrame({"latitude": ["abc"], "longitude": [85.1]})
    with pytest.raises(DataValidationError, match="non-numeric"):
        data_validation.validate_coordinates(df)


def test_validate_types_coerces_counts(canonical_aggregated: pd.DataFrame) -> None:
    out = data_validation.validate_types(canonical_aggregated)
    assert out["fatalities"].dtype.kind == "i"
    assert out["events"].dtype.kind == "i"


def test_validate_types_rejects_negative() -> None:
    df = pd.DataFrame({"fatalities": [-1], "events": [1]})
    with pytest.raises(DataValidationError, match="negative"):
        data_validation.validate_types(df)


def test_validate_types_rejects_non_numeric() -> None:
    df = pd.DataFrame({"fatalities": ["abc"], "events": [1]})
    with pytest.raises(DataValidationError, match="non-numeric"):
        data_validation.validate_types(df)


def test_actor1_filled_unknown(event_level_df: pd.DataFrame) -> None:
    df = event_level_df.copy()
    df.loc[1, "actor1"] = None
    raw = data_loader.canonicalize(df)
    out = data_validation.validate_types(raw)
    assert out.loc[1, "actor1"] == config.DEFAULT_ACTOR_VALUE


# ---------------------------------------------------------------------------
# Normalization / geo unit / event id
# ---------------------------------------------------------------------------


def test_normalize_admin_names_strips_and_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "ADMIN_NAME_NORMALIZATION", {"Sind": "Sindh"})
    df = pd.DataFrame({"admin1": ["  Sind  "], "admin2": ["N/A"]})
    out = data_validation.normalize_admin_names(df)
    assert out.loc[0, "admin1"] == "Sindh"
    assert out.loc[0, "admin2"] == "N/A"


def test_normalize_admin_names_keeps_acronyms() -> None:
    df = pd.DataFrame({"admin1": ["FATA"]})
    out = data_validation.normalize_admin_names(df)
    assert out.loc[0, "admin1"] == "FATA"


def test_derive_geo_unit_admin1_when_no_admin2(
    canonical_aggregated: pd.DataFrame,
) -> None:
    out = data_validation.derive_geo_unit(canonical_aggregated)
    assert (out["geo_unit"] == out["admin1"]).all()


def test_derive_geo_unit_prefers_admin2(event_level_df: pd.DataFrame) -> None:
    raw = data_loader.canonicalize(event_level_df)
    out = data_validation.derive_geo_unit(raw)
    assert out["geo_unit"].tolist() == ["Patna", "Gaya", "Peshawar"]


def test_derive_event_ids_from_event_id_cnty(event_level_df: pd.DataFrame) -> None:
    raw = data_loader.canonicalize(event_level_df)
    out = data_validation.derive_event_ids(raw)
    assert out["event_id"].tolist() == ["IND1", "IND2", "PAK1"]


def test_derive_event_ids_composite(canonical_aggregated: pd.DataFrame) -> None:
    out = data_validation.derive_event_ids(canonical_aggregated)
    assert out["event_id"].nunique() == len(out)
    assert all("|" in eid for eid in out["event_id"])


# ---------------------------------------------------------------------------
# District master
# ---------------------------------------------------------------------------


def test_build_district_master(canonical_aggregated: pd.DataFrame) -> None:
    clean = data_validation.validate_dataset(canonical_aggregated)
    master = data_validation.build_district_master(clean)
    assert len(master) == clean["geo_unit"].nunique()
    assert {
        "geo_unit",
        "admin1",
        "country",
        "latitude",
        "longitude",
        "total_events",
        "n_rows",
        "first_date",
        "last_date",
    } <= set(master.columns)
    bihar = master[master["geo_unit"] == "Bihar"].iloc[0]
    assert bihar["total_events"] == 5  # 3 protests + 2 battles
    assert bihar["n_rows"] == 2


def test_build_district_master_missing_columns_raises() -> None:
    with pytest.raises(DataValidationError, match="geo_unit"):
        data_validation.build_district_master(pd.DataFrame({"country": ["India"]}))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_validate_dataset_end_to_end(canonical_aggregated: pd.DataFrame) -> None:
    clean = data_validation.validate_dataset(canonical_aggregated)
    assert len(clean) == len(canonical_aggregated)
    assert clean["event_id"].is_unique
    assert clean["fatalities"].dtype.kind == "i"
    assert clean["event_date"].dtype.kind == "M"
    assert set(clean["country"]) <= set(config.COUNTRIES)


def test_validate_dataset_with_duplicate_and_out_of_scope(
    canonical_aggregated: pd.DataFrame,
) -> None:
    df = append_row(
        canonical_aggregated,
        {
            "event_date": "2024-01-06",
            "country": "India",
            "admin1": "Bihar",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "events": 3,
            "fatalities": 0,
            "latitude": 25.6,
            "longitude": 85.1,
        },
    )
    df = append_row(
        df,
        {
            "event_date": "2024-01-20",
            "country": "Nepal",
            "admin1": "Bagmati",
            "event_type": "Battles",
            "sub_event_type": "Armed clash",
            "events": 1,
            "fatalities": 1,
            "latitude": 27.7,
            "longitude": 85.3,
        },
    )
    clean = data_validation.validate_dataset(df)
    assert len(clean) == len(df) - 2  # duplicate removed + Nepal filtered
    assert "Nepal" not in clean["country"].values


def test_full_pipeline_from_aggregated_csv(aggregated_csv: Path) -> None:
    raw = data_loader.load_raw_data(aggregated_csv.parent)
    clean = data_validation.validate_dataset(raw)
    assert len(clean) == 4
    assert clean["event_id"].is_unique


def test_optional_columns_preserved(canonical_aggregated: pd.DataFrame) -> None:
    clean = data_validation.validate_dataset(canonical_aggregated)
    present = set(clean.columns) & set(config.OPTIONAL_COLUMNS)
    assert "sub_event_type" in present
    assert "events" in present


def test_full_pipeline_from_event_level_csv(event_level_csv: Path) -> None:
    raw = data_loader.load_raw_data(event_level_csv.parent)
    clean = data_validation.validate_dataset(raw)
    assert len(clean) == 3
    assert clean["geo_unit"].tolist() == ["Patna", "Gaya", "Peshawar"]
    assert clean["event_id"].tolist() == ["IND1", "IND2", "PAK1"]
    assert clean["events"].sum() == 3  # defaulted to one per row
