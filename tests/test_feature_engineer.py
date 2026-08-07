"""Tests for ``src.feature_engineer``.

Every feature group is tested with hand-computed expectations, and the
no-future-leakage guarantee is verified both with a targeted spike test and a
seeded randomized property test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import config
from src import feature_engineer
from src.exceptions import FeatureEngineeringError

D0 = pd.Timestamp("2024-01-06")
D1 = pd.Timestamp("2024-01-13")
D2 = pd.Timestamp("2024-01-20")
D3 = pd.Timestamp("2024-01-27")
D4 = pd.Timestamp("2024-02-03")


def row(
    geo_unit: str,
    date: pd.Timestamp,
    event_type: str,
    events: int,
    fatalities: int,
    country: str = "India",
    admin1: str = "A1",
    lat: float = 25.6,
    lon: float = 85.1,
    **extra: object,
) -> dict[str, object]:
    """Build one cleaned-schema row (adaptable via ``extra``)."""
    base = {
        "geo_unit": geo_unit,
        "admin1": admin1,
        "country": country,
        "event_date": date,
        "event_type": event_type,
        "events": events,
        "fatalities": fatalities,
        "latitude": lat,
        "longitude": lon,
    }
    base.update(extra)
    return base


@pytest.fixture
def single_unit_clean() -> pd.DataFrame:
    """One unit, four active weeks, two event types (hand-computable)."""
    return pd.DataFrame(
        [
            row("U1", D0, "Battles", 4, 2),
            row("U1", D0, "Protests", 2, 0),
            row("U1", D1, "Battles", 1, 1),
            row("U1", D2, "Protests", 3, 5),
            row("U1", D3, "Battles", 2, 0),
        ]
    )


@pytest.fixture
def single_unit_features(single_unit_clean: pd.DataFrame) -> pd.DataFrame:
    return feature_engineer.build_features(single_unit_clean)


def _at(features: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    return features.loc[features["event_date"] == date].iloc[0]


# ---------------------------------------------------------------------------
# Shape / identities
# ---------------------------------------------------------------------------


def test_feature_table_shape_and_identities(single_unit_features: pd.DataFrame) -> None:
    feats = single_unit_features
    assert len(feats) == 4  # one row per active week
    assert list(feats["geo_unit"].unique()) == ["U1"]
    assert {"geo_unit", "admin1", "country", "event_date"} <= set(feats.columns)
    assert {"geo_unit_code", "admin1_code", "country_code"} <= set(feats.columns)
    assert "events_w7d" in feats.columns
    # raw current-week counts must not survive into the feature table
    assert "events" not in feats.columns
    assert "fatalities" not in feats.columns
    # no NaNs anywhere
    assert not feats.isna().any().any()


def test_country_admin_codes_deterministic(single_unit_clean: pd.DataFrame) -> None:
    a = feature_engineer.build_features(single_unit_clean)
    b = feature_engineer.build_features(single_unit_clean)
    assert (a["country_code"] == b["country_code"]).all()
    assert (a["geo_unit_code"] == b["geo_unit_code"]).all()


# ---------------------------------------------------------------------------
# Volume features (event + fatality counts, log1p)
# ---------------------------------------------------------------------------


def test_event_count_windows(single_unit_features: pd.DataFrame) -> None:
    assert _at(single_unit_features, D1)["events_w7d"] == 6  # window = D0
    assert _at(single_unit_features, D2)["events_w7d"] == 1  # window = D1
    assert _at(single_unit_features, D3)["events_w7d"] == 3  # window = D2
    assert _at(single_unit_features, D2)["events_w14d"] == 7  # D0 + D1
    assert _at(single_unit_features, D3)["events_w30d"] == 10  # D0 + D1 + D2


def test_fatality_count_windows(single_unit_features: pd.DataFrame) -> None:
    assert _at(single_unit_features, D1)["fatalities_w7d"] == 2
    assert _at(single_unit_features, D2)["fatalities_w7d"] == 1
    assert _at(single_unit_features, D3)["fatalities_w7d"] == 5
    assert _at(single_unit_features, D2)["fatalities_w14d"] == 3


def test_log1p_transforms(single_unit_features: pd.DataFrame) -> None:
    assert _at(single_unit_features, D1)["events_log1p_w7d"] == pytest.approx(
        math.log1p(6)
    )
    assert _at(single_unit_features, D1)["fatalities_log1p_w7d"] == pytest.approx(
        math.log1p(2)
    )


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------


def test_velocity_features(single_unit_features: pd.DataFrame) -> None:
    # at D2: s7=1, prior7 = s14 - s7 = 7 - 1 = 6 -> velocity = -5
    assert _at(single_unit_features, D2)["velocity_events_w7d"] == -5
    # at D2: s7f=1, prior7f = 3 - 1 = 2 -> velocity = -1
    assert _at(single_unit_features, D2)["velocity_fatalities_w7d"] == -1
    # at D3: s14e = 4, prior14 = s28 - s14 = 10 - 4 = 6 -> velocity = -2
    assert _at(single_unit_features, D3)["velocity_events_w14d"] == -2


# ---------------------------------------------------------------------------
# Recency / persistence / volatility
# ---------------------------------------------------------------------------


def test_days_since_event(single_unit_features: pd.DataFrame) -> None:
    assert _at(single_unit_features, D1)["days_since_event"] == 7
    assert _at(single_unit_features, D2)["days_since_event"] == 7
    assert _at(single_unit_features, D3)["days_since_event"] == 7


def test_days_since_event_with_gap() -> None:
    clean = pd.DataFrame(
        [
            row("U1", D0, "Battles", 2, 1),
            row("U1", D2, "Battles", 3, 2),  # D1 week inactive (no row)
        ]
    )
    feats = feature_engineer.build_features(clean)
    assert _at(feats, D2)["days_since_event"] == 14
    assert _at(feats, D2)["persistence_w7d"] == 0


def test_persistence_counts_active_previous_week(single_unit_features: pd.DataFrame) -> None:
    assert _at(single_unit_features, D1)["persistence_w7d"] == 1
    assert _at(single_unit_features, D2)["persistence_w7d"] == 1


def test_fatality_mean_and_std(single_unit_features: pd.DataFrame) -> None:
    # at D1 the 14d window holds only D0 -> mean 2, std 0
    assert _at(single_unit_features, D1)["fat_mean_w14d"] == pytest.approx(2.0)
    assert _at(single_unit_features, D1)["fat_std_w14d"] == pytest.approx(0.0)
    # at D2 the 14d window holds D0,D1 -> mean 1.5, population std 0.5
    assert _at(single_unit_features, D2)["fat_mean_w14d"] == pytest.approx(1.5)
    assert _at(single_unit_features, D2)["fat_std_w14d"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def test_entropy_hand_computed(single_unit_features: pd.DataFrame) -> None:
    # at D1 the 7d window holds D0 only: Battles 4, Protests 2
    p_battles, p_protests = 4 / 6, 2 / 6
    expected = -(p_battles * math.log(p_battles) + p_protests * math.log(p_protests))
    assert _at(single_unit_features, D1)["entropy_w7d"] == pytest.approx(expected)
    # at D3 the window holds a single type -> entropy 0
    assert _at(single_unit_features, D3)["entropy_w7d"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def test_calendar_features(single_unit_features: pd.DataFrame) -> None:
    assert _at(single_unit_features, D1)["month"] == 1
    assert _at(single_unit_features, D1)["day_of_week"] == 5  # Saturday


# ---------------------------------------------------------------------------
# Leakage guarantees
# ---------------------------------------------------------------------------


def test_no_future_leakage_spike_not_in_own_row() -> None:
    clean = pd.DataFrame(
        [
            row("U1", D0, "Battles", 2, 0),
            row("U1", D1, "Battles", 100, 0),  # spike
            row("U1", D2, "Battles", 1, 0),
        ]
    )
    feats = feature_engineer.build_features(clean)
    # the spike must NOT appear in its own row (window is strictly historical)
    assert _at(feats, D1)["events_w7d"] == 2  # only D0
    assert _at(feats, D1)["events_w14d"] == 2
    # it may appear only one week later
    assert _at(feats, D2)["events_w7d"] == 100


def test_no_future_leakage_random_property() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range(D0, periods=26, freq="7D")
    frames = []
    for unit_idx in range(3):
        for date in dates:
            events = int(rng.integers(0, 10))
            if events == 0:  # inactive weeks simply have no row
                continue
            frames.append(
                row(f"U{unit_idx}", date, "Battles", events, int(rng.integers(0, 5)))
            )
    clean = pd.DataFrame(frames)
    feats = feature_engineer.build_features(clean)

    for _, frow in feats.iterrows():
        unit = frow["geo_unit"]
        as_of = frow["event_date"]
        past = clean[(clean["geo_unit"] == unit) & (clean["event_date"] < as_of)]
        manual_7 = past[past["event_date"] >= as_of - pd.Timedelta(days=7)][
            "events"
        ].sum()
        manual_14 = past[past["event_date"] >= as_of - pd.Timedelta(days=14)][
            "events"
        ].sum()
        manual_30 = past[past["event_date"] >= as_of - pd.Timedelta(days=30)][
            "events"
        ].sum()
        assert frow["events_w7d"] == manual_7
        assert frow["events_w14d"] == manual_14
        assert frow["events_w30d"] == manual_30


# ---------------------------------------------------------------------------
# Actor diversity (only when actor columns exist)
# ---------------------------------------------------------------------------


@pytest.fixture
def actor_clean() -> pd.DataFrame:
    return pd.DataFrame(
        [
            row("U1", D0, "Battles", 1, 0, actor1="A", actor2="X"),
            row("U1", D0, "Protests", 1, 0, actor1="B", actor2="Y"),
            row("U1", D1, "Battles", 1, 0, actor1="A", actor2="X"),
            row("U1", D2, "Protests", 1, 0, actor1="C", actor2="Z"),
            row("U1", D3, "Battles", 1, 0, actor1="D", actor2="W"),
        ]
    )


def test_actor_diversity_features(actor_clean: pd.DataFrame) -> None:
    feats = feature_engineer.build_features(actor_clean)
    # at D0 the 14d window is empty (no strictly-past rows) -> 0
    assert _at(feats, D0)["actor1_div_w14d"] == 0
    # at D2: 14d window holds D0,D1 -> actors {A,B}
    assert _at(feats, D2)["actor1_div_w14d"] == 2
    assert _at(feats, D2)["actor1_div_w30d"] == 2
    assert _at(feats, D2)["actor2_div_w14d"] == 2  # D0,D1 -> {X, Y}
    # at D3: 30d window holds D0..D2 -> actors {A,B,C}
    assert _at(feats, D3)["actor1_div_w30d"] == 3
    assert _at(feats, D3)["actor2_div_w30d"] == 3  # {X, Y, Z}


def test_actor_features_absent_without_actor_columns(
    single_unit_features: pd.DataFrame,
) -> None:
    assert "actor1_div_w14d" not in single_unit_features.columns
    assert "actor2_div_w14d" not in single_unit_features.columns


# ---------------------------------------------------------------------------
# Min-events filter
# ---------------------------------------------------------------------------


def test_min_events_filter_drops_sparse_units() -> None:
    clean = pd.DataFrame(
        [
            row("U1", D0, "Battles", 6, 0),
            row("U2", D0, "Battles", 3, 0),  # 3 total < MIN_EVENTS_PER_UNIT
        ]
    )
    feats = feature_engineer.build_features(clean)
    assert list(feats["geo_unit"].unique()) == ["U1"]


# ---------------------------------------------------------------------------
# Spillover
# ---------------------------------------------------------------------------


@pytest.fixture
def spillover_clean() -> pd.DataFrame:
    return pd.DataFrame(
        [
            row("U1", D0, "Battles", 3, 0, lat=25.60, lon=85.10),
            row("U1", D1, "Battles", 1, 0, lat=25.60, lon=85.10),
            row("U1", D2, "Battles", 2, 0, lat=25.60, lon=85.10),
            row("U2", D0, "Battles", 1, 0, lat=25.61, lon=85.11),  # near U1
            row("U2", D1, "Battles", 4, 0, lat=25.61, lon=85.11),
            row("U3", D0, "Battles", 5, 0, country="Pakistan", lat=30.0, lon=70.0),
        ]
    )


def test_spillover_neighbor_events(spillover_clean: pd.DataFrame) -> None:
    feats = feature_engineer.build_features(spillover_clean)
    u1 = feats[feats["geo_unit"] == "U1"].sort_values("event_date").reset_index(drop=True)
    # at D1: 14d window [2023-12-30, D1) over neighbor U2 -> its D0 count of 1
    assert u1.loc[u1["event_date"] == D1, "spillover_w14d"].iloc[0] == 1
    # at D2: 14d window [D0, D2) over neighbor U2 -> its D0 (1) + D1 (4) = 5
    assert u1.loc[u1["event_date"] == D2, "spillover_w14d"].iloc[0] == 5
    # U3 has no same-country neighbours -> spillover 0
    u3 = feats[feats["geo_unit"] == "U3"]
    assert (u3["spillover_w14d"] == 0).all()


def test_spillover_disabled_when_config_off(
    spillover_clean: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "SPILLOVER_ENABLED", False)
    feats = feature_engineer.build_features(spillover_clean)
    assert "spillover_w14d" not in feats.columns


# ---------------------------------------------------------------------------
# Validation / summary / errors
# ---------------------------------------------------------------------------


def test_validate_features_rejects_nan(single_unit_features: pd.DataFrame) -> None:
    bad = single_unit_features.copy()
    bad.loc[0, "events_w7d"] = np.nan
    with pytest.raises(FeatureEngineeringError, match="NaN"):
        feature_engineer.validate_features(bad)


def test_validate_features_rejects_duplicates(single_unit_features: pd.DataFrame) -> None:
    bad = pd.concat([single_unit_features, single_unit_features.iloc[[0]]])
    with pytest.raises(FeatureEngineeringError, match="duplicate"):
        feature_engineer.validate_features(bad)


def test_validate_features_rejects_missing_columns(
    single_unit_features: pd.DataFrame,
) -> None:
    bad = single_unit_features.drop(columns=["events_w7d"])
    with pytest.raises(FeatureEngineeringError, match="missing columns"):
        feature_engineer.validate_features(bad)


def test_validate_inputs_rejects_missing_column(single_unit_clean: pd.DataFrame) -> None:
    with pytest.raises(FeatureEngineeringError, match="Missing required columns"):
        feature_engineer.validate_inputs(single_unit_clean.drop(columns=["geo_unit"]))


def test_write_feature_summary(single_unit_features: pd.DataFrame, tmp_path) -> None:
    path = feature_engineer.write_feature_summary(
        single_unit_features, tmp_path / "feature_summary.md"
    )
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "| feature |" in text
    assert "events_w7d" in text
    assert "geo_unit" in text
