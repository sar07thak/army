"""Tests for ``src.label_engineer``.

Covers the PRD §11.2 rule, threshold boundaries, empty/incomplete future
windows, chronological correctness, geo-unit isolation, and leakage
detection (events at ``as_of`` must never enter the label).
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src import label_engineer
from src.exceptions import LabelEngineeringError

D0 = pd.Timestamp("2024-01-06")
D1 = pd.Timestamp("2024-01-13")
D2 = pd.Timestamp("2024-01-20")
D3 = pd.Timestamp("2024-01-27")
D4 = pd.Timestamp("2024-02-03")


def make_features(unit_dates: list[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    """Minimal feature frame: identity + one dummy feature per (unit, date)."""
    return pd.DataFrame(
        [
            {
                "geo_unit": unit,
                "admin1": "A1",
                "country": "India",
                "event_date": date,
                "events_w7d": 1.0,
            }
            for unit, date in unit_dates
        ]
    )


def make_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Event frame with the canonical columns label engineering needs."""
    return pd.DataFrame(rows)


def ev(unit: str, date: pd.Timestamp, events: int, fatalities: int) -> dict[str, object]:
    return {
        "geo_unit": unit,
        "event_date": date,
        "events": events,
        "fatalities": fatalities,
    }


# ---------------------------------------------------------------------------
# Rule correctness (all thresholds from config)
# ---------------------------------------------------------------------------


@pytest.fixture
def scenario() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six units covering every branch of the PRD §11.2 rule.

    U1: history median 2, future events 3  -> boundary escalation (3 >= 3).
    U2: empty history, future events 5     -> absolute fallback escalation.
    U3: future fatalities 7                -> fatality escalation.
    U4: future events 1, fatalities 1      -> negative.
    U5: history median 10, future events 4 -> suppressed by multiplier.
    U6: tail row, extends the global end so all windows are complete.
    U7: empty history, future events 3     -> below absolute fallback.
    Uq: quiet after as_of                  -> empty future window -> 0.
    """
    events = make_events(
        [
            ev("U1", D0, 2, 0),
            ev("U1", D1, 2, 0),
            ev("U1", D2, 1, 0),
            ev("U1", D3, 3, 0),
            ev("U2", D1, 5, 0),
            ev("U3", D1, 1, 7),
            ev("U4", D1, 1, 1),
            ev("U5", D0, 10, 0),
            ev("U5", D2, 4, 0),
            ev("U6", D4, 1, 0),
            ev("U7", D1, 3, 0),
            ev("Uq", D0, 1, 0),
        ]
    )
    features = make_features(
        [
            ("U1", D2),
            ("U2", D0),
            ("U3", D0),
            ("U4", D0),
            ("U5", D1),
            ("U7", D0),
            ("Uq", D1),
        ]
    )
    return features, events


def test_label_rule_all_branches(scenario: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    features, events = scenario
    labeled = label_engineer.create_labels(features, events)
    got = dict(zip(labeled["geo_unit"], labeled[config.LABEL_COLUMN]))
    assert got == {"U1": 1, "U2": 1, "U3": 1, "U4": 0, "U5": 0, "U7": 0, "Uq": 0}


def test_label_fatality_boundaries() -> None:
    events = make_events(
        [
            ev("UF4", D1, 1, 4),
            ev("UF5", D1, 1, 5),
            ev("UEXT", D4, 1, 0),  # keeps windows complete
        ]
    )
    features = make_features([("UF4", D0), ("UF5", D0)])
    labeled = label_engineer.create_labels(features, events)
    got = dict(zip(labeled["geo_unit"], labeled[config.LABEL_COLUMN]))
    assert got == {"UF4": 0, "UF5": 1}


def test_label_event_at_as_of_never_counts(scenario: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    # A spike exactly at as_of must not leak into the future window.
    events = make_events(
        [
            ev("US", D1, 100, 50),  # spike at the prediction date
            ev("UEXT", D4, 1, 0),
        ]
    )
    features = make_features([("US", D1)])
    labeled = label_engineer.create_labels(features, events)
    assert labeled.loc[0, config.LABEL_COLUMN] == 0


def test_geo_unit_isolation() -> None:
    events = make_events(
        [
            ev("UHOT", D1, 6, 0),  # escalates via absolute rule
            ev("UCOLD", D1, 1, 0),
            ev("UEXT", D4, 1, 0),
        ]
    )
    features = make_features([("UHOT", D0), ("UCOLD", D0)])
    labeled = label_engineer.create_labels(features, events)
    got = dict(zip(labeled["geo_unit"], labeled[config.LABEL_COLUMN]))
    assert got == {"UHOT": 1, "UCOLD": 0}


# ---------------------------------------------------------------------------
# Incomplete future windows
# ---------------------------------------------------------------------------


def test_incomplete_window_dropped() -> None:
    events = make_events(
        [
            ev("UOK", D1, 5, 0),  # future window for as_of D0 is complete
            ev("UTAIL", D3, 5, 0),  # global end: as_of D3 has no future coverage
        ]
    )
    features = make_features([("UOK", D0), ("UTAIL", D3)])
    labeled = label_engineer.create_labels(features, events)
    assert list(labeled["geo_unit"]) == ["UOK"]
    assert labeled.loc[0, config.LABEL_COLUMN] == 1


def test_unit_fully_dropped_when_all_rows_incomplete() -> None:
    events = make_events([ev("UVAN", D3, 5, 0), ev("UOK", D1, 5, 0)])
    features = make_features([("UVAN", D3), ("UOK", D0)])
    labeled = label_engineer.create_labels(features, events)
    assert "UVAN" not in labeled["geo_unit"].values
    assert list(labeled["geo_unit"]) == ["UOK"]
    assert labeled.loc[0, config.LABEL_COLUMN] == 1


def test_incomplete_window_raise_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "INCOMPLETE_WINDOW", "raise")
    events = make_events([ev("UTAIL", D3, 5, 0)])
    features = make_features([("UTAIL", D3)])
    with pytest.raises(LabelEngineeringError, match="incomplete future windows"):
        label_engineer.create_labels(features, events)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_features_unchanged_and_chronological(
    scenario: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, events = scenario
    labeled = label_engineer.create_labels(features, events)
    assert list(labeled.columns) == list(features.columns) + [config.LABEL_COLUMN]
    for unit, sub in labeled.groupby("geo_unit"):
        assert sub["event_date"].is_monotonic_increasing


def test_validate_labels_rejects_bad_values() -> None:
    features = make_features([("U1", D0)])
    bad = features.copy()
    bad[config.LABEL_COLUMN] = [2]
    with pytest.raises(LabelEngineeringError, match=r"\{0, 1\}"):
        label_engineer.validate_labels(bad, features, 0)


def test_validate_labels_rejects_missing_label() -> None:
    features = make_features([("U1", D0)])
    with pytest.raises(LabelEngineeringError, match="Label column"):
        label_engineer.validate_labels(features, features, 0)


def test_validate_labels_rejects_nan_label() -> None:
    features = make_features([("U1", D0)])
    bad = features.copy()
    bad[config.LABEL_COLUMN] = [float("nan")]
    with pytest.raises(LabelEngineeringError, match="Missing label"):
        label_engineer.validate_labels(bad, features, 0)


def test_validate_labels_rejects_chronology_violation() -> None:
    features = make_features([("U1", D1), ("U1", D0)])  # out of order
    labeled = features.copy()
    labeled[config.LABEL_COLUMN] = [0, 0]
    with pytest.raises(LabelEngineeringError, match="Chronological"):
        label_engineer.validate_labels(labeled, features, 0)


def test_validate_inputs_rejects_missing_columns() -> None:
    features = make_features([("U1", D0)])
    events = make_events([ev("U1", D1, 1, 0)]).drop(columns=["fatalities"])
    with pytest.raises(LabelEngineeringError, match="Events missing columns"):
        label_engineer.create_labels(features, events)


# ---------------------------------------------------------------------------
# Summary + disk pipeline
# ---------------------------------------------------------------------------


def test_write_label_summary(
    scenario: tuple[pd.DataFrame, pd.DataFrame], tmp_path
) -> None:
    features, events = scenario
    labeled = label_engineer.create_labels(features, events)
    path = label_engineer.write_label_summary(
        labeled,
        tmp_path / "label_summary.md",
        timeline_path=tmp_path / "label_timeline.png",
    )
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "positive (1)" in text
    assert "negative (0)" in text
    assert "Monthly distribution" in text
    assert (tmp_path / "label_timeline.png").is_file()


def test_build_labeled_dataset_from_disk(
    scenario: tuple[pd.DataFrame, pd.DataFrame], tmp_path
) -> None:
    features, events = scenario
    features_path = tmp_path / "features.parquet"
    events_path = tmp_path / "events.parquet"
    features.to_parquet(features_path, index=False)
    events.to_parquet(events_path, index=False)
    labeled = label_engineer.build_labeled_dataset(features_path, events_path)
    assert config.LABEL_COLUMN in labeled.columns
    assert len(labeled) == len(features)


def test_build_labeled_dataset_missing_file_raises(tmp_path) -> None:
    with pytest.raises(LabelEngineeringError, match="not found"):
        label_engineer.build_labeled_dataset(
            tmp_path / "nope.parquet", tmp_path / "also.parquet"
        )
