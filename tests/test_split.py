"""Tests for ``src.split`` (M7 — strict chronological train/val/test split).

Covers date-axis quantile cuts, chronological boundary assertions
(no leakage), row conservation, no-shuffle determinism, label preservation,
geo-unit calendar alignment, and every documented error path.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src import split
from src.exceptions import SplitError

D0 = pd.Timestamp("2024-01-01")


def _dates(n: int, start: pd.Timestamp = D0) -> list[pd.Timestamp]:
    """Return ``n`` weekly dates starting at ``start``."""
    return [start + timedelta(weeks=i) for i in range(n)]


def _labeled_frame(n_dates: int = 20, n_units: int = 3) -> pd.DataFrame:
    """Synthetic labeled frame: ``n_units`` geo units x ``n_dates`` weeks."""
    dates = _dates(n_dates)
    rows = []
    for unit in range(n_units):
        for d in dates:
            rows.append(
            {
                "geo_unit": f"U{unit}",
                "event_date": d,
                "country": "India",
                "some_feature": float(unit + d.day),
                "escalation": 1 if unit == 0 else 0,
            }
            )
    return pd.DataFrame(rows)


def _make_splits(n_dates: int = 20, n_units: int = 3) -> dict[str, pd.DataFrame]:
    return split.chronological_split(_labeled_frame(n_dates, n_units))


# ---------------------------------------------------------------------------
# Chronology / leakage
# ---------------------------------------------------------------------------


def test_boundary_assertion_green() -> None:
    parts = _make_splits()
    split.assert_no_leakage(parts)
    # strict ordering: max(train) < min(val) < min(test)
    assert parts["train"]["event_date"].max() < parts["val"]["event_date"].min()
    assert parts["val"]["event_date"].max() < parts["test"]["event_date"].min()


def test_assert_no_leakage_raises_on_overlap() -> None:
    parts = _make_splits()
    parts["train"] = pd.concat([parts["train"], parts["val"].iloc[[0]]])
    with pytest.raises(SplitError, match="Chronological boundary violated"):
        split.assert_no_leakage(parts)


def test_no_shuffle_preserves_order() -> None:
    parts = _make_splits()
    for part in parts.values():
        assert part["event_date"].is_monotonic_increasing


def test_boundary_row_assigned_once() -> None:
    parts = _make_splits()
    all_dates = pd.concat(
        [part[["geo_unit", "event_date"]] for part in parts.values()]
    )
    assert len(all_dates) == len(all_dates.drop_duplicates())
    assert not parts["train"]["event_date"].isin(parts["val"]["event_date"]).any()
    assert not parts["val"]["event_date"].isin(parts["test"]["event_date"]).any()


# ---------------------------------------------------------------------------
# Date-axis quantile semantics
# ---------------------------------------------------------------------------


def test_cutoffs_follow_date_axis_quantiles() -> None:
    dates = _dates(20)
    ranges = split.compute_cutoffs(dates, {"train": 0.7, "val": 0.15, "test": 0.15})
    # 20 dates: train = first 14, val = next 3, test = last 3
    assert ranges["train"] == ("2024-01-01", "2024-04-01")
    assert ranges["val"] == ("2024-04-08", "2024-04-22")
    assert ranges["test"] == ("2024-04-29", "2024-05-13")


def test_cutoffs_date_axis_not_row_counts() -> None:
    # 100 rows crammed into 10 dates: splits must follow the 10 dates, not rows
    frame = pd.DataFrame(
        {
            "row_id": range(100),  # makes rows unique (real frames have 38 cols)
            "event_date": [d for d in _dates(10) for _ in range(10)],
            "escalation": [1] * 100,
        }
    )
    parts = split.chronological_split(frame, ratios={"train": 0.5, "test": 0.5})
    # 10 dates, 50/50 -> train first 5 dates (50 rows), test last 5 (50 rows)
    assert len(parts["train"]) == 50
    assert len(parts["test"]) == 50
    assert parts["train"]["event_date"].max() < parts["test"]["event_date"].min()


def test_custom_ratios_two_way() -> None:
    parts = split.chronological_split(
        _labeled_frame(20), ratios={"train": 0.8, "test": 0.2}
    )
    assert set(parts) == {"train", "test"}
    split.assert_no_leakage(parts)
    assert len(parts["train"]) + len(parts["test"]) == 60


# ---------------------------------------------------------------------------
# Validation / errors
# ---------------------------------------------------------------------------


def test_empty_split_raises() -> None:
    with pytest.raises(SplitError, match="Too few unique dates"):
        split.chronological_split(_labeled_frame(3), ratios={"a": 0.7, "b": 0.2, "c": 0.1})


def test_bad_ratios_raise() -> None:
    frame = _labeled_frame(20)
    for bad in (
        {},
        {"train": 1.0},
        {"train": 0.0, "test": 1.0},
        {"train": 0.6, "test": 0.5},
    ):
        with pytest.raises(SplitError):
            split.chronological_split(frame, ratios=bad)


def test_missing_columns_raise() -> None:
    frame = _labeled_frame(20).drop(columns=["escalation"])
    with pytest.raises(SplitError, match="missing columns"):
        split.chronological_split(frame)


def test_non_datetime_date_raises() -> None:
    frame = _labeled_frame(20)
    frame["event_date"] = frame["event_date"].astype(str)
    with pytest.raises(SplitError, match="must be datetime64"):
        split.chronological_split(frame)


def test_nat_dates_raise() -> None:
    frame = _labeled_frame(20)
    frame.loc[0, "event_date"] = pd.NaT
    with pytest.raises(SplitError, match="NaT"):
        split.chronological_split(frame)


def test_cut_date_belongs_to_later_split() -> None:
    parts = _make_splits(20)
    # ranges are contiguous: the val start date is the first date NOT in train
    val_start = parts["val"]["event_date"].min()
    assert val_start not in parts["train"]["event_date"].values
    assert (parts["train"]["event_date"] < val_start).all()
    test_start = parts["test"]["event_date"].min()
    assert (parts["val"]["event_date"] < test_start).all()


def test_validate_splits_row_conservation() -> None:
    parts = _make_splits()
    parts["train"] = parts["train"].iloc[:-1]  # drop one row
    with pytest.raises(SplitError, match="Row conservation"):
        split.validate_splits(parts, 60)


def test_validate_splits_rejects_bad_labels() -> None:
    parts = _make_splits()
    parts["test"] = parts["test"].assign(escalation=2)
    with pytest.raises(SplitError, match="invalid label"):
        split.validate_splits(parts, 60)


def test_validate_splits_rejects_duplicates() -> None:
    parts = _make_splits()
    parts["train"] = pd.concat([parts["train"], parts["train"].iloc[[0]]])
    with pytest.raises(SplitError, match="duplicated"):
        split.validate_splits(parts, 61)


# ---------------------------------------------------------------------------
# Content integrity
# ---------------------------------------------------------------------------


def test_row_and_label_conservation() -> None:
    original = _labeled_frame(20)
    parts = split.chronological_split(original)
    assert sum(len(p) for p in parts.values()) == len(original)
    labels = pd.concat(
        [p[["geo_unit", "event_date", "escalation"]] for p in parts.values()]
    ).sort_values(["geo_unit", "event_date"]).reset_index(drop=True)
    expected = original[["geo_unit", "event_date", "escalation"]].sort_values(
        ["geo_unit", "event_date"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(labels, expected)


def test_all_geo_units_in_every_split() -> None:
    parts = _make_splits()
    for part in parts.values():
        assert set(part["geo_unit"]) == {"U0", "U1", "U2"}


def test_deterministic() -> None:
    frame = _labeled_frame(20)
    first = split.chronological_split(frame)
    second = split.chronological_split(frame)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_ratios_default_match_config() -> None:
    parts = _make_splits(20)
    # 20 unique dates, 70/15/15 -> 14 / 3 / 3 dates per split
    assert parts["train"]["event_date"].nunique() == 14
    assert parts["val"]["event_date"].nunique() == 3
    assert parts["test"]["event_date"].nunique() == 3


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def test_write_split_summary(tmp_path) -> None:
    parts = _make_splits()
    out = split.write_split_summary(parts, tmp_path / "split_summary.md")
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    for name in ("train", "val", "test"):
        assert name in text
    assert "chronological" in text
    assert "Cut dates" in text


# ---------------------------------------------------------------------------
# Real-shape sanity
# ---------------------------------------------------------------------------


def test_split_on_real_shaped_frame() -> None:
    """Split a frame shaped like the real labeled dataset (38 cols, weekly)."""
    rng = np.random.default_rng(7)
    dates = _dates(60)
    units = [f"G{i}" for i in range(10)]
    frame = pd.DataFrame(
        {
            "geo_unit": [u for u in units for _ in dates],
            "event_date": [d for _ in units for d in dates],
            "country": ["India"] * (10 * 60),
            "escalation": rng.integers(0, 2, size=10 * 60),
        }
    )
    parts = split.chronological_split(frame)
    split.validate_splits(parts, len(frame))
    assert all(p["event_date"].is_monotonic_increasing for p in parts.values())
