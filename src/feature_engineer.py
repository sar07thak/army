"""Feature engineering for conflict escalation forecasting.

Computes every feature group in PRD §11.3 on the cleaned event data, one row
per ``(geo_unit, event_date)``. All windows are **half-open**
``[as_of - W, as_of)`` and therefore use only strictly historical rows — a
spike on date T can never influence the feature row at T.

Missing-history policy: window features default to 0 when a unit has no rows
in the window; ``days_since_event`` uses ``config.RECENCY_SENTINEL``. The
final table contains no NaN values (enforced by :func:`validate_features`).

Granularity note: the current dataset is weekly-aggregated at admin-1 level,
so a 7-day window equals the previous week's bucket. The engine is cadence
agnostic and works identically on event-level data.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

# Columns the engine needs from the cleaned frame (source schema).
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "geo_unit",
    "event_date",
    "event_type",
    "events",
    "fatalities",
    "country",
    "admin1",
    "latitude",
    "longitude",
)

# Identifier columns kept on every feature row.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "geo_unit",
    "admin1",
    "country",
    "event_date",
)


# ---------------------------------------------------------------------------
# Window helpers (leakage-safe by construction)
# ---------------------------------------------------------------------------


def _windowed_sums(
    dates: np.ndarray, values: np.ndarray, as_of: np.ndarray, window_days: int
) -> np.ndarray:
    """Sum ``values`` for rows with ``dates`` in ``[as_of - W, as_of)``.

    ``dates`` must be sorted ascending. Uses a prefix-sum so each query is
    O(log n); the window is strictly historical (``date < as_of``).
    """
    prefix = np.concatenate([[0], np.cumsum(values)])
    left = np.searchsorted(
        dates, as_of - np.timedelta64(window_days, "D"), side="left"
    )
    right = np.searchsorted(dates, as_of, side="left")
    return prefix[right] - prefix[left]


def _windowed_entropy(
    dates: np.ndarray, matrix: np.ndarray, as_of: np.ndarray, window_days: int
) -> np.ndarray:
    """Shannon entropy (natural log) of the event-type distribution.

    ``matrix`` is ``(n_dates, n_types)`` with per-date event counts per type.
    Windows with zero events yield entropy 0.
    """
    prefix = np.vstack([np.zeros((1, matrix.shape[1])), np.cumsum(matrix, axis=0)])
    left = np.searchsorted(
        dates, as_of - np.timedelta64(window_days, "D"), side="left"
    )
    right = np.searchsorted(dates, as_of, side="left")
    window_sums = prefix[right] - prefix[left]
    total = window_sums.sum(axis=1)
    p = np.divide(
        window_sums, total[:, None], out=np.zeros_like(window_sums), where=total[:, None] > 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1)
    return np.where(total > 0, entropy, 0.0)


def _days_since_event(
    dates: np.ndarray, events: np.ndarray, as_of: np.ndarray
) -> np.ndarray:
    """Days since the last active (events > 0) date strictly before ``as_of``."""
    active = dates[events > 0]
    if active.size == 0:
        return np.full(len(as_of), config.RECENCY_SENTINEL, dtype="int64")
    counts = np.searchsorted(active, as_of, side="left")
    has_prev = counts > 0
    last = active[np.maximum(counts - 1, 0)]
    days = (as_of - last).astype("timedelta64[D]").astype("int64")
    return np.where(has_prev, days, config.RECENCY_SENTINEL).astype("int64")


def _windowed_distinct(
    rows: pd.DataFrame, as_of_dates: np.ndarray, column: str, window_days: int
) -> np.ndarray:
    """Distinct count of ``column`` values in ``[as_of - W, as_of)``.

    Operates on the unit's own rows (event-level rows carry the actor
    columns). Simple per-window scan; fine for the project's data volumes.
    """
    dates = rows["event_date"].to_numpy()
    values = rows[column].to_numpy()
    out = np.empty(len(as_of_dates), dtype="int64")
    for i, as_of in enumerate(as_of_dates):
        lo = np.searchsorted(
            dates, as_of - np.timedelta64(window_days, "D"), side="left"
        )
        hi = np.searchsorted(dates, as_of, side="left")
        out[i] = len(np.unique(values[lo:hi])) if hi > lo else 0
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_unit_date(clean: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long cleaned frame to one row per (geo_unit, event_date).

    Keeps country/admin1 for the identity columns and sums event/fatality
    counts across the per-type rows of each date.
    """
    unit_date = (
        clean.groupby(["geo_unit", "event_date"], as_index=False)
        .agg(
            country=("country", "first"),
            admin1=("admin1", "first"),
            events=("events", "sum"),
            fatalities=("fatalities", "sum"),
        )
        .sort_values(["geo_unit", "event_date"])
        .reset_index(drop=True)
    )
    return unit_date


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------


def _add_volume_columns(
    row: pd.DataFrame,
    sums: dict[str, np.ndarray],
    dates: np.ndarray,
    matrix: np.ndarray,
    as_of: np.ndarray,
) -> None:
    """Add event/fatality window counts, their log1p, and type entropy."""
    for w in config.ROLLING_WINDOWS:
        row[f"events_w{w}d"] = sums[f"e{w}"]
        row[f"events_log1p_w{w}d"] = np.log1p(sums[f"e{w}"])
        row[f"fatalities_w{w}d"] = sums[f"f{w}"]
        row[f"fatalities_log1p_w{w}d"] = np.log1p(sums[f"f{w}"])
        row[f"entropy_w{w}d"] = _windowed_entropy(dates, matrix, as_of, w)


def _add_velocity_columns(row: pd.DataFrame, sums: dict[str, np.ndarray]) -> None:
    """Add escalation velocity: current window minus the prior window."""
    for w in config.VELOCITY_WINDOWS:
        row[f"velocity_events_w{w}d"] = sums[f"e{w}"] - (sums[f"e{2 * w}"] - sums[f"e{w}"])
        row[f"velocity_fatalities_w{w}d"] = sums[f"f{w}"] - (
            sums[f"f{2 * w}"] - sums[f"f{w}"]
        )


def _add_volatility_columns(row: pd.DataFrame, sums: dict[str, np.ndarray]) -> None:
    """Add rolling fatality mean and population standard deviation (ddof=0)."""
    for w in config.VOLATILITY_WINDOWS:
        count = np.where(sums[f"n{w}"] > 0, sums[f"n{w}"], np.nan)
        mean = sums[f"f{w}"] / count
        variance = sums[f"q{w}"] / count - mean**2
        row[f"fat_mean_w{w}d"] = np.nan_to_num(mean)
        row[f"fat_std_w{w}d"] = np.nan_to_num(np.sqrt(np.maximum(variance, 0.0)))


def _build_window_features(
    unit_date: pd.DataFrame, clean: pd.DataFrame
) -> pd.DataFrame:
    """Compute window features per (geo_unit, event_date), unit by unit.

    Every unit is processed in chronological order; all windows are strictly
    historical (half-open ``[as_of - W, as_of)``).
    """
    windows = tuple(
        sorted(
            set(config.ROLLING_WINDOWS)
            | {2 * w for w in config.VELOCITY_WINDOWS}
            | {config.PERSISTENCE_WINDOW_DAYS}
        )
    )
    types_sorted = sorted(clean["event_type"].unique())
    typed = clean.groupby(["geo_unit", "event_date", "event_type"], as_index=False)[
        "events"
    ].sum()
    actor_columns = [c for c in ("actor1", "actor2") if c in clean.columns]
    if actor_columns:
        logger.info("Actor diversity features will use columns: %s", actor_columns)

    per_unit: list[pd.DataFrame] = []
    for unit in sorted(unit_date["geo_unit"].unique()):
        sub = unit_date[unit_date["geo_unit"] == unit].sort_values("event_date")
        dates = sub["event_date"].to_numpy()
        as_of = dates
        events = sub["events"].to_numpy(dtype="float64")
        fatalities = sub["fatalities"].to_numpy(dtype="float64")
        ones = np.ones_like(events)
        active = (events > 0).astype("float64")
        squared = fatalities**2

        sums: dict[str, np.ndarray] = {}
        for w in windows:
            sums[f"e{w}"] = _windowed_sums(dates, events, as_of, w)
            sums[f"f{w}"] = _windowed_sums(dates, fatalities, as_of, w)
            sums[f"n{w}"] = _windowed_sums(dates, ones, as_of, w)
            sums[f"q{w}"] = _windowed_sums(dates, squared, as_of, w)
            sums[f"a{w}"] = _windowed_sums(dates, active, as_of, w)

        # Per-date x per-type event matrix for entropy (zeros when no rows).
        typed_u = typed[typed["geo_unit"] == unit]
        matrix = np.zeros((len(sub), len(types_sorted)), dtype="float64")
        if len(typed_u):
            matrix = (
                typed_u.pivot(index="event_date", columns="event_type", values="events")
                .reindex(index=sub["event_date"], columns=types_sorted, fill_value=0)
                .to_numpy(dtype="float64")
            )

        row = sub.copy()
        _add_volume_columns(row, sums, dates, matrix, as_of)
        _add_velocity_columns(row, sums)
        _add_volatility_columns(row, sums)
        row["persistence_w7d"] = sums[f"a{config.PERSISTENCE_WINDOW_DAYS}"]
        row["days_since_event"] = _days_since_event(dates, events, as_of)
        for column in actor_columns:
            unit_rows = clean[clean["geo_unit"] == unit].sort_values("event_date")
            for w in (14, 30):
                row[f"{column}_div_w{w}d"] = _windowed_distinct(
                    unit_rows, as_of, column, w
                )
        per_unit.append(row)

    return pd.concat(per_unit, ignore_index=True)


def build_identity_features(features: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic integer codes for country, admin1, and geo_unit.

    Codes are stable because they are derived from sorted unique values.
    """
    country_map = {c: i for i, c in enumerate(sorted(clean["country"].unique()))}
    admin1_map = {a: i for i, a in enumerate(sorted(clean["admin1"].unique()))}
    unit_map = {u: i for i, u in enumerate(sorted(clean["geo_unit"].unique()))}
    features["country_code"] = features["country"].map(country_map).astype("int64")
    features["admin1_code"] = features["admin1"].map(admin1_map).astype("int64")
    features["geo_unit_code"] = features["geo_unit"].map(unit_map).astype("int64")
    return features


def build_calendar_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features: month and day-of-week from ``event_date``."""
    features["month"] = features["event_date"].dt.month.astype("int64")
    features["day_of_week"] = features["event_date"].dt.dayofweek.astype("int64")
    return features


# ---------------------------------------------------------------------------
# Spillover (PRD FR-13, config-gated)
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two coordinates."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _neighbor_map(centroids: pd.DataFrame, k: int) -> dict[str, list[str]]:
    """Map each geo unit to its K nearest same-country neighbours (by centroid).

    Ties are broken by unit index order, so the result is deterministic.
    """
    units = centroids["geo_unit"].tolist()
    lats = centroids["latitude"].to_numpy(dtype="float64")
    lons = centroids["longitude"].to_numpy(dtype="float64")
    country_of = dict(zip(units, centroids["country"]))
    result: dict[str, list[str]] = {}
    for i, unit in enumerate(units):
        candidates = [
            j
            for j in range(len(units))
            if j != i and country_of[units[j]] == country_of[unit]
        ]
        ranked = sorted(
            candidates,
            key=lambda j: (_haversine_km(lats[i], lons[i], lats[j], lons[j]), j),
        )
        result[unit] = [units[j] for j in ranked[:k]]
    return result


def build_spillover_features(
    features: pd.DataFrame, clean: pd.DataFrame
) -> pd.DataFrame:
    """Add ``spillover_w14d``: events in the window across neighbour units.

    Neighbours are the K nearest same-country geo units by centroid distance.
    Precondition: ``features`` must still carry the raw ``events`` column
    (``build_features`` drops it after spillover is computed).
    """
    centroids = clean.groupby("geo_unit", as_index=False).agg(
        country=("country", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    neighbors = _neighbor_map(centroids, config.SPILLOVER_K_NEIGHBORS)
    unit_dates: dict[str, np.ndarray] = {}
    unit_prefix: dict[str, np.ndarray] = {}
    for unit, sub in features.groupby("geo_unit"):
        dates = sub["event_date"].to_numpy()
        unit_dates[unit] = dates
        unit_prefix[unit] = np.concatenate([[0], np.cumsum(sub["events"].to_numpy(dtype="float64"))])

    features["spillover_w14d"] = 0.0
    for unit, nbrs in neighbors.items():
        idx = features["geo_unit"] == unit
        as_of = features.loc[idx, "event_date"].to_numpy()
        total = np.zeros(len(as_of), dtype="float64")
        for nbr in nbrs:
            left = np.searchsorted(
                unit_dates[nbr], as_of - np.timedelta64(config.SPILLOVER_WINDOW, "D"), side="left"
            )
            right = np.searchsorted(unit_dates[nbr], as_of, side="left")
            total += unit_prefix[nbr][right] - unit_prefix[nbr][left]
        features.loc[idx, "spillover_w14d"] = total
    logger.info("Spillover features computed for %d units (K=%d)", len(neighbors), config.SPILLOVER_K_NEIGHBORS)
    return features


# ---------------------------------------------------------------------------
# Orchestration, validation, summary
# ---------------------------------------------------------------------------


def apply_min_events_filter(
    clean: pd.DataFrame, min_events: int | None = None
) -> pd.DataFrame:
    """Drop geo units whose total events fall below the threshold.

    The default threshold is ``config.MIN_EVENTS_PER_UNIT`` (PRD §9.2). Note:
    this is a unit-selection rule based on full-history totals — it decides
    which units enter the table, it never changes any feature value.
    """
    min_events = config.MIN_EVENTS_PER_UNIT if min_events is None else min_events
    totals = clean.groupby("geo_unit")["events"].sum()
    keep = totals[totals >= min_events].index
    dropped = totals[totals < min_events]
    if len(dropped):
        logger.warning(
            "Dropped %d geo unit(s) with fewer than %d events: %s",
            len(dropped),
            min_events,
            dropped.index.tolist(),
        )
    return clean[clean["geo_unit"].isin(keep)].copy()


def validate_inputs(clean: pd.DataFrame) -> None:
    """Assert the cleaned frame has everything feature engineering needs.

    Raises:
        FeatureEngineeringError: if required columns are missing or empty.
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in clean.columns]
    if missing:
        raise FeatureEngineeringError(
            f"Missing required columns for feature engineering: {missing}. "
            f"Present: {sorted(clean.columns)}."
        )
    if clean.empty:
        raise FeatureEngineeringError("Cannot build features from an empty frame.")


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Build the complete feature table from a cleaned event frame.

    Pipeline: input validation -> min-events filter -> per-date aggregation
    -> window features -> identity codes -> calendar -> spillover -> output
    validation.

    Returns:
        One row per (geo_unit, event_date) with all PRD §11.3 features plus
        identity columns. Contains no NaN values.

    Raises:
        FeatureEngineeringError: on invalid input or output.
    """
    validate_inputs(clean)
    clean = apply_min_events_filter(clean)
    unit_date = _aggregate_unit_date(clean)
    features = _build_window_features(unit_date, clean)
    features = build_identity_features(features, clean)
    features = build_calendar_features(features)
    if config.SPILLOVER_ENABLED:
        features = build_spillover_features(features, clean)
    # The raw current-week counts are not features (they are unknown at the
    # prediction date) — drop them so they can never leak into a model.
    features = features.drop(columns=["events", "fatalities"], errors="ignore")
    features = features[
        list(IDENTITY_COLUMNS) + [c for c in features.columns if c not in IDENTITY_COLUMNS]
    ]
    validate_features(features)
    return features


def validate_features(features: pd.DataFrame) -> None:
    """Assert the feature table is structurally sound.

    Checks: required columns present, non-empty, no NaN, no duplicate
    (geo_unit, event_date) rows, and datetime64 event dates.

    Raises:
        FeatureEngineeringError: on any violation.
    """
    required = [
        "geo_unit",
        "admin1",
        "country",
        "event_date",
        "geo_unit_code",
        "admin1_code",
        "country_code",
        "month",
        "day_of_week",
        "events_w7d",
    ]
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise FeatureEngineeringError(f"Feature table is missing columns: {missing}.")
    if features.empty:
        raise FeatureEngineeringError("Feature table is empty.")
    if not pd.api.types.is_datetime64_any_dtype(features["event_date"]):
        raise FeatureEngineeringError("Feature table event_date must be datetime64.")
    nan_cols = [c for c in features.columns if features[c].isna().any()]
    if nan_cols:
        raise FeatureEngineeringError(
            f"Feature table contains NaN values in columns: {nan_cols}."
        )
    duplicates = features.duplicated(subset=["geo_unit", "event_date"]).sum()
    if duplicates:
        raise FeatureEngineeringError(
            f"Feature table has {duplicates} duplicate (geo_unit, event_date) rows."
        )
    logger.info(
        "Feature table validated: %d rows, %d columns, %d geo units",
        len(features),
        features.shape[1],
        features["geo_unit"].nunique(),
    )


def write_feature_summary(features: pd.DataFrame, path: Path) -> Path:
    """Write a markdown report of every feature: dtype, missing, and stats.

    Returns:
        The path the report was written to.
    """
    lines = [
        "# Feature Summary",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Rows: {len(features)}  ·  Columns: {features.shape[1]}  ·  "
        f"Geo units: {features['geo_unit'].nunique()}",
        "",
        "| feature | dtype | missing | mean | std | min | 25% | median | 75% | max |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for col in features.columns:
        series = features[col]
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        stats = series.describe() if numeric else None
        missing = int(series.isna().sum())
        if stats is None:
            cells = ["—"] * 7
        else:
            cells = [
                f"{stats['mean']:.4g}",
                f"{stats['std']:.4g}",
                f"{stats['min']:.4g}",
                f"{stats['25%']:.4g}",
                f"{stats['50%']:.4g}",
                f"{stats['75%']:.4g}",
                f"{stats['max']:.4g}",
            ]
        lines.append(f"| {col} | {series.dtype} | {missing} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote feature summary report: %s", path)
    return path
