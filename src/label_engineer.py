"""Label engineering for conflict escalation forecasting.

Implements the PRD §11.2 escalation definition as a leakage-safe binary
target attached to the engineered feature rows:

    escalation = 1 if, over the NEXT ``LABEL_HORIZON_DAYS`` days:
        (future_events >= ESCALATION_MIN_EVENTS
         AND future_events >= ESCALATION_MULTIPLIER * trailing-median events)
        OR future_fatalities >= ESCALATION_MIN_FATALITIES

where the trailing median is the unit's median per-date event count over the
previous ``TRAILING_MEDIAN_WINDOW_DAYS`` days. Units without any trailing
history use the absolute fallback ``future_events >= ABSOLUTE_MIN_EVENTS``.

Temporal separation is guaranteed by construction:

- Features (already computed) use only rows with ``date < as_of``.
- Labels use only rows with ``as_of < date <= as_of + horizon``.

Rows whose future window extends past the end of the dataset are dropped
(``INCOMPLETE_WINDOW = 'drop'``) so a label is never computed over
partially-observed data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.exceptions import LabelEngineeringError

logger = logging.getLogger(__name__)


def validate_inputs(features: pd.DataFrame, events: pd.DataFrame) -> None:
    """Assert both inputs carry the columns label engineering needs.

    Raises:
        LabelEngineeringError: listing the missing columns or empty inputs.
    """
    missing_features = [
        col for col in ("geo_unit", "event_date") if col not in features.columns
    ]
    if missing_features:
        raise LabelEngineeringError(
            f"Features missing columns: {missing_features}. "
            f"Present: {sorted(features.columns)}."
        )
    missing_events = [
        col
        for col in ("geo_unit", "event_date", "events", "fatalities")
        if col not in events.columns
    ]
    if missing_events:
        raise LabelEngineeringError(
            f"Events missing columns: {missing_events}. "
            f"Present: {sorted(events.columns)}."
        )
    if features.empty or events.empty:
        raise LabelEngineeringError("Cannot build labels from empty inputs.")


def _aggregate_unit_date(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse events to one row per (geo_unit, event_date) with sums.

    Local helper mirroring the feature-engine aggregation (kept private and
    small to avoid cross-module imports of private helpers).
    """
    return (
        events.groupby(["geo_unit", "event_date"], as_index=False)
        .agg(events=("events", "sum"), fatalities=("fatalities", "sum"))
        .sort_values(["geo_unit", "event_date"])
        .reset_index(drop=True)
    )


def _future_window_counts(
    dates: np.ndarray,
    events: np.ndarray,
    fatalities: np.ndarray,
    as_of: np.ndarray,
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum events/fatalities for rows with ``as_of < date <= as_of + horizon``.

    ``dates`` must be sorted ascending; prefix sums make each query O(log n).
    """
    left = np.searchsorted(dates, as_of, side="right")
    right = np.searchsorted(
        dates, as_of + np.timedelta64(horizon_days, "D"), side="right"
    )
    ev_prefix = np.concatenate([[0], np.cumsum(events)])
    fa_prefix = np.concatenate([[0], np.cumsum(fatalities)])
    return ev_prefix[right] - ev_prefix[left], fa_prefix[right] - fa_prefix[left]


def _trailing_medians(
    dates: np.ndarray, counts: np.ndarray, as_of: np.ndarray, window_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Median per-date event count in ``[as_of - W, as_of)``, and row counts.

    Returns ``(medians, n_rows)``; medians are NaN where the window is empty
    (that is the signal for the absolute-fallback rule).
    """
    left = np.searchsorted(
        dates, as_of - np.timedelta64(window_days, "D"), side="left"
    )
    right = np.searchsorted(dates, as_of, side="left")
    n_rows = right - left
    medians = np.full(len(as_of), np.nan)
    for i in range(len(as_of)):
        lo, hi = int(left[i]), int(right[i])
        if hi > lo:
            medians[i] = float(np.median(counts[lo:hi]))
    return medians, n_rows


def create_labels(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe escalation labels to the feature rows.

    Args:
        features: Engineered feature table (from ``features.parquet``).
        events: Cleaned event table (from ``cleaned_events.parquet``), used
            only to observe the future and trailing windows.

    Returns:
        ``features`` (sorted by geo_unit, then event_date) with the
        escalation label column appended; rows with incomplete future
        windows dropped.

    Raises:
        LabelEngineeringError: on invalid inputs or output.
    """
    validate_inputs(features, events)
    features = features.sort_values(["geo_unit", "event_date"]).reset_index(drop=True)
    unit_date = _aggregate_unit_date(events)
    global_end = events["event_date"].max()
    horizon = np.timedelta64(config.LABEL_HORIZON_DAYS, "D")

    escalation = np.zeros(len(features), dtype="int64")
    incomplete = np.zeros(len(features), dtype=bool)
    for unit in sorted(features["geo_unit"].unique()):
        mask = features["geo_unit"] == unit
        idx = features.index[mask].to_numpy()
        as_of = features.loc[idx, "event_date"].to_numpy()
        unit_events = unit_date[unit_date["geo_unit"] == unit].sort_values("event_date")
        dates = unit_events["event_date"].to_numpy()
        counts = unit_events["events"].to_numpy(dtype="float64")
        fatals = unit_events["fatalities"].to_numpy(dtype="float64")

        future_events, future_fatalities = _future_window_counts(
            dates, counts, fatals, as_of, config.LABEL_HORIZON_DAYS
        )
        median, history_rows = _trailing_medians(
            dates, counts, as_of, config.TRAILING_MEDIAN_WINDOW_DAYS
        )
        empty_history = history_rows == 0
        event_rule = np.where(
            empty_history,
            future_events >= config.ABSOLUTE_MIN_EVENTS,
            (future_events >= config.ESCALATION_MIN_EVENTS)
            & (future_events >= config.ESCALATION_MULTIPLIER * median),
        )
        fatality_rule = future_fatalities >= config.ESCALATION_MIN_FATALITIES
        escalation[idx] = (event_rule | fatality_rule).astype("int64")
        incomplete[idx] = (as_of + horizon) > global_end

    if config.INCOMPLETE_WINDOW not in {"drop", "raise"}:
        raise LabelEngineeringError(
            f"Unknown INCOMPLETE_WINDOW: {config.INCOMPLETE_WINDOW!r}."
        )
    dropped = int(incomplete.sum())
    if dropped and config.INCOMPLETE_WINDOW == "raise":
        raise LabelEngineeringError(
            f"{dropped} row(s) have incomplete future windows (data ends "
            f"{global_end.date()}); INCOMPLETE_WINDOW='raise'."
        )
    if dropped:
        logger.warning(
            "Dropping %d row(s) with incomplete future windows (horizon %dd, "
            "data ends %s)",
            dropped,
            config.LABEL_HORIZON_DAYS,
            global_end.date(),
        )
    keep = ~incomplete
    labeled = features.loc[keep].copy()
    labeled[config.LABEL_COLUMN] = escalation[keep]
    validate_labels(labeled, features, dropped)
    return labeled


def validate_labels(
    labeled: pd.DataFrame, original: pd.DataFrame, dropped_rows: int
) -> None:
    """Structurally validate the labeled dataset.

    Checks: label column present, labels in {0, 1}, no missing labels,
    expected dimensions, feature columns unchanged, per-unit chronological
    ordering.

    Raises:
        LabelEngineeringError: on any violation.
    """
    col = config.LABEL_COLUMN
    if col not in labeled.columns:
        raise LabelEngineeringError(f"Label column {col!r} missing.")
    if labeled.empty:
        raise LabelEngineeringError("Labeled dataset is empty.")
    if labeled[col].isna().any():
        raise LabelEngineeringError("Missing label values found.")
    if not set(labeled[col].unique()).issubset({0, 1}):
        raise LabelEngineeringError("Label values must be in {0, 1}.")
    if len(labeled) != len(original) - dropped_rows:
        raise LabelEngineeringError(
            f"Dimension mismatch: expected {len(original) - dropped_rows} rows, "
            f"got {len(labeled)}."
        )
    if set(original.columns) != set(labeled.columns) - {col}:
        raise LabelEngineeringError("Feature columns changed during labeling.")
    vanished = set(original["geo_unit"].unique()) - set(labeled["geo_unit"].unique())
    if vanished:
        logger.info(
            "Geo unit(s) fully dropped (all rows incomplete): %s", sorted(vanished)
        )
    for unit, sub in labeled.groupby("geo_unit", sort=False):
        diffs = np.diff(sub["event_date"].to_numpy())
        if len(diffs) and (diffs <= np.timedelta64(0, "D")).any():
            raise LabelEngineeringError(
                f"Chronological order violated for geo unit {unit!r}."
            )
    total = len(labeled)
    positives = int(labeled[col].sum())
    logger.info(
        "Labels validated: %d rows, %d positive (%.2f%%), %d negative",
        total,
        positives,
        100 * positives / total,
        total - positives,
    )


def _write_timeline_png(labeled: pd.DataFrame, path: Path) -> Path:
    """Render a monthly escalation timeline; skipped if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping timeline PNG (%s)", path)
        return path
    monthly = labeled.set_index("event_date").resample("ME")[config.LABEL_COLUMN].agg(
        ["size", "sum"]
    )
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(monthly.index, monthly["size"], alpha=0.4, label="rows")
    ax2 = ax1.twinx()
    ax2.plot(
        monthly.index, monthly["sum"], color="crimson", marker="o", label="escalations"
    )
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Feature rows")
    ax2.set_ylabel("Escalations")
    fig.suptitle(f"Escalation labels over time ({config.LABEL_HORIZON_DAYS}-day horizon)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote label timeline: %s", path)
    return path


def _markdown_table(header: list[str], rows: list[list[object]]) -> list[str]:
    """Render a markdown table (header, separator, rows) as text lines."""
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def write_label_summary(
    labeled: pd.DataFrame, path: Path, timeline_path: Path | None = None
) -> Path:
    """Write the label summary report (and optionally the timeline PNG).

    Report contents: class distribution, labels by country, labels by geo
    unit (top 15), and the monthly distribution.

    Returns:
        The path the report was written to.
    """
    col = config.LABEL_COLUMN
    total = len(labeled)
    positives = int(labeled[col].sum())
    negatives = total - positives
    lines = [
        "# Label Summary",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Horizon: {config.LABEL_HORIZON_DAYS} days (future-only labels)",
        f"- Rows: {total}  ·  Geo units: {labeled['geo_unit'].nunique()}",
        "",
        "## Class distribution",
        "",
    ]
    lines += _markdown_table(
        ["class", "count", "share"],
        [
            ["positive (1)", positives, f"{100 * positives / total:.2f}%"],
            ["negative (0)", negatives, f"{100 * negatives / total:.2f}%"],
        ],
    )
    lines += ["", "## Labels by country", ""]
    country_rows = [
        [
            country,
            len(sub),
            int(sub[col].sum()),
            f"{100 * int(sub[col].sum()) / len(sub):.2f}%",
        ]
        for country, sub in labeled.groupby("country", sort=True)
    ]
    lines += _markdown_table(["country", "rows", "positives", "positive %"], country_rows)
    lines += ["", "## Labels by geo unit (top 15 by positives)", ""]
    top = (
        labeled.groupby(["geo_unit", "country"], as_index=False)
        .agg(rows=("event_date", "size"), positives=(col, "sum"))
        .sort_values("positives", ascending=False)
        .head(15)
    )
    lines += _markdown_table(
        ["geo_unit", "country", "rows", "positives"],
        [[t["geo_unit"], t["country"], t["rows"], t["positives"]] for _, t in top.iterrows()],
    )
    monthly = labeled.groupby(
        labeled["event_date"].dt.to_period("M"), as_index=False
    ).agg(rows=("event_date", "size"), positives=(col, "sum"))
    monthly["positive_rate"] = 100 * monthly["positives"] / monthly["rows"]
    lines += ["", "## Monthly distribution", ""]
    lines += _markdown_table(
        ["month", "rows", "positives", "positive %"],
        [
            [m["event_date"], m["rows"], m["positives"], f"{m['positive_rate']:.2f}%"]
            for _, m in monthly.iterrows()
        ],
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote label summary: %s", path)
    if timeline_path is not None:
        _write_timeline_png(labeled, timeline_path)
    return path


def build_labeled_dataset(
    features_path: Path | None = None, events_path: Path | None = None
) -> pd.DataFrame:
    """Load the engineered features and cleaned events, then attach labels.

    Reads ``features.parquet`` and ``cleaned_events.parquet`` (defaulting to
    the configured ``data/processed`` paths) and returns the labeled frame.

    Raises:
        LabelEngineeringError: if a required dataset file is missing.
    """
    features_path = Path(
        features_path
        or (config.DATA_PROCESSED_DIR / f"{config.FEATURES_FILE}.parquet")
    )
    events_path = Path(
        events_path or (config.DATA_PROCESSED_DIR / f"{config.CLEANED_EVENTS_FILE}.parquet")
    )
    for label, path in (("features", features_path), ("cleaned events", events_path)):
        if not path.is_file():
            raise LabelEngineeringError(f"{label} dataset not found: {path}")
    features = pd.read_parquet(features_path)
    events = pd.read_parquet(events_path)
    logger.info(
        "Loaded %d feature rows and %d event rows for labeling",
        len(features),
        len(events),
    )
    return create_labels(features, events)
