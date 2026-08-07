"""Strict chronological train/validation/test split (PRD §11.5, plan M7).

The split is the **only** allowed form of data partitioning in this project:
no shuffle, no random cross-validation, no stratification across time, and
no oversampling across a boundary. Cut points are quantiles of the **date
axis** implied by ``SPLIT_RATIOS`` — not row counts — so every geo unit sees
the same calendar alignment in all splits (plan M7).

Guarantees enforced here (and re-checked at runtime):

- ``max(train dates) < min(val dates) < min(test dates)`` — no overlap.
- Every row belongs to exactly one split; the union equals the input.
- Order within a split is strictly chronological.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.exceptions import SplitError

logger = logging.getLogger(__name__)

_DEFAULT_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def _validate_input(df: pd.DataFrame, date_col: str, label_col: str) -> None:
    """Assert the frame carries the columns a chronological split needs.

    Raises:
        SplitError: naming missing columns, empty inputs, or a non-datetime
            date column.
    """
    missing = [col for col in (date_col, label_col) if col not in df.columns]
    if missing:
        raise SplitError(
            f"Split input missing columns: {missing}. Present: {sorted(df.columns)}."
        )
    if df.empty:
        raise SplitError("Cannot split an empty dataframe.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise SplitError(
            f"Column {date_col!r} must be datetime64, got {df[date_col].dtype}."
        )
    if df[date_col].isna().any():
        raise SplitError(f"Column {date_col!r} contains NaT dates.")
    if df[label_col].isna().any():
        raise SplitError(f"Column {label_col!r} contains missing values.")


def _validate_ratios(ratios: dict[str, float]) -> tuple[str, ...]:
    """Validate the split-ratio map and return its ordered split names.

    Raises:
        SplitError: for empty maps, non-positive or non-finite fractions, or
            fractions that do not sum to one.
    """
    if len(ratios) < 2:
        raise SplitError(f"Need at least 2 splits, got {len(ratios)}.")
    for name, frac in ratios.items():
        if not isinstance(frac, (int, float)) or not np.isfinite(frac) or frac <= 0.0:
            raise SplitError(f"Split ratio for {name!r} must be a positive number.")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise SplitError(
            f"Split ratios must sum to 1.0, got {sum(ratios.values()):.4f}."
        )
    return tuple(ratios)


def compute_cutoffs(
    dates: pd.Series | np.ndarray, ratios: dict[str, float]
) -> dict[str, str]:
    """Compute exact inclusive date ranges per split over the date axis.

    The unique dates are sorted and cut at the quantiles implied by ``ratios``
    (applied to the *number of unique dates*, not row counts). Each split is
    returned as ``(start_date, end_date)`` ISO strings; boundaries are exact
    dates present in the data, so no row sits on a seam.

    Args:
        dates: The date column of the input frame (any order).
        ratios: Ordered name -> fraction map (insertion order = split order).

    Returns:
        ``{name: (start, end)}`` with ``start <= end`` for every split.

    Raises:
        SplitError: on invalid ratios, or when there are too few unique dates
            to form non-empty splits.
    """
    names = _validate_ratios(ratios)
    unique = np.unique(pd.to_datetime(pd.Series(dates)).to_numpy())
    n = len(unique)
    ends: list[int] = []
    cumulative = 0.0
    for i, frac in enumerate(ratios.values()):
        cumulative += frac
        ends.append(n if i == len(ratios) - 1 else int(round(n * cumulative)))
    if len(ends) != len(set(ends)) or ends[-1] != n or ends[0] < 1:
        raise SplitError(
            f"Too few unique dates ({n}) to form {len(names)} non-empty "
            f"chronological splits with ratios {dict(ratios)}."
        )
    starts = [0] + ends[:-1]
    return {
        name: (
            pd.Timestamp(unique[s]).strftime("%Y-%m-%d"),
            pd.Timestamp(unique[e - 1]).strftime("%Y-%m-%d"),
        )
        for name, s, e in zip(names, starts, ends)
    }


def chronological_split(
    df: pd.DataFrame,
    date_col: str | None = None,
    label_col: str | None = None,
    ratios: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split a labeled frame into chronological train/val/test subsets.

    Rows are ordered by ``date_col`` (never shuffled), then assigned to the
    split whose date range contains them. The result is validated by
    :func:`validate_splits` before being returned.

    Args:
        df: The labeled feature frame (must contain ``date_col`` and
            ``label_col``).
        date_col: Column used for ordering; defaults to
            ``config.SPLIT_DATE_COLUMN``.
        label_col: Label column; defaults to ``config.LABEL_COLUMN``.
        ratios: Ordered name -> fraction map; defaults to
            ``config.SPLIT_RATIOS``.

    Returns:
        ``{name: DataFrame}`` ordered by split; each frame keeps the input
        columns and is sorted ascending by ``date_col``.

    Raises:
        SplitError: on invalid input, ratios, or an invalid result.
    """
    date_col = date_col or config.SPLIT_DATE_COLUMN
    label_col = label_col or config.LABEL_COLUMN
    ratios = ratios if ratios is not None else dict(config.SPLIT_RATIOS)
    _validate_input(df, date_col, label_col)
    names = _validate_ratios(ratios)

    frame = df.sort_values(date_col).reset_index(drop=True)
    ranges = compute_cutoffs(frame[date_col], ratios)
    splits: dict[str, pd.DataFrame] = {}
    for name in names:
        start, end = ranges[name]
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        mask = (frame[date_col] >= start_ts) & (frame[date_col] <= end_ts)
        splits[name] = frame[mask].copy().reset_index(drop=True)
    validate_splits(splits, len(frame), date_col, label_col)
    for name, cut in ranges.items():
        logger.info(
            "Split %s: %s..%s (%d rows)",
            name,
            cut[0],
            cut[1],
            len(splits[name]),
        )
    return splits


def assert_no_leakage(
    splits: dict[str, pd.DataFrame], date_col: str | None = None
) -> None:
    """Assert strict temporal separation between consecutive splits.

    Every earlier split's newest date must be strictly older than every
    later split's oldest date: ``max(train) < min(val) < min(test)``.

    Raises:
        SplitError: on overlapping or out-of-order date ranges.
    """
    date_col = date_col or config.SPLIT_DATE_COLUMN
    names = list(splits)
    for earlier, later in zip(names, names[1:]):
        max_earlier = splits[earlier][date_col].max()
        min_later = splits[later][date_col].min()
        if pd.isna(max_earlier) or pd.isna(min_later):
            raise SplitError(f"Split {earlier!r} or {later!r} is empty.")
        if max_earlier >= min_later:
            raise SplitError(
                f"Chronological boundary violated: max({earlier}) "
                f"({max_earlier}) >= min({later}) ({min_later})."
            )


def validate_splits(
    splits: dict[str, pd.DataFrame],
    expected_rows: int,
    date_col: str | None = None,
    label_col: str | None = None,
) -> None:
    """Structurally validate the split outputs.

    Checks: non-empty splits, exact row conservation, strict chronological
    separation (via :func:`assert_no_leakage`), no duplicated rows, and
    label values in {0, 1}.

    Raises:
        SplitError: on any violation.
    """
    date_col = date_col or config.SPLIT_DATE_COLUMN
    label_col = label_col or config.LABEL_COLUMN
    if not splits:
        raise SplitError("No splits produced.")
    total = sum(len(part) for part in splits.values())
    if total != expected_rows:
        raise SplitError(
            f"Row conservation violated: {total} rows across splits, "
            f"expected {expected_rows}."
        )
    for name, part in splits.items():
        if part.empty:
            raise SplitError(f"Split {name!r} is empty.")
        if not pd.api.types.is_datetime64_any_dtype(part[date_col]):
            raise SplitError(f"Split {name!r} date column is not datetime64.")
        if part.duplicated().any():
            raise SplitError(f"Split {name!r} contains duplicated rows.")
        if part[label_col].isna().any() or not set(part[label_col].unique()).issubset(
            {0, 1}
        ):
            raise SplitError(f"Split {name!r} has invalid label values.")
    assert_no_leakage(splits, date_col)


def _markdown_table(header: list[str], rows: list[list[object]]) -> list[str]:
    """Render a markdown table (header, separator, rows) as text lines.

    Intentionally mirrors the private helper in ``label_engineer`` rather
    than sharing it, to keep the completed M6 module untouched.
    """
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def write_split_summary(
    splits: dict[str, pd.DataFrame],
    path: Path,
    date_col: str | None = None,
    label_col: str | None = None,
) -> Path:
    """Write a markdown split report: cut dates, rows, and class balance.

    Returns:
        The path the report was written to.
    """
    date_col = date_col or config.SPLIT_DATE_COLUMN
    label_col = label_col or config.LABEL_COLUMN
    ranges = {
        name: (
            part[date_col].min().strftime("%Y-%m-%d"),
            part[date_col].max().strftime("%Y-%m-%d"),
        )
        for name, part in splits.items()
    }
    lines = [
        "# Split Summary",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Method: strict chronological cut over the date axis (no shuffle)",
        f"- Ratios: {config.SPLIT_RATIOS}",
        f"- Total rows: {sum(len(p) for p in splits.values())}",
        "",
        "## Per-split overview",
        "",
    ]
    header = ["split", "rows", "date range", "positives", "positive %"]
    rows: list[list[object]] = []
    for name, part in splits.items():
        positives = int(part[label_col].sum())
        rows.append(
            [
                name,
                len(part),
                f"{ranges[name][0]} → {ranges[name][1]}",
                positives,
                f"{100 * positives / len(part):.2f}%",
            ]
        )
    lines += _markdown_table(header, rows)
    lines += ["", "## Cut dates (exact, logged)", ""]
    lines += _markdown_table(
        ["split", "start", "end"],
        [[name, ranges[name][0], ranges[name][1]] for name in splits],
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote split summary: %s", path)
    return path
