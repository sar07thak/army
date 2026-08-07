"""Data-quality validation and normalization for ACLED data.

Implements every rule in PRD FR-2 and the milestone requirements: required
columns, date parsing, duplicates, country scope, missing values, coordinate
bounds, type coercion, admin-name normalization, geo-unit derivation, and the
district master table. Errors are descriptive and name the offending rows;
nothing is silently dropped.
"""

from __future__ import annotations

import logging

import pandas as pd

import config
from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Fallback date styles tried after pandas' default parser (ACLED event-level style).
_DATE_FORMATS: tuple[str, ...] = ("%d %b %Y",)


def validate_required_columns(frame: pd.DataFrame) -> None:
    """Raise if any canonical required column is missing from ``frame``.

    Raises:
        DataValidationError: listing the missing columns.
    """
    missing = [col for col in config.REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}. Present: {sorted(frame.columns)}."
        )


def parse_event_dates(series: pd.Series) -> pd.Series:
    """Parse ``event_date`` values to ``datetime64``.

    Tries pandas' default parser first, then the ACLED abbreviated format
    (``%d %b %Y``). Unparsable values raise with a sample of offenders.
    Mixed formats within a single column are rejected (with offenders named)
    rather than partially parsed.

    Raises:
        DataValidationError: if any value cannot be parsed.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="raise")
    except (ValueError, TypeError):
        pass
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except (ValueError, TypeError):
            continue
    coerced = pd.to_datetime(series, errors="coerce")
    bad = series[coerced.isna()]
    sample = bad.head(5).tolist()
    raise DataValidationError(
        f"Failed to parse {len(bad)} event_date value(s) (e.g. {sample}); "
        "expected ISO dates or the ACLED '%d %b %Y' format."
    )


def filter_date_range(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict the frame to the configured study window (inclusive).

    Out-of-window rows are counted, logged, and removed.
    """
    start = pd.Timestamp(config.DATE_START)
    end = pd.Timestamp(config.DATE_END)
    mask = frame["event_date"].between(start, end)
    dropped = int((~mask).sum())
    if dropped:
        logger.warning(
            "Dropped %d rows outside the study window %s..%s",
            dropped,
            start.date(),
            end.date(),
        )
    return frame[mask].copy()


def _effectively_missing(series: pd.Series) -> pd.Series:
    """Return True where a value is NaN or an empty/whitespace-only string."""
    if pd.api.types.is_string_dtype(series):
        stripped = series.astype("string").str.strip()
        return stripped.isna() | (stripped == "")
    return series.isna()


def validate_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Report missing values and drop rows lacking critical geo/date fields.

    Missing (including empty-string) counts per column are logged. Rows with
    a null or empty ``event_date``, ``country``, or ``admin1`` are dropped
    after logging; if the dropped fraction exceeds
    ``config.MAX_DROPPED_FRACTION`` an error is raised instead.

    Raises:
        DataValidationError: if the missing-data fraction is excessive.
    """
    critical = ("event_date", "country", "admin1")
    for col in frame.columns:
        n_missing = int(_effectively_missing(frame[col]).sum())
        if n_missing:
            logger.warning("Column %r has %d missing/empty value(s)", col, n_missing)
    mask = pd.Series(True, index=frame.index)
    for col in critical:
        mask &= ~_effectively_missing(frame[col])
    dropped = int((~mask).sum())
    if dropped:
        fraction = dropped / len(frame)
        if fraction > config.MAX_DROPPED_FRACTION:
            raise DataValidationError(
                f"{dropped} rows ({fraction:.1%}) are missing critical fields "
                f"{critical}; exceeds MAX_DROPPED_FRACTION={config.MAX_DROPPED_FRACTION}."
            )
        logger.warning("Dropped %d rows missing critical fields %s", dropped, critical)
    return frame[mask].copy()


def validate_countries(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the country scope.

    ``COUNTRIES_MODE="filter"`` removes out-of-scope countries with a logged
    per-country count; ``"error"`` raises instead.

    Raises:
        DataValidationError: if ``COUNTRIES_MODE == "error"`` and out-of-scope
            countries are present.
    """
    in_scope = frame["country"].isin(config.COUNTRIES)
    n_out = int((~in_scope).sum())
    if n_out == 0:
        return frame
    if config.COUNTRIES_MODE == "error":
        raise DataValidationError(
            f"{n_out} rows belong to countries outside the scope "
            f"{config.COUNTRIES} (COUNTRIES_MODE='error')."
        )
    counts = frame.loc[~in_scope, "country"].value_counts().to_dict()
    logger.warning("Filtered %d out-of-scope row(s) by country: %s", n_out, counts)
    return frame[in_scope].copy()


def validate_coordinates(frame: pd.DataFrame) -> None:
    """Assert latitude/longitude are numeric and within valid bounds.

    Non-numeric or missing coordinates raise with a sample of offenders;
    numeric columns are written back to ``frame`` in place.

    Raises:
        DataValidationError: if coordinates are missing/non-numeric or
            out-of-bounds.
    """
    numeric = frame[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid = frame[["latitude", "longitude"]][numeric.isna().any(axis=1)]
    if not invalid.empty:
        sample = invalid.head(5).to_dict("records")
        raise DataValidationError(
            f"{len(invalid)} row(s) have missing or non-numeric coordinates; "
            f"e.g. {sample}"
        )
    frame["latitude"] = numeric["latitude"]
    frame["longitude"] = numeric["longitude"]
    lat_ok = frame["latitude"].between(config.LAT_MIN, config.LAT_MAX)
    lon_ok = frame["longitude"].between(config.LON_MIN, config.LON_MAX)
    bad = frame[~(lat_ok & lon_ok)]
    if not bad.empty:
        sample = bad[["latitude", "longitude"]].head(5).to_dict("records")
        raise DataValidationError(
            f"{len(bad)} row(s) have out-of-bounds coordinates (lat in "
            f"[{config.LAT_MIN}, {config.LAT_MAX}], lon in "
            f"[{config.LON_MIN}, {config.LON_MAX}]); e.g. {sample}"
        )


def _coerce_count(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Coerce a count column to non-negative int64; raise on bad values."""
    coerced = pd.to_numeric(frame[column], errors="coerce")
    bad_values = frame[column][coerced.isna()]
    if len(bad_values):
        raise DataValidationError(
            f"{len(bad_values)} non-numeric value(s) in column {column!r}: "
            f"{bad_values.head(5).tolist()}"
        )
    negatives = coerced[coerced < 0]
    if len(negatives):
        raise DataValidationError(
            f"{len(negatives)} negative value(s) in column {column!r}: "
            f"{negatives.head(5).tolist()}"
        )
    frame[column] = coerced.astype("int64")
    return frame


def validate_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns and normalize string columns.

    - ``fatalities``/``events`` are coerced to non-negative int64.
    - ``actor1`` nulls become ``config.DEFAULT_ACTOR_VALUE``.
    - ``event_type`` is stripped of surrounding whitespace.

    Raises:
        DataValidationError: on non-numeric or negative counts.
    """
    for column in ("fatalities", "events"):
        if column in frame.columns:
            frame = _coerce_count(frame, column)
    if "actor1" in frame.columns:
        n_null = int(frame["actor1"].isna().sum())
        if n_null:
            logger.warning(
                "Filled %d null actor1 value(s) with %r",
                n_null,
                config.DEFAULT_ACTOR_VALUE,
            )
        frame["actor1"] = frame["actor1"].fillna(config.DEFAULT_ACTOR_VALUE)
    if "event_type" in frame.columns:
        frame["event_type"] = frame["event_type"].astype("string").str.strip()
    return frame


def normalize_admin_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize admin1/admin2 names: strip, collapse whitespace, map fixes.

    ``config.ADMIN_NAME_NORMALIZATION`` fixes known inconsistencies. All-caps
    acronyms such as "FATA" are left untouched — they are legitimate names.
    """
    for column in ("admin1", "admin2"):
        if column not in frame.columns:
            continue
        cleaned = (
            frame[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        )
        if config.ADMIN_NAME_NORMALIZATION:
            cleaned = cleaned.replace(config.ADMIN_NAME_NORMALIZATION)
        frame[column] = cleaned
    return frame


def derive_geo_unit(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``geo_unit`` = finest available admin level (admin2 > admin1).

    When ``admin2`` exists and is non-empty for a row it becomes the geo
    unit; otherwise ``admin1`` is used (which is the case for the aggregated
    weekly file).
    """
    if "admin2" in frame.columns:
        admin2_clean = frame["admin2"].astype("string").str.strip()
        has_admin2 = admin2_clean.notna() & (admin2_clean != "")
        frame["geo_unit"] = admin2_clean.where(has_admin2, frame["admin1"])
    else:
        frame["geo_unit"] = frame["admin1"]
    logger.info(
        "Derived geo_unit: %d distinct units (finest admin level)",
        frame["geo_unit"].nunique(),
    )
    return frame


def derive_event_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``event_id`` from ``event_id_cnty`` when present, else a composite.

    The composite key joins the date, country, available admin levels, and
    event-type fields with ``|`` so aggregated weekly rows get a stable
    identity for deduplication.
    """
    if config.EVENT_ID_SOURCE in frame.columns:
        frame["event_id"] = frame[config.EVENT_ID_SOURCE].astype("string")
        return frame
    key_cols = [
        col
        for col in (
            "event_date",
            "country",
            "admin1",
            "admin2",
            "event_type",
            "sub_event_type",
        )
        if col in frame.columns
    ]
    key_frame = frame[key_cols].copy()
    if pd.api.types.is_datetime64_any_dtype(key_frame["event_date"]):
        key_frame["event_date"] = key_frame["event_date"].dt.strftime("%Y-%m-%d")
    frame["event_id"] = key_frame.astype(str).agg("|".join, axis=1)
    logger.info(
        "No %r column; derived composite event_id from %s",
        config.EVENT_ID_SOURCE,
        key_cols,
    )
    return frame


def validate_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove (or reject) rows with a duplicated ``event_id``.

    ``event_id`` must exist (built by :func:`derive_event_ids`). With
    ``config.DUPLICATES_MODE == "drop"`` duplicates are removed after a
    warning; with ``"raise"`` they raise.

    Raises:
        DataValidationError: if ``event_id`` is missing, or if
            ``DUPLICATES_MODE == "raise"`` and duplicates exist.
    """
    if "event_id" not in frame.columns:
        raise DataValidationError("Cannot check duplicates: 'event_id' column missing.")
    mask = frame.duplicated(subset="event_id", keep="first")
    n_dupes = int(mask.sum())
    if n_dupes == 0:
        return frame
    if config.DUPLICATES_MODE == "raise":
        raise DataValidationError(
            f"Found {n_dupes} duplicate event_id rows; DUPLICATES_MODE='raise'."
        )
    logger.warning("Removed %d duplicate event_id row(s) (DUPLICATES_MODE='drop')", n_dupes)
    return frame[~mask].copy()


def validate_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the full validation pipeline on a canonical raw frame.

    Steps (in order): required columns -> date parsing -> date window ->
    missing critical fields -> country scope -> type coercion -> coordinates
    -> admin-name normalization -> geo-unit derivation -> event-id derivation
    -> duplicates.

    Returns a clean, typed, normalized DataFrame ready for feature
    engineering.

    Raises:
        DataValidationError: on any violated rule, or if the result is empty.
    """
    validate_required_columns(raw)
    frame = raw.copy()
    frame["event_date"] = parse_event_dates(frame["event_date"])
    frame = filter_date_range(frame)
    frame = validate_missing(frame)
    frame = validate_countries(frame)
    frame = validate_types(frame)
    validate_coordinates(frame)
    frame = normalize_admin_names(frame)
    frame = derive_geo_unit(frame)
    frame = derive_event_ids(frame)
    frame = validate_duplicates(frame)
    if frame.empty:
        raise DataValidationError(
            "Validation produced an empty dataset — check the scope filters."
        )
    logger.info(
        "Validation complete: %d rows, %d geo units, dates %s..%s",
        len(frame),
        frame["geo_unit"].nunique(),
        frame["event_date"].min().date(),
        frame["event_date"].max().date(),
    )
    return frame


def build_district_master(clean: pd.DataFrame) -> pd.DataFrame:
    """Build the district (geo-unit) master table.

    One row per ``geo_unit`` with its admin1, country, coordinates (first
    non-null location), event and row counts, and first/last event dates.
    Sorted by country, admin1, geo_unit.

    Raises:
        DataValidationError: if a required column is missing.
    """
    for col in ("geo_unit", "admin1", "country"):
        if col not in clean.columns:
            raise DataValidationError(
                f"Cannot build district master: missing column {col!r}."
            )
    grouped = clean.groupby("geo_unit", as_index=False)
    master = grouped.agg(
        admin1=("admin1", "first"),
        country=("country", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        total_events=("events", "sum"),
        n_rows=("events", "size"),
        first_date=("event_date", "min"),
        last_date=("event_date", "max"),
    )
    master = master.sort_values(["country", "admin1", "geo_unit"]).reset_index(drop=True)
    logger.info("District master: %d geo units", len(master))
    return master
