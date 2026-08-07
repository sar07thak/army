"""Adaptive loading of ACLED data from local CSV files.

The loader is deliberately schema-adaptive: it canonicalizes whatever columns
the provided export contains onto one internal schema (IMPLEMENTATION_PLAN
§3.4). Both the event-level ACLED export and the aggregated weekly count file
are supported. No API or data-export functionality is implemented.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config
from src.exceptions import DataLoadError

logger = logging.getLogger(__name__)


def discover_raw_files(raw_dir: Path) -> list[Path]:
    """Return the sorted, case-deduplicated CSV files under ``raw_dir``.

    Args:
        raw_dir: Directory containing the raw ACLED CSV export(s).

    Raises:
        DataLoadError: if the directory does not exist or contains no CSVs.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise DataLoadError(f"Raw data directory not found: {raw_dir}")
    files = sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.CSV"))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in files:
        key = path.name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise DataLoadError(
            f"No CSV files found in {raw_dir}. Place the ACLED export here "
            "(e.g. the Data Export Tool CSV) before running the pipeline."
        )
    logger.info("Discovered %d raw CSV file(s) in %s", len(unique), raw_dir)
    return unique


def load_single_csv(path: Path) -> pd.DataFrame:
    """Read one ACLED CSV into a DataFrame.

    Raises:
        DataLoadError: if the file cannot be read as a table.
    """
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas raises several low-level errors on bad files
        raise DataLoadError(f"Failed to read {path.name}: {exc}") from exc
    logger.info(
        "Loaded %s: %d rows x %d columns", path.name, frame.shape[0], frame.shape[1]
    )
    return frame


def merge_raw_files(files: list[Path]) -> pd.DataFrame:
    """Load and concatenate several raw CSV files into a single DataFrame.

    Raises:
        DataLoadError: if any file fails to load (propagated from
            :func:`load_single_csv`).
    """
    frames = [load_single_csv(path) for path in files]
    if len(frames) == 1:
        return frames[0]
    merged = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Merged %d file(s) -> %d rows", len(frames), len(merged))
    return merged


def canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw source columns onto the canonical schema.

    Adapts to whichever columns are present:

    - ``event_date`` comes from ``event_date`` or ``week`` (aggregated file).
    - ``latitude``/``longitude`` come from the same names or the
      ``centroid_latitude``/``centroid_longitude`` pairs.
    - ``events`` comes from ``events`` or defaults to ``1`` per row for
      event-level files (each row is one event).
    - ``event_id`` and ``geo_unit`` are deliberately NOT derived here —
      validation builds them after name normalization.

    Raises:
        DataLoadError: if no date source or no coordinate source exists.
    """
    frame = raw.copy()
    available = set(frame.columns)

    if "event_date" not in available and "week" not in available:
        raise DataLoadError(
            "No event date source found. Expected an 'event_date' or 'week' "
            f"column; present columns: {sorted(available)}"
        )
    if "event_date" not in available and "week" in available:
        frame = frame.rename(columns={"week": "event_date"})

    if "latitude" not in frame.columns or "longitude" not in frame.columns:
        if {"centroid_latitude", "centroid_longitude"} <= available:
            frame = frame.rename(
                columns={
                    "centroid_latitude": "latitude",
                    "centroid_longitude": "longitude",
                }
            )
        else:
            raise DataLoadError(
                "No coordinate source found. Expected 'latitude'/'longitude' or "
                f"'centroid_latitude'/'centroid_longitude'; present columns: "
                f"{sorted(available)}"
            )

    if "events" not in frame.columns:
        frame["events"] = 1
        logger.info("No 'events' column (event-level file); each row counts as 1 event.")

    logger.info(
        "Canonicalized frame: %d rows x %d columns", frame.shape[0], frame.shape[1]
    )
    return frame


def load_raw_data(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load, merge, and canonicalize all ACLED CSVs under ``raw_dir``.

    Args:
        raw_dir: Override for the raw data directory (defaults to
            ``config.DATA_RAW_DIR``).

    Raises:
        DataLoadError: if no CSVs exist or canonicalization fails.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else config.DATA_RAW_DIR
    files = discover_raw_files(raw_dir)
    merged = merge_raw_files(files)
    return canonicalize(merged)


def save_dataframe(
    frame: pd.DataFrame,
    name: str,
    out_dir: Path,
    formats: tuple[str, ...] | None = None,
) -> list[Path]:
    """Write ``frame`` to ``out_dir`` in the requested formats.

    Args:
        frame: DataFrame to persist.
        name: Base file name without extension.
        out_dir: Destination directory (created if missing).
        formats: Extensions to write (default ``config.OUTPUT_FORMATS``).

    Raises:
        DataLoadError: if a format is unsupported or writing fails.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or config.OUTPUT_FORMATS
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        try:
            if fmt == "parquet":
                frame.to_parquet(path, index=False)
            elif fmt == "csv":
                frame.to_csv(path, index=False)
            else:
                raise DataLoadError(f"Unsupported output format: {fmt!r}")
        except DataLoadError:
            raise
        except Exception as exc:
            raise DataLoadError(f"Failed to write {path.name}: {exc}") from exc
        written.append(path)
        logger.info("Saved %s (%d rows)", path.name, len(frame))
    return written


def save_clean_outputs(
    clean: pd.DataFrame, master: pd.DataFrame, out_dir: Path
) -> dict[str, list[Path]]:
    """Persist the cleaned events and the district master table.

    Returns:
        Mapping of artifact name -> list of written paths.
    """
    return {
        "cleaned_events": save_dataframe(
            clean, config.CLEANED_EVENTS_FILE, out_dir
        ),
        "district_master": save_dataframe(
            master, config.DISTRICT_MASTER_FILE, out_dir
        ),
    }
