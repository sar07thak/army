"""Live 14-day forecast stage (post-M13 ``--stage forecast``).

Loads the winning model (``models/escalation_best.pkl`` — never retrains)
and the **latest available feature row per geo unit** from
``data/processed/features.parquet`` (the real "as of today" state, NOT the
held-out test window), computes predictions + SHAP values, and produces the
operational forecast artifacts:

- ``reports/forecast_next_14_days.csv`` — one row per geo unit with the
  predicted escalation probability, class, risk category, top-3 SHAP
  drivers, and recent 7-day events/fatalities
- ``reports/maps/forecast_risk_map.html`` — interactive risk map anchored
  at the forecast date (folium)
- ``reports/forecast_summary.md`` — highest/safest regions, country
  averages, top drivers, interpretation, observations

Unlike the visualization stage (which explains the model on the out-of-sample
test window), this stage is the *operational* forecast: it scores the most
recent state of every geo unit and frames the output as "next 14 days" risk.
Leakage guarantees are inherited — features are past-only by construction and
the model never sees labels here.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import models
from src.exceptions import DataLoadError, ForecastError
from src.explainability import (
    _operating_threshold,
    compute_shap_values,
    load_winning_model,
    mean_abs_importance,
)
from src.visualization import (
    _country_metrics,
    _country_rows,
    _driver_rows,
    _load_centroids,
    _md_table,
    _region_rows,
    _validate_shap_consistency,
    build_prediction_table,
    save_risk_map,
)

logger = logging.getLogger(__name__)


def load_latest_features(data_dir: Path) -> pd.DataFrame:
    """Load the latest feature row per geo unit from the features table.

    ``features.parquet`` holds one row per geo unit per date; the forecast
    uses each unit's most recent row (its current state).

    Raises:
        DataLoadError: if the features table is missing.
        ForecastError: if no rows remain.
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{config.FEATURES_FILE}.parquet"
    if not path.is_file():
        raise DataLoadError(
            f"Features table not found: {path} — run the 'features' stage first."
        )
    frame = pd.read_parquet(path)
    if len(frame) == 0:
        raise ForecastError("Features table is empty — nothing to forecast.")
    latest = frame.loc[frame.groupby("geo_unit")["event_date"].idxmax()].reset_index(drop=True)
    logger.info(
        "Forecast as-of state: %d geo units, latest date %s",
        len(latest),
        latest["event_date"].max().date(),
    )
    return latest


def resolve_forecast_features(
    latest: pd.DataFrame, models_dir: Path
) -> list[str]:
    """Feature names in the exact order the model was trained on.

    Uses ``model_comparison.json``'s ``features`` list (the single source of
    truth for the winner), falling back to the manifest's
    ``feature_columns`` and then to :func:`models.resolve_feature_columns`.

    Raises:
        ForecastError: if any training feature is missing from the frame.
    """
    models_dir = Path(models_dir)
    for fname, key in (
        (config.MODEL_COMPARISON_FILE, "features"),
        (config.MODEL_MANIFEST_FILE, "feature_columns"),
    ):
        path = models_dir / fname
        if not path.is_file():
            continue
        try:
            import json

            doc = json.loads(path.read_text(encoding="utf-8"))
            features = doc.get(key)
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            logger.warning("Could not read %s (%s); trying next source", path.name, exc)
            continue
        if features:
            missing = [f for f in features if f not in latest.columns]
            if missing:
                raise ForecastError(
                    f"Training features missing from the features table: {missing}. "
                    f"Re-run the 'features' stage after any config change."
                )
            logger.info("Resolved %d forecast features from %s", len(features), fname)
            return list(features)
    features = models.resolve_feature_columns(latest)
    logger.info("Resolved %d forecast features by column derivation", len(features))
    return features


def _prepare_matrix(latest: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Float feature matrix for the latest rows, NaN-checked.

    Raises:
        ForecastError: on non-finite feature values.
    """
    X = latest[features].to_numpy(dtype="float64")
    if not np.isfinite(X).all():
        raise ForecastError("Latest feature matrix contains NaN or infinite values.")
    return X


def build_forecast_table(
    model: object,
    latest: pd.DataFrame,
    features: list[str],
    threshold: float,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions + SHAP + snapshot table for the latest feature rows.

    Args:
        model: The winning classifier (``escalation_best.pkl``).
        latest: One latest feature row per geo unit.
        features: Training feature order (see :func:`resolve_forecast_features`).
        threshold: Operating threshold for the predicted class.
        data_dir: Processed-data dir (for centroids).

    Returns:
        ``(table, importance)`` — the one-row-per-geo-unit snapshot table and
        the mean-|SHAP| feature ranking.

    Raises:
        ForecastError: on any prediction/SHAP inconsistency.
    """
    X = _prepare_matrix(latest, features)
    if not hasattr(model, "predict_proba"):
        raise ForecastError("Loaded artifact is not a fitted classifier.")
    trained = getattr(model, "n_features_in_", None)
    if trained is not None and trained != len(features):
        raise ForecastError(
            f"Model expects {trained} features but the features table provides "
            f"{len(features)}. Re-run 'features' + 'compare' after any config change."
        )
    proba = models.predict_proba(model, X)
    _, shap_values, base_value = compute_shap_values(model, X, features)
    _validate_shap_consistency(shap_values, proba, base_value)
    importance = mean_abs_importance(shap_values, features)
    table = build_prediction_table(
        latest, proba, shap_values, features, threshold, _load_centroids(data_dir)
    )
    return table, importance


def save_forecast_csv(table: pd.DataFrame, reports_dir: Path) -> Path:
    """Write the one-row-per-geo-unit forecast CSV.

    Raises:
        ForecastError: if a required column is missing.
    """
    cols = [
        "geo_unit",
        "country",
        "admin1",
        "event_date",
        config.PREDICTION_PROBA_COLUMN,
        config.PREDICTION_CLASS_COLUMN,
        config.PREDICTION_CATEGORY_COLUMN,
        "top_drivers",
        config.RECENT_EVENTS_COLUMN,
        config.RECENT_FATALITIES_COLUMN,
    ]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise ForecastError(f"Forecast table missing columns: {missing}.")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / config.FORECAST_CSV_FILE
    table[cols].to_csv(path, index=False)
    logger.info("Wrote forecast CSV: %s", path)
    return path


def save_forecast_map(table: pd.DataFrame, reports_dir: Path) -> Path:
    """Interactive risk map anchored at the forecast date."""
    map_dir = Path(reports_dir) / config.MAPS_DIR.name
    return save_risk_map(table, map_dir, filename=config.FORECAST_MAP_FILE)


def _observations(
    table: pd.DataFrame,
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
) -> list[str]:
    """Data-driven observations for the forecast summary report."""
    top = table.nlargest(1, config.PREDICTION_PROBA_COLUMN).iloc[0]
    counts = table[config.PREDICTION_CATEGORY_COLUMN].value_counts()
    risky_share = (
        counts.get("Critical", 0) + counts.get("High", 0)
    ) / len(table) if len(table) else 0.0
    top_country = metrics.iloc[0]["country"]
    top_driver = importance.iloc[0]["feature"]
    return [
        f"Highest-risk geo unit over the next 14 days: **{top['geo_unit']}** "
        f"({top['country']}) with probability {top[config.PREDICTION_PROBA_COLUMN]:.3f}.",
        f"{100 * risky_share:.1f}% of geo units are High or Critical risk "
        f"({counts.to_dict()}).",
        f"Highest average risk country: **{top_country}** "
        f"(avg {metrics.iloc[0]['avg_risk']:.3f}).",
        f"Strongest overall risk driver: **{top_driver}** "
        f"(mean |SHAP| {importance.iloc[0]['mean_abs_shap']:.4f}).",
        "This is a live forecast on each unit's most recent state — the model "
        "was never retrained for forecasting.",
    ]


def write_forecast_summary(
    table: pd.DataFrame,
    importance: pd.DataFrame,
    metrics: pd.DataFrame,
    threshold: float,
    path: Path,
) -> Path:
    """Render ``reports/forecast_summary.md`` (regions, averages, drivers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = table.nlargest(config.FORECAST_TOP_K, config.PREDICTION_PROBA_COLUMN)
    safest = table.nsmallest(config.FORECAST_TOP_K, config.PREDICTION_PROBA_COLUMN)
    region_header = ["rank", "geo unit", "country", "probability", "class", "category"]
    lines = [
        "# Live 14-Day Forecast Summary",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model: {config.MODEL_BEST_FILE} · operating threshold: {threshold:.2f}",
        f"- As-of date: {table['event_date'].max().date()} "
        f"(prediction window: next {config.LABEL_HORIZON_DAYS} days)",
        f"- Scope: {len(table)} geo units across {table['country'].nunique()} countries",
        "",
        f"## Highest-risk regions (top {config.FORECAST_TOP_K})",
        "",
        _md_table(region_header, _region_rows(top)),
        "",
        f"## Safest regions (bottom {config.FORECAST_TOP_K})",
        "",
        _md_table(region_header, _region_rows(safest)),
        "",
        "## Average risk by country",
        "",
        _md_table(
            ["country", "avg risk", "positive rate", "mean events (7d)", "mean fatalities (7d)"],
            _country_rows(metrics),
        ),
        "",
        "## Top SHAP drivers",
        "",
        _md_table(["rank", "feature", "mean |SHAP|"], _driver_rows(importance)),
        "",
        "## Interpretation",
        "",
        "Each geo unit is scored at its most recent available feature date; the "
        "predicted probability is the model's assessment that escalation occurs "
        "within the next 14 days. The risk category comes from "
        "`config.RISK_LEVEL_BOUNDARIES`; the predicted class uses the operating "
        "threshold from `model_comparison.json`.",
        "",
        "## Important observations",
        "",
    ]
    lines += [f"- {obs}" for obs in _observations(table, metrics, importance)]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote forecast summary: %s", path)
    return path


def forecast_stage(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, object]:
    """Run the live 14-day forecast stage on the winning model.

    Loads ``escalation_best.pkl`` + the latest feature row per geo unit,
    computes predictions and SHAP, and writes the forecast CSV, risk map and
    summary report under ``reports/``.

    Returns:
        Summary dict (artifact paths, geo units, top driver, as-of date).

    Raises:
        ConflictForecastError: on missing artifacts or invalid inputs.
    """
    data_dir = Path(data_dir or config.DATA_PROCESSED_DIR)
    models_dir = Path(models_dir or config.MODELS_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    model = load_winning_model(models_dir)
    latest = load_latest_features(data_dir)
    features = resolve_forecast_features(latest, models_dir)
    threshold = _operating_threshold(models_dir)
    table, importance = build_forecast_table(
        model, latest, features, threshold, data_dir
    )

    csv_path = save_forecast_csv(table, reports_dir)
    map_path = save_forecast_map(table, reports_dir)
    metrics = _country_metrics(table)
    summary_path = write_forecast_summary(
        table,
        importance,
        metrics,
        threshold,
        reports_dir / config.FORECAST_SUMMARY_FILE,
    )

    top = table.nlargest(1, config.PREDICTION_PROBA_COLUMN).iloc[0]
    summary = {
        "geo_units": int(len(table)),
        "countries": int(table["country"].nunique()),
        "as_of_date": str(table["event_date"].max().date()),
        "operating_threshold": threshold,
        "risk_categories": table[config.PREDICTION_CATEGORY_COLUMN]
        .value_counts()
        .to_dict(),
        "top_driver": {
            "feature": str(importance.iloc[0]["feature"]),
            "mean_abs_shap": float(importance.iloc[0]["mean_abs_shap"]),
        },
        "highest_risk": {
            "geo_unit": str(top["geo_unit"]),
            "probability": float(top[config.PREDICTION_PROBA_COLUMN]),
        },
        "forecast_csv": str(csv_path),
        "risk_map": str(map_path),
        "summary_report": str(summary_path),
    }
    logger.info(
        "Forecast stage complete: %d geo units as of %s; highest risk = %s (%.3f)",
        len(table),
        summary["as_of_date"],
        top["geo_unit"],
        top[config.PREDICTION_PROBA_COLUMN],
    )
    return summary
