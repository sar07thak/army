"""Tests for ``src.forecast`` (post-M13 --stage forecast).

Covers latest-row selection, feature-order resolution, the forecast table
build (predictions + SHAP + drivers), CSV/map/summary outputs, the
end-to-end ``forecast_stage`` on a small trained model, and error paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import forecast, models
from src.exceptions import DataLoadError, ForecastError


def _features_frame(n: int = 60, seed: int = 11) -> pd.DataFrame:
    """Synthetic features table (no label) for two geo units over dates."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    dates = pd.date_range("2026-01-01", periods=n, freq="7D")
    return pd.DataFrame(
        {
            "geo_unit": ["G1"] * n1 + ["G2"] * (n - n1),
            "country": ["India"] * n1 + ["Pakistan"] * (n - n1),
            "admin1": ["Bihar"] * n1 + ["Sindh"] * (n - n1),
            "event_date": dates,
            "events_w7d": rng.integers(0, 10, size=n).astype(float),
            "fatalities_w7d": rng.integers(0, 8, size=n).astype(float),
            "velocity_events_w14d": rng.normal(size=n),
            "spillover_w14d": rng.integers(0, 20, size=n).astype(float),
        }
    )


def _labeled_frame(n: int = 120, seed: int = 13) -> pd.DataFrame:
    """Synthetic labeled frame with the SAME feature columns (for training)."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    dates = pd.date_range("2025-01-01", periods=n, freq="7D")
    return pd.DataFrame(
        {
            "geo_unit": ["G1"] * n1 + ["G2"] * (n - n1),
            "country": ["India"] * n1 + ["Pakistan"] * (n - n1),
            "admin1": ["Bihar"] * n1 + ["Sindh"] * (n - n1),
            "event_date": dates,
            "events_w7d": rng.integers(0, 10, size=n).astype(float),
            "fatalities_w7d": rng.integers(0, 8, size=n).astype(float),
            "velocity_events_w14d": rng.normal(size=n),
            "spillover_w14d": rng.integers(0, 20, size=n).astype(float),
            config.LABEL_COLUMN: rng.integers(0, 2, size=n),
        }
    )


def _fit_tiny_model(frame: pd.DataFrame) -> tuple[object, list[str]]:
    """Train a tiny deterministic LGBM; return (model, features)."""
    X, y, features = models.prepare_xy(frame)
    model, _ = models.train_model(
        X, y, params={"n_estimators": 30, "num_leaves": 7, "verbosity": -1}
    )
    return model, features


def _write_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write features.parquet + escalation_best.pkl + comparison + centroids."""
    features = _features_frame(60)
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()

    labeled = _labeled_frame(120)
    model, feature_names = _fit_tiny_model(labeled)
    models.save_model(model, models_dir / config.MODEL_BEST_FILE)
    features.to_parquet(data_dir / f"{config.FEATURES_FILE}.parquet")

    cleaned = features[["geo_unit", "event_date", "events_w7d", "fatalities_w7d"]].copy()
    cleaned["latitude"] = [25.5] * 60
    cleaned["longitude"] = [85.0] * 60
    cleaned.to_parquet(data_dir / f"{config.CLEANED_EVENTS_FILE}.parquet")

    (models_dir / config.MODEL_COMPARISON_FILE).write_text(
        json.dumps({"operating_threshold": 0.30, "features": feature_names}),
        encoding="utf-8",
    )
    return data_dir, models_dir, reports_dir


# ---------------------------------------------------------------------------
# Latest-row selection
# ---------------------------------------------------------------------------


def test_load_latest_features_one_row_per_unit(tmp_path: Path) -> None:
    data_dir, _, _ = _write_env(tmp_path)
    latest = forecast.load_latest_features(data_dir)
    assert len(latest) == 2
    assert set(latest["geo_unit"]) == {"G1", "G2"}
    # Each row is the max-date row for its unit.
    expected_max = _features_frame(60).groupby("geo_unit")["event_date"].max()
    assert latest.set_index("geo_unit")["event_date"].equals(expected_max)


# ---------------------------------------------------------------------------
# Feature resolution
# ---------------------------------------------------------------------------


def test_resolve_forecast_features_from_comparison(tmp_path: Path) -> None:
    data_dir, models_dir, _ = _write_env(tmp_path)
    latest = forecast.load_latest_features(data_dir)
    features = forecast.resolve_forecast_features(latest, models_dir)
    assert set(features) == {
        "events_w7d",
        "fatalities_w7d",
        "velocity_events_w14d",
        "spillover_w14d",
    }


def test_resolve_forecast_features_missing_raises(tmp_path: Path) -> None:
    data_dir, models_dir, _ = _write_env(tmp_path)
    latest = forecast.load_latest_features(data_dir).drop(columns=["spillover_w14d"])
    with pytest.raises(ForecastError, match="missing from the features table"):
        forecast.resolve_forecast_features(latest, models_dir)


def test_resolve_forecast_features_fallback_derivation(tmp_path: Path) -> None:
    data_dir, models_dir, _ = _write_env(tmp_path)
    (models_dir / config.MODEL_COMPARISON_FILE).unlink()
    (models_dir / config.MODEL_MANIFEST_FILE).unlink(missing_ok=True)
    latest = forecast.load_latest_features(data_dir)
    features = forecast.resolve_forecast_features(latest, models_dir)
    assert set(features) == {
        "events_w7d",
        "fatalities_w7d",
        "velocity_events_w14d",
        "spillover_w14d",
    }


# ---------------------------------------------------------------------------
# Forecast table
# ---------------------------------------------------------------------------


def test_build_forecast_table(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_env(tmp_path)
    model = models.load_model(models_dir / config.MODEL_BEST_FILE)
    latest = forecast.load_latest_features(data_dir)
    features = forecast.resolve_forecast_features(latest, models_dir)
    table, importance = forecast.build_forecast_table(
        model, latest, features, 0.30, data_dir
    )
    assert len(table) == 2
    assert set(table["geo_unit"]) == {"G1", "G2"}
    assert table[config.PREDICTION_PROBA_COLUMN].between(0, 1).all()
    assert table[config.PREDICTION_CLASS_COLUMN].isin([0, 1]).all()
    assert table[config.PREDICTION_CATEGORY_COLUMN].isin(config.RISK_LEVEL_NAMES).all()
    assert table["top_drivers"].str.contains(": ").all()
    assert table["lat"].tolist() == pytest.approx([25.5, 25.5])
    assert list(importance["feature"]) == importance.sort_values(
        "mean_abs_shap", ascending=False
    )["feature"].tolist()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def test_save_forecast_csv(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_env(tmp_path)
    model = models.load_model(models_dir / config.MODEL_BEST_FILE)
    latest = forecast.load_latest_features(data_dir)
    features = forecast.resolve_forecast_features(latest, models_dir)
    table, _ = forecast.build_forecast_table(model, latest, features, 0.30, data_dir)
    path = forecast.save_forecast_csv(table, reports_dir)
    assert path.name == config.FORECAST_CSV_FILE
    df = pd.read_csv(path)
    assert len(df) == 2
    for col in (
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
    ):
        assert col in df.columns


def test_save_forecast_map(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_env(tmp_path)
    model = models.load_model(models_dir / config.MODEL_BEST_FILE)
    latest = forecast.load_latest_features(data_dir)
    features = forecast.resolve_forecast_features(latest, models_dir)
    table, _ = forecast.build_forecast_table(model, latest, features, 0.30, data_dir)
    path = forecast.save_forecast_map(table, reports_dir)
    assert path.name == config.FORECAST_MAP_FILE
    html = path.read_text(encoding="utf-8")
    assert "leaflet" in html.lower()
    assert "G1" in html and "G2" in html


def test_write_forecast_summary(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "geo_unit": ["G1", "G2"],
            "country": ["India", "Pakistan"],
            "event_date": pd.to_datetime(["2026-07-25", "2026-07-25"]),
            config.PREDICTION_PROBA_COLUMN: [0.9, 0.1],
            config.PREDICTION_CLASS_COLUMN: [1, 0],
            config.PREDICTION_CATEGORY_COLUMN: ["Critical", "Low"],
        }
    )
    importance = pd.DataFrame(
        {"feature": ["events_w7d", "fatalities_w7d"], "mean_abs_shap": [0.6, 0.4]}
    )
    metrics = pd.DataFrame(
        {
            "country": ["India", "Pakistan"],
            "avg_risk": [0.9, 0.1],
            "positive_rate": [1.0, 0.0],
            "mean_events": [10.0, 5.0],
            "mean_fatalities": [3.0, 1.0],
        }
    )
    path = forecast.write_forecast_summary(
        table, importance, metrics, 0.25, tmp_path / config.FORECAST_SUMMARY_FILE
    )
    text = path.read_text(encoding="utf-8")
    for section in (
        "Live 14-Day Forecast Summary",
        "Highest-risk regions",
        "Safest regions",
        "Average risk by country",
        "Top SHAP drivers",
        "Interpretation",
        "Important observations",
    ):
        assert section in text
    assert "next 14 days" in text
    assert "G1" in text


# ---------------------------------------------------------------------------
# End-to-end stage
# ---------------------------------------------------------------------------


def test_forecast_stage_end_to_end(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_env(tmp_path)
    summary = forecast.forecast_stage(
        data_dir=data_dir, models_dir=models_dir, reports_dir=reports_dir
    )
    assert summary["geo_units"] == 2
    assert summary["countries"] == 2
    assert summary["as_of_date"] == str(_features_frame(60)["event_date"].max().date())
    assert summary["operating_threshold"] == pytest.approx(0.30)
    assert set(summary["risk_categories"]) <= set(config.RISK_LEVEL_NAMES)
    assert summary["top_driver"]["feature"] in {
        "events_w7d",
        "fatalities_w7d",
        "velocity_events_w14d",
        "spillover_w14d",
    }
    assert Path(summary["forecast_csv"]).is_file()
    assert "leaflet" in Path(summary["risk_map"]).read_text(encoding="utf-8").lower()
    text = Path(summary["summary_report"]).read_text(encoding="utf-8")
    assert "Live 14-Day Forecast Summary" in text


def test_forecast_stage_missing_model_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    _features_frame(10).to_parquet(data_dir / f"{config.FEATURES_FILE}.parquet")
    with pytest.raises(DataLoadError, match="Winning model not found"):
        forecast.forecast_stage(
            data_dir=data_dir, models_dir=models_dir, reports_dir=tmp_path / "reports"
        )


def test_forecast_stage_missing_features_raises(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    labeled = _labeled_frame(120)
    model, _ = _fit_tiny_model(labeled)
    models.save_model(model, models_dir / config.MODEL_BEST_FILE)
    with pytest.raises(DataLoadError, match="Features table not found"):
        forecast.forecast_stage(
            data_dir=tmp_path / "data", models_dir=models_dir, reports_dir=tmp_path / "reports"
        )
