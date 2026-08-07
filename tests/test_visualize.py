"""Tests for ``src.visualization`` (M11 — risk map & visualization).

Covers risk-category banding, the latest-per-geo-unit prediction table with
SHAP drivers, SHAP/prediction consistency, the folium risk-map HTML, the
country dashboard (matplotlib PNG + plotly HTML), hotspot ranking/bar/heatmap,
temporal trend plots, importance/category charts, prediction distribution,
the risk summary report, and the end-to-end ``visualize_stage`` on a small
trained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import models, visualization
from src.exceptions import DataLoadError, VisualizationError


def _frame(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic labeled frame: features + meta + label (two geo units)."""
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
    """Train a tiny deterministic LGBM on the frame; return (model, features)."""
    X, y, features = models.prepare_xy(frame)
    model, _ = models.train_model(
        X, y, params={"n_estimators": 30, "num_leaves": 7, "verbosity": -1}
    )
    return model, features


def _write_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write split_test + escalation_best.pkl + comparison json + centroids."""
    frame = _frame(120)
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()
    model, _ = _fit_tiny_model(frame)
    models.save_model(model, models_dir / config.MODEL_BEST_FILE)
    frame.to_parquet(data_dir / "split_test.parquet")
    cleaned = frame[["geo_unit", "event_date", "events_w7d", "fatalities_w7d"]].copy()
    cleaned["latitude"] = [25.5] * 120
    cleaned["longitude"] = [85.0] * 120
    cleaned.to_parquet(data_dir / f"{config.CLEANED_EVENTS_FILE}.parquet")
    (models_dir / config.MODEL_COMPARISON_FILE).write_text(
        json.dumps({"operating_threshold": 0.30}), encoding="utf-8"
    )
    return data_dir, models_dir, reports_dir


def _png_ok(path: Path) -> bool:
    """True when the file exists, is non-empty, and has PNG magic bytes."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------


def test_risk_category_bands() -> None:
    bounds = config.RISK_LEVEL_BOUNDARIES
    assert visualization.risk_category(0.0) == "Low"
    assert visualization.risk_category(bounds[0] - 0.01) == "Low"
    assert visualization.risk_category(bounds[0]) == "Medium"
    assert visualization.risk_category(bounds[1] - 0.01) == "Medium"
    assert visualization.risk_category(bounds[1]) == "High"
    assert visualization.risk_category(bounds[2] - 0.01) == "High"
    assert visualization.risk_category(bounds[2]) == "Critical"
    assert visualization.risk_category(1.0) == "Critical"


def test_risk_category_out_of_range_raises() -> None:
    with pytest.raises(VisualizationError, match="out of range"):
        visualization.risk_category(1.5)


# ---------------------------------------------------------------------------
# Prediction table / SHAP drivers
# ---------------------------------------------------------------------------


def test_top_drivers_format() -> None:
    values = np.array([0.1, -0.7, 0.3])
    drivers = visualization._top_drivers(values, ["a", "b", "c"], 2)
    assert drivers == ["b: -0.700", "c: +0.300"]


def test_build_prediction_table_latest_per_unit() -> None:
    frame = _frame(120)
    model, features = _fit_tiny_model(frame)
    X, _, _ = models.prepare_xy(frame)
    proba = models.predict_proba(model, X)
    _, shap_values, _ = visualization.compute_shap_values(model, X, features)
    centroids = pd.DataFrame(
        {"geo_unit": ["G1", "G2"], "lat": [25.5, 24.0], "lon": [85.0, 68.0]}
    )
    table = visualization.build_prediction_table(
        frame, proba, shap_values, features, 0.3, centroids
    )
    assert len(table) == 2
    assert set(table["geo_unit"]) == {"G1", "G2"}
    assert table[config.PREDICTION_CATEGORY_COLUMN].isin(
        config.RISK_LEVEL_NAMES
    ).all()
    assert table["top_drivers"].str.contains(": ").all()
    assert table["lat"].tolist() == pytest.approx([25.5, 24.0])


def test_validate_shap_consistency_passes() -> None:
    frame = _frame(80)
    model, features = _fit_tiny_model(frame)
    X, _, _ = models.prepare_xy(frame)
    proba = models.predict_proba(model, X)
    _, shap_values, base = visualization.compute_shap_values(model, X, features)
    visualization._validate_shap_consistency(shap_values, proba, base)  # no raise


# ---------------------------------------------------------------------------
# Risk map
# ---------------------------------------------------------------------------


def test_save_risk_map_html(tmp_path: Path) -> None:
    frame = _frame(60)
    model, features = _fit_tiny_model(frame)
    X, _, _ = models.prepare_xy(frame)
    proba = models.predict_proba(model, X)
    _, shap_values, _ = visualization.compute_shap_values(model, X, features)
    centroids = pd.DataFrame({"geo_unit": ["G1", "G2"], "lat": [25.5, 24.0], "lon": [85.0, 68.0]})
    table = visualization.build_prediction_table(
        frame, proba, shap_values, features, 0.3, centroids
    )
    path = visualization.save_risk_map(table, tmp_path / "maps")
    html = path.read_text(encoding="utf-8")
    assert "leaflet" in html.lower()
    assert "G1" in html and "G2" in html
    assert "Risk probability" in html


# ---------------------------------------------------------------------------
# Country dashboard
# ---------------------------------------------------------------------------


def test_country_metrics_aggregates() -> None:
    table = pd.DataFrame(
        {
            "country": ["India", "India", "Pakistan"],
            config.PREDICTION_PROBA_COLUMN: [0.8, 0.6, 0.2],
            config.PREDICTION_CLASS_COLUMN: [1, 1, 0],
            config.RECENT_EVENTS_COLUMN: [10.0, 20.0, 5.0],
            config.RECENT_FATALITIES_COLUMN: [2.0, 4.0, 1.0],
        }
    )
    metrics = visualization._country_metrics(table)
    india = metrics[metrics["country"] == "India"].iloc[0]
    assert india["avg_risk"] == pytest.approx(0.7)
    assert india["positive_rate"] == pytest.approx(1.0)
    assert india["mean_events"] == pytest.approx(15.0)
    assert india["mean_fatalities"] == pytest.approx(3.0)


def test_save_country_dashboard(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "country": ["India", "Pakistan"],
            config.PREDICTION_PROBA_COLUMN: [0.7, 0.2],
            config.PREDICTION_CLASS_COLUMN: [1, 0],
            config.RECENT_EVENTS_COLUMN: [10.0, 5.0],
            config.RECENT_FATALITIES_COLUMN: [3.0, 1.0],
        }
    )
    png, html = visualization.save_country_dashboard(
        table, tmp_path / "figures", tmp_path / "dashboard"
    )
    assert _png_ok(png)
    assert "plotly" in html.read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------------
# Hotspot analysis
# ---------------------------------------------------------------------------


def test_save_hotspot_analysis(tmp_path: Path) -> None:
    frame = _frame(120)
    model, features = _fit_tiny_model(frame)
    X, _, _ = models.prepare_xy(frame)
    proba = models.predict_proba(model, X)
    _, shap_values, _ = visualization.compute_shap_values(model, X, features)
    centroids = pd.DataFrame({"geo_unit": ["G1", "G2"], "lat": [25.5, 24.0], "lon": [85.0, 68.0]})
    table = visualization.build_prediction_table(
        frame, proba, shap_values, features, 0.3, centroids
    )
    weekly = frame[["geo_unit", "event_date"]].copy()
    weekly[config.PREDICTION_PROBA_COLUMN] = proba
    csv_path, bar_path, heat_path = visualization.save_hotspot_analysis(
        table, weekly, tmp_path / "reports", tmp_path / "figures"
    )
    ranking = pd.read_csv(csv_path)
    assert len(ranking) == 2
    assert list(ranking[config.PREDICTION_PROBA_COLUMN]) == sorted(
        ranking[config.PREDICTION_PROBA_COLUMN], reverse=True
    )
    assert "rank" in ranking.columns
    assert _png_ok(bar_path)
    assert _png_ok(heat_path)


# ---------------------------------------------------------------------------
# Temporal trends
# ---------------------------------------------------------------------------


def test_save_temporal_trends(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "geo_unit": ["G1"] * 40,
            "country": "India",
            "event_date": pd.date_range("2025-01-01", periods=40, freq="7D"),
            config.PREDICTION_PROBA_COLUMN: rng.random(40),
        }
    )
    paths = visualization.save_temporal_trends(frame, tmp_path / "figures")
    assert len(paths) == 4
    for p in paths:
        assert _png_ok(p)


# ---------------------------------------------------------------------------
# Feature importance / category contribution
# ---------------------------------------------------------------------------


def test_feature_family_mapping() -> None:
    assert visualization._feature_family("events_w30d") == "volume"
    assert visualization._feature_family("fatalities_log1p_w14d") == "lethality"
    assert visualization._feature_family("velocity_events_w7d") == "velocity"
    assert visualization._feature_family("fat_std_w14d") == "volatility"
    assert visualization._feature_family("persistence_w7d") == "persistence"
    assert visualization._feature_family("days_since_event") == "recency"
    assert visualization._feature_family("spillover_w14d") == "spillover"
    assert visualization._feature_family("admin1_code") == "identity"
    assert visualization._feature_family("month") == "calendar"


def test_category_contribution_sums() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["events_w7d", "fatalities_w7d", "velocity_events_w14d", "month"],
            "mean_abs_shap": [0.5, 0.3, 0.1, 0.1],
        }
    )
    families = visualization._category_contribution(importance)
    assert families["mean_abs_shap"].sum() == pytest.approx(1.0)
    assert set(families["family"]) == {"volume", "lethality", "velocity", "calendar"}


def test_save_importance_dashboard(tmp_path: Path) -> None:
    importance = pd.DataFrame(
        {"feature": ["events_w7d", "fatalities_w7d"], "mean_abs_shap": [0.6, 0.4]}
    )
    bar, cat = visualization.save_importance_dashboard(importance, tmp_path / "figures")
    assert _png_ok(bar)
    assert _png_ok(cat)


# ---------------------------------------------------------------------------
# Prediction distribution / risk summary
# ---------------------------------------------------------------------------


def test_save_prediction_distribution(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            config.PREDICTION_PROBA_COLUMN: np.linspace(0, 1, 50),
            config.PREDICTION_CATEGORY_COLUMN: ["Low"] * 25 + ["Critical"] * 25,
        }
    )
    dist, cat = visualization.save_prediction_distribution(table, tmp_path / "figures")
    assert _png_ok(dist)
    assert _png_ok(cat)


def test_write_risk_summary_contents(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "geo_unit": ["G1", "G2"],
            "country": ["India", "Pakistan"],
            "event_date": pd.to_datetime(["2025-06-01", "2025-06-01"]),
            config.PREDICTION_PROBA_COLUMN: [0.9, 0.1],
            config.PREDICTION_CLASS_COLUMN: [1, 0],
            config.PREDICTION_CATEGORY_COLUMN: ["Critical", "Low"],
        }
    )
    importance = pd.DataFrame(
        {"feature": ["events_w7d", "fatalities_w7d"], "mean_abs_shap": [0.6, 0.4]}
    )
    metrics = visualization._country_metrics(
        pd.DataFrame(
            {
                "country": ["India", "Pakistan"],
                config.PREDICTION_PROBA_COLUMN: [0.9, 0.1],
                config.PREDICTION_CLASS_COLUMN: [1, 0],
                config.RECENT_EVENTS_COLUMN: [10.0, 5.0],
                config.RECENT_FATALITIES_COLUMN: [3.0, 1.0],
            }
        )
    )
    path = visualization.write_risk_summary(table, importance, metrics, 0.25, tmp_path / "risk_summary.md")
    text = path.read_text(encoding="utf-8")
    for section in (
        "Highest-risk regions",
        "Safest regions",
        "Average risk by country",
        "Top SHAP drivers",
        "Interpretation",
        "Important observations",
    ):
        assert section in text
    assert "G1" in text and "events_w7d" in text


# ---------------------------------------------------------------------------
# End-to-end stage
# ---------------------------------------------------------------------------


def test_visualize_stage_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "HOTSPOT_TOP_K", 2)
    monkeypatch.setattr(config, "HOTSPOT_HEATMAP_WEEKS", 4)
    data_dir, models_dir, reports_dir = _write_env(tmp_path)

    summary = visualization.visualize_stage(
        data_dir=data_dir, models_dir=models_dir, reports_dir=reports_dir
    )

    assert summary["geo_units"] == 2
    assert summary["countries"] == 2
    assert summary["operating_threshold"] == pytest.approx(0.30)
    assert set(summary["risk_categories"]) <= set(config.RISK_LEVEL_NAMES)
    assert summary["top_driver"]["feature"] in {"events_w7d", "fatalities_w7d", "velocity_events_w14d", "spillover_w14d"}

    map_html = Path(summary["risk_map"]).read_text(encoding="utf-8")
    assert "leaflet" in map_html.lower()

    for key in (
        "dashboard_png",
        "hotspot_bar",
        "hotspot_heatmap",
        "temporal_plots",
        "importance_plots",
        "distribution_plots",
    ):
        value = summary[key]
        paths = value if isinstance(value, list) else [value]
        assert paths, f"no artifacts for {key}"
        for p in paths:
            assert _png_ok(Path(p)), f"corrupt/empty PNG: {p}"

    assert "plotly" in Path(summary["dashboard_html"]).read_text(encoding="utf-8").lower()
    ranking = pd.read_csv(summary["hotspot_ranking"])
    assert "geo_unit" in ranking.columns
    text = Path(summary["summary_report"]).read_text(encoding="utf-8")
    assert "Highest-risk regions" in text


def test_visualize_stage_missing_model_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    frame = _frame(10)
    frame.to_parquet(data_dir / "split_test.parquet")
    with pytest.raises(DataLoadError, match="Winning model not found"):
        visualization.visualize_stage(
            data_dir=data_dir, models_dir=models_dir, reports_dir=tmp_path / "reports"
        )
