"""Tests for ``src.explainability`` (M10 — SHAP explainability).

Covers SHAP value computation/shape checks, mean-|SHAP| ranking, the
representative-row selector (correct pos/neg/borderline), PNG artifact
validity, the markdown summary contents, and the end-to-end ``explain_stage``
on a small trained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import explainability, models
from src.exceptions import DataLoadError, ExplainabilityError


def _frame(n: int = 120, seed: int = 5) -> pd.DataFrame:
    """Synthetic labeled frame: 4 numeric features + meta + label."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "geo_unit": ["G1"] * n,
            "country": "India",
            "admin1": "Bihar",
            "event_date": pd.date_range("2025-01-01", periods=n, freq="7D"),
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
    """Write split_test + escalation_best.pkl + comparison json; return dirs."""
    frame = _frame(120)
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()
    model, _ = _fit_tiny_model(frame)
    models.save_model(model, models_dir / config.MODEL_BEST_FILE)
    frame.to_parquet(data_dir / "split_test.parquet")
    (models_dir / config.MODEL_COMPARISON_FILE).write_text(
        json.dumps({"operating_threshold": 0.30}), encoding="utf-8"
    )
    return data_dir, models_dir, reports_dir


# ---------------------------------------------------------------------------
# Loading / SHAP computation
# ---------------------------------------------------------------------------


def test_load_winning_model_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="Winning model not found"):
        explainability.load_winning_model(tmp_path)


def test_load_test_window_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="Test split not found"):
        explainability.load_test_window(tmp_path)


def test_compute_shap_values_shape_matches_features() -> None:
    frame = _frame(80)
    model, features = _fit_tiny_model(frame)
    X, _, _ = models.prepare_xy(frame)
    _, values, base = explainability.compute_shap_values(model, X, features)
    assert values.shape == (len(X), len(features))
    assert isinstance(base, float)
    assert np.isfinite(values).all()


def test_compute_shap_values_feature_mismatch_raises() -> None:
    frame = _frame(60)
    model, _ = _fit_tiny_model(frame)
    X, _, _ = models.prepare_xy(frame)
    with pytest.raises(ExplainabilityError, match="Feature count mismatch"):
        explainability.compute_shap_values(model, X, ["only_one"])


def test_mean_abs_importance_ranked_descending() -> None:
    rng = np.random.default_rng(1)
    values = np.abs(rng.normal(size=(50, 4))) * np.array([1, 4, 2, 3])
    imp = explainability.mean_abs_importance(values, ["a", "b", "c", "d"])
    assert list(imp["feature"]) == ["b", "d", "c", "a"]
    assert list(imp["rank"]) == [1, 2, 3, 4]
    assert imp["share"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------


def test_select_representative_rows_categories() -> None:
    proba = np.array([0.9, 0.8, 0.3, 0.2, 0.31, 0.29, 0.1])
    y_true = np.array([1, 1, 1, 0, 0, 1, 0])
    selected = explainability.select_representative_rows(proba, y_true, 0.3, 2)
    # pos: correct positives by confidence -> idx 0 (0.9), idx 1 (0.8)
    assert list(selected["pos"]) == [0, 1]
    # neg: correct negatives by lowest proba -> idx 6 (0.1), idx 3 (0.2)
    assert list(selected["neg"]) == [6, 3]
    # border: closest to 0.3 -> idx 2 (exact 0.3), then idx 4 (0.31) / 5 (0.29)
    assert list(selected["border"]) == [2, 4]


def test_select_representative_rows_missing_categories() -> None:
    proba = np.array([0.1, 0.2, 0.1])
    y_true = np.array([0, 0, 0])  # no positives at all
    selected = explainability.select_representative_rows(proba, y_true, 0.5, 2)
    assert len(selected["pos"]) == 0
    assert len(selected["neg"]) == 2
    assert len(selected["border"]) == 2


# ---------------------------------------------------------------------------
# Feature interpretation
# ---------------------------------------------------------------------------


def test_interpret_feature_patterns() -> None:
    assert "7-day" in explainability._interpret_feature("events_w7d")
    assert "Log1p" in explainability._interpret_feature("events_log1p_w30d")
    assert "velocity" in explainability._interpret_feature("velocity_events_w14d")
    assert "spillover" in explainability._interpret_feature("spillover_w14d")
    assert "Days since" in explainability._interpret_feature("days_since_event")
    assert "999" in explainability._interpret_feature("days_since_event")
    assert "feature_summary" in explainability._interpret_feature("unknown_feature")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _png_ok(path: Path) -> bool:
    """True when the file exists, is non-empty, and has the PNG magic bytes."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_explain_stage_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SHAP_SAMPLE_CAP", 60)
    monkeypatch.setattr(config, "SHAP_WATERFALL_COUNT", 2)
    monkeypatch.setattr(config, "SHAP_DEPENDENCE_TOP_K", 4)
    monkeypatch.setattr(config, "SHAP_TOP_N", 5)
    monkeypatch.setattr(config, "SHAP_MAX_DISPLAY", 6)
    data_dir, models_dir, reports_dir = _write_env(tmp_path)

    summary = explainability.explain_stage(
        data_dir=data_dir, models_dir=models_dir, reports_dir=reports_dir
    )

    # Summary structure
    assert summary["n_features"] == 4
    assert summary["n_explained_rows"] == 60  # cap applied
    assert summary["operating_threshold"] == pytest.approx(0.30)
    assert len(summary["top_features"]) == 4  # min(SHAP_TOP_N, n_features)
    assert summary["top_features"][0]["rank"] == 1

    # Every PNG is valid
    assert summary["plots"], "no plots generated"
    for p in summary["plots"]:
        assert _png_ok(Path(p)), f"corrupt/empty PNG: {p}"

    # Required plot families present; waterfalls match the categories that
    # actually had representative rows (a tiny model may have no correct
    # negatives under the threshold).
    names = {Path(p).name for p in summary["plots"]}
    assert "summary_plot.png" in names
    assert "bar_plot.png" in names
    waterfalls = [n for n in names if n.startswith("waterfall_")]
    expected_waterfalls = 2 * sum(1 for v in summary["representatives"].values() if v)
    assert len(waterfalls) == expected_waterfalls
    assert summary["representatives"]["pos"] >= 1
    dependencies = [n for n in names if n.startswith("dependence_")]
    assert len(dependencies) == 4

    # Summary report contents
    report = Path(summary["summary_report"]).read_text(encoding="utf-8")
    assert "## Top 5 features by mean |SHAP|" in report
    assert "## Most influential risk drivers" in report
    assert "## Model behaviour observations" in report
    assert "correctly predicted POSITIVE" in report
    assert "correctly predicted NEGATIVE" in report
    assert "difficult / borderline" in report
    assert "interpretation" in report or "Log1p" in report


def test_explain_stage_missing_model_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    frame = _frame(10)
    frame.to_parquet(data_dir / "split_test.parquet")
    with pytest.raises(DataLoadError, match="Winning model not found"):
        explainability.explain_stage(
            data_dir=data_dir, models_dir=models_dir, reports_dir=tmp_path / "reports"
        )


def test_explain_stage_missing_split_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    frame = _frame(10)
    model, _ = _fit_tiny_model(frame)
    models.save_model(model, models_dir / config.MODEL_BEST_FILE)
    with pytest.raises(DataLoadError, match="Test split not found"):
        explainability.explain_stage(
            data_dir=data_dir, models_dir=models_dir, reports_dir=tmp_path / "reports"
        )


def test_operating_threshold_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "DEFAULT_THRESHOLD", 0.42)
    assert explainability._operating_threshold(tmp_path) == pytest.approx(0.42)
