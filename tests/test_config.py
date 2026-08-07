"""Tests for ``config``."""

from __future__ import annotations

import pytest

import config
from src.exceptions import ConfigurationError


def test_paths_resolve_inside_repo() -> None:
    assert config.DATA_RAW_DIR.name == "raw"
    assert config.DATA_PROCESSED_DIR.name == "processed"
    assert config.LOGS_DIR.name == "logs"
    assert config.MODELS_DIR.name == "models"
    assert config.REPO_ROOT.is_dir()


def test_scope_constants() -> None:
    assert "India" in config.COUNTRIES
    assert "Pakistan" in config.COUNTRIES
    assert "Afghanistan" in config.COUNTRIES
    assert "Myanmar" in config.COUNTRIES
    assert "Sudan" in config.COUNTRIES
    assert "South Sudan" in config.COUNTRIES
    assert config.LABEL_HORIZON_DAYS == 14
    assert config.ROLLING_WINDOWS == (7, 14, 30)
    assert config.LABELED_FEATURES_FILE == "labeled_features"
    assert config.LABEL_COLUMN == "escalation"
    assert sum(config.SPLIT_RATIOS.values()) == pytest.approx(1.0)
    assert set(config.SPLIT_RATIOS) == {"train", "val", "test"}
    assert config.SPLIT_DATE_COLUMN == "event_date"
    assert config.SPLIT_FILE_PREFIX == "split"
    assert config.LABEL_COLUMN not in config.META_COLUMNS
    assert not set(config.FEATURE_COLUMNS) & set(config.META_COLUMNS)
    assert config.DEFAULT_THRESHOLD == 0.5
    assert config.MODEL_COMPARISON_FILE == "model_comparison.json"
    assert config.MODEL_COMPARISON_REPORT == "model_comparison.md"
    assert config.THRESHOLD_MIN < config.THRESHOLD_MAX
    assert config.THRESHOLD_STEP > 0
    assert config.MODEL_TIE_EPSILON > 0
    assert config.HEURISTIC_MIN_EVENTS >= 1
    assert set(config.MODEL_SIMPLICITY_ORDER) == {"lightgbm", "xgboost"}
    assert config.SHAP_SAMPLE_CAP >= 1
    assert config.SHAP_TOP_N >= 1
    assert config.SHAP_DEPENDENCE_TOP_K >= 1
    assert config.SHAP_WATERFALL_COUNT >= 1
    assert config.SHAP_REPORT_DIR.name == "shap"
    assert config.SHAP_SUMMARY_FILE == "shap_summary.md"


def test_validate_config_passes() -> None:
    config.validate_config()  # must not raise


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("COUNTRIES", ()),
        ("COUNTRIES_MODE", "bogus"),
        ("DUPLICATES_MODE", "bogus"),
        ("MAX_DROPPED_FRACTION", 0.0),
        ("LABEL_HORIZON_DAYS", 0),
        ("ESCALATION_MULTIPLIER", 1.0),
        ("IMBALANCE_METHOD", "bogus"),
        ("OPERATING_THRESHOLD_MODE", "bogus"),
        ("SPLIT_RATIOS", {"train": 0.5, "val": 0.2, "test": 0.1}),
        ("ROLLING_WINDOWS", (30, 7)),
        ("INCOMPLETE_WINDOW", "bogus"),
        ("TRAILING_MEDIAN_WINDOW_DAYS", 0),
        ("ESCALATION_MIN_FATALITIES", 0),
        ("LGBM_PARAMS", {"scale_pos_weight": "bogus"}),
        ("THRESHOLD_MIN", 0.95),  # must stay below THRESHOLD_MAX (0.90)
        ("THRESHOLD_STEP", 0.0),
        ("THRESHOLD_MAX", 0.87),  # 0.77/0.05 not a whole number -> sweep miss
        ("MODEL_TIE_EPSILON", 0.0),
        ("HEURISTIC_MIN_EVENTS", 0),
        ("MODEL_SIMPLICITY_ORDER", ("lightgbm",)),
        ("SHAP_SAMPLE_CAP", 0),
        ("SHAP_TOP_N", 0),
        ("SHAP_DEPENDENCE_TOP_K", 0),
        ("SHAP_WATERFALL_COUNT", 0),
    ],
)
def test_validate_config_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch, attr: str, value: object
) -> None:
    monkeypatch.setattr(config, attr, value)
    with pytest.raises(ConfigurationError):
        config.validate_config()


def test_validate_config_rejects_inverted_coords(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LAT_MIN", 10.0)
    monkeypatch.setattr(config, "LAT_MAX", -10.0)
    with pytest.raises(ConfigurationError):
        config.validate_config()


def test_validate_config_rejects_label_in_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "META_COLUMNS", ("geo_unit", "escalation"))
    with pytest.raises(ConfigurationError):
        config.validate_config()


def test_validate_config_rejects_feature_meta_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "FEATURE_COLUMNS", ("geo_unit", "events_w7d"))
    with pytest.raises(ConfigurationError):
        config.validate_config()


def test_validate_config_rejects_bad_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DEFAULT_THRESHOLD", 1.5)
    with pytest.raises(ConfigurationError):
        config.validate_config()
