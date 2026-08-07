"""Tests for ``src.models`` and ``src.pipeline`` (M8 — LightGBM training).

Covers feature resolution, imbalance math, determinism, save/load
round-trips, hand-computed metrics, error paths, and the end-to-end
``train_stage`` smoke test on synthetic split files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import models, pipeline
from src.exceptions import DataLoadError, ModelError


def _frame(n: int = 200, seed: int = 3) -> pd.DataFrame:
    """Synthetic labeled frame: 3 numeric features + meta + label."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "geo_unit": ["G1"] * n,
            "country": "India",
            "event_date": pd.date_range("2024-01-01", periods=n, freq="7D"),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.integers(0, 3, size=n).astype(float),
            config.LABEL_COLUMN: rng.integers(0, 2, size=n),
        }
    )


def _x_y(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    frame = _frame(n)
    X, y, features = models.prepare_xy(frame)
    assert features == ["f1", "f2", "f3"]
    return X, y


# ---------------------------------------------------------------------------
# Feature resolution / prepare_xy
# ---------------------------------------------------------------------------


def test_resolve_feature_columns_excludes_meta_and_label() -> None:
    frame = _frame(10)
    assert models.resolve_feature_columns(frame) == ["f1", "f2", "f3"]


def test_resolve_feature_columns_explicit_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "FEATURE_COLUMNS", ("f1", "f3"))
    assert models.resolve_feature_columns(_frame(5)) == ["f1", "f3"]


def test_prepare_xy_shapes_and_dtypes() -> None:
    X, y, features = models.prepare_xy(_frame(50))
    assert X.shape == (50, 3) and X.dtype == np.float64
    assert y.shape == (50,) and set(np.unique(y)) == {0, 1}
    assert features == ["f1", "f2", "f3"]


def test_prepare_xy_missing_label_raises() -> None:
    with pytest.raises(ModelError, match="Label column"):
        models.prepare_xy(_frame(5).drop(columns=[config.LABEL_COLUMN]))


def test_prepare_xy_non_numeric_feature_raises() -> None:
    frame = _frame(5)
    frame["f2"] = frame["f2"].astype(str)
    with pytest.raises(ModelError, match="Non-numeric"):
        models.prepare_xy(frame)


def test_prepare_xy_nan_feature_raises() -> None:
    frame = _frame(5)
    frame.loc[0, "f1"] = np.nan
    with pytest.raises(ModelError, match="NaN"):
        models.prepare_xy(frame)


def test_prepare_xy_single_class_raises() -> None:
    frame = _frame(5)
    frame[config.LABEL_COLUMN] = 1
    with pytest.raises(ModelError, match="both classes"):
        models.prepare_xy(frame)


# ---------------------------------------------------------------------------
# Imbalance handling
# ---------------------------------------------------------------------------


def test_scale_pos_weight_balanced_is_one() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    assert models.resolve_scale_pos_weight(y) == pytest.approx(1.0)


def test_scale_pos_weight_formula() -> None:
    y = np.array([0, 0, 0, 1])  # 3 neg / 1 pos
    assert models.resolve_scale_pos_weight(y) == pytest.approx(3.0)


def test_scale_pos_weight_single_class_raises() -> None:
    with pytest.raises(ModelError, match="both classes"):
        models.resolve_scale_pos_weight(np.array([0, 0, 0]))


def test_class_weight_sample_weights_balance() -> None:
    y = np.array([0, 0, 0, 1])
    weights = models.class_weight_sample_weights(y)
    # sklearn balanced weights: n / (n_classes * n_class_count)
    assert weights[0] == pytest.approx(4 / 6)  # 0.667 for the majority class
    assert weights[3] == pytest.approx(2.0)  # 2.0 for the minority class
    assert weights[:3].sum() == pytest.approx(weights[3])  # equal mass per class


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_model_fits_and_predicts_in_range() -> None:
    X, y = _x_y()
    model, effective = models.train_model(X, y)
    proba = models.predict_proba(model, X)
    assert proba.shape == (len(X),)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0
    assert effective["scale_pos_weight"] == pytest.approx(
        float(np.bincount(y)[0] / np.bincount(y)[1])
    )
    assert effective["random_state"] == config.RANDOM_SEED


def test_train_model_class_weight_method(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "IMBALANCE_METHOD", "class_weight")
    X, y = _x_y()
    model, effective = models.train_model(X, y)
    assert "scale_pos_weight" not in effective
    assert models.predict_proba(model, X).shape == (len(X),)


def test_train_deterministic_same_seed() -> None:
    X, y = _x_y()
    _, _ = models.train_model(X, y, seed=42)
    model_a, _ = models.train_model(X, y, seed=42)
    model_b, _ = models.train_model(X, y, seed=42)
    np.testing.assert_array_equal(
        models.predict_proba(model_a, X), models.predict_proba(model_b, X)
    )


def test_train_model_mismatched_lengths_raises() -> None:
    X, y = _x_y()
    with pytest.raises(ModelError, match="same number of rows"):
        models.train_model(X, y[:-1])


def test_train_model_keeps_numeric_scale_pos_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "IMBALANCE_METHOD", "scale_pos_weight")
    X, y = _x_y()
    model, effective = models.train_model(X, y, params={"scale_pos_weight": 2.5})
    assert effective["scale_pos_weight"] == pytest.approx(2.5)
    assert models.predict_proba(model, X).shape == (len(X),)


def test_train_model_unknown_imbalance_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "IMBALANCE_METHOD", "bogus")
    X, y = _x_y()
    with pytest.raises(ModelError, match="Unknown IMBALANCE_METHOD"):
        models.train_model(X, y)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip_identical_predictions(tmp_path: Path) -> None:
    X, y = _x_y()
    model, _ = models.train_model(X, y)
    path = models.save_model(model, tmp_path / "model.pkl")
    assert path.is_file()
    reloaded = models.load_model(path)
    np.testing.assert_array_equal(
        models.predict_proba(model, X), models.predict_proba(reloaded, X)
    )


def test_load_model_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        models.load_model(tmp_path / "nope.pkl")


def test_load_model_not_a_classifier_raises(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "not_a_model.pkl"
    joblib.dump({"a": 1}, path)
    with pytest.raises(ModelError, match="not a fitted classifier"):
        models.load_model(path)


def test_predict_proba_bad_shape_raises() -> None:
    class _Fake:
        def predict_proba(self, X):
            return np.ones((len(X), 1))  # single column — wrong

    with pytest.raises(ModelError, match="Unexpected predict_proba shape"):
        models.predict_proba(_Fake(), np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_binary_metrics_hand_computed() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.2])
    metrics = models.binary_metrics(y_true, y_prob, threshold=0.5)
    # predictions: 1,1,0,0,1,0 -> precision 1.0, recall 3/3, f1 1.0
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert 0.0 <= metrics["auc_pr"] <= 1.0


def test_binary_metrics_threshold_shift() -> None:
    y_true = np.array([1, 0, 0])
    y_prob = np.array([0.6, 0.7, 0.8])
    low = models.binary_metrics(y_true, y_prob, threshold=0.5)
    high = models.binary_metrics(y_true, y_prob, threshold=0.75)
    assert low["recall"] == pytest.approx(1.0)
    assert high["recall"] == pytest.approx(0.0)
    assert low["precision"] == pytest.approx(1 / 3)


def test_binary_metrics_mismatch_raises() -> None:
    with pytest.raises(ModelError, match="same length"):
        models.binary_metrics(np.array([1, 0]), np.array([0.5]))


def test_binary_metrics_empty_raises() -> None:
    with pytest.raises(ModelError, match="empty vector"):
        models.binary_metrics(np.array([]), np.array([]))


def test_binary_metrics_out_of_range_raises() -> None:
    with pytest.raises(ModelError, match=r"\[0, 1\]"):
        models.binary_metrics(np.array([1, 0]), np.array([1.5, 0.5]))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    path = models.write_manifest(tmp_path / "manifest.json", {"family": "lightgbm", "k": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"family": "lightgbm", "k": 1}


def test_json_safe_params_coerces_numpy_scalars() -> None:
    safe = pipeline._json_safe_params({"a": np.int64(3), "b": np.float64(0.5), "c": "x"})
    assert safe == {"a": 3, "b": 0.5, "c": "x"}
    assert isinstance(safe["a"], int) and isinstance(safe["b"], float)


# ---------------------------------------------------------------------------
# Pipeline train_stage
# ---------------------------------------------------------------------------


def _write_splits(tmp_path: Path) -> tuple[Path, Path]:
    """Write small synthetic split_train/val parquet files; return dirs."""
    train = _frame(150)
    val = _frame(60)
    val[config.LABEL_COLUMN] = 1 - val[config.LABEL_COLUMN]  # ensure val has both classes
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    train.to_parquet(data_dir / "split_train.parquet")
    val.to_parquet(data_dir / "split_val.parquet")
    return data_dir, models_dir


def test_train_stage_end_to_end(tmp_path: Path) -> None:
    data_dir, models_dir = _write_splits(tmp_path)
    summary = pipeline.train_stage(data_dir=data_dir, models_dir=models_dir)
    assert summary["family"] == "lightgbm"
    assert summary["n_train"] == 150 and summary["n_val"] == 60
    assert summary["n_features"] == 3
    assert 0.0 <= summary["validation_metrics"]["f1"] <= 1.0
    assert Path(summary["model_file"]).is_file()
    assert Path(summary["manifest_file"]).is_file()

    manifest = json.loads(Path(summary["manifest_file"]).read_text(encoding="utf-8"))
    assert manifest["family"] == "lightgbm"
    assert manifest["feature_columns"] == ["f1", "f2", "f3"]
    assert manifest["n_train"] == 150
    assert "validation_metrics" in manifest
    assert manifest["seed"] == config.RANDOM_SEED


def test_train_stage_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="Split file not found"):
        pipeline.train_stage(data_dir=tmp_path, models_dir=tmp_path)


# ---------------------------------------------------------------------------
# M9 — XGBoost family
# ---------------------------------------------------------------------------


def test_train_model_xgboost_fits_and_predicts_in_range() -> None:
    X, y = _x_y()
    model, effective = models.train_model(X, y, family="xgboost")
    proba = models.predict_proba(model, X)
    assert proba.shape == (len(X),)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0
    assert effective["scale_pos_weight"] == pytest.approx(
        float(np.bincount(y)[0] / np.bincount(y)[1])
    )
    assert effective["random_state"] == config.RANDOM_SEED


def test_train_model_xgboost_deterministic_same_seed() -> None:
    X, y = _x_y()
    model_a, _ = models.train_model(X, y, family="xgboost", seed=42)
    model_b, _ = models.train_model(X, y, family="xgboost", seed=42)
    np.testing.assert_array_equal(
        models.predict_proba(model_a, X), models.predict_proba(model_b, X)
    )


def test_train_model_unknown_family_raises() -> None:
    X, y = _x_y()
    with pytest.raises(ModelError, match="Unknown model family"):
        models.train_model(X, y, family="catboost")


def test_save_load_round_trip_xgboost(tmp_path: Path) -> None:
    X, y = _x_y()
    model, _ = models.train_model(X, y, family="xgboost")
    path = models.save_model(model, tmp_path / "xgb.pkl")
    reloaded = models.load_model(path)
    np.testing.assert_array_equal(
        models.predict_proba(model, X), models.predict_proba(reloaded, X)
    )


# ---------------------------------------------------------------------------
# M9 — full metric set
# ---------------------------------------------------------------------------


def test_full_metrics_hand_computed() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.2])
    m = models.full_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == [[3, 0], [0, 3]]  # [[tn, fp], [fn, tp]]
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(
        float(np.mean((y_true - y_prob) ** 2))
    )
    assert m["log_loss"] >= 0.0
    assert 0.0 <= m["auc_pr"] <= 1.0


def test_full_metrics_roc_auc_constant_scores() -> None:
    # sklearn 1.9 returns 0.5 (random-level) for constant scores; the
    # None path is reserved for a single-class y_true (ValueError).
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.full(4, 0.6)
    m = models.full_metrics(y_true, y_prob, threshold=0.5)
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(float(np.mean((y_true - 0.6) ** 2)))


def test_full_metrics_empty_raises() -> None:
    with pytest.raises(ModelError, match="empty vector"):
        models.full_metrics(np.array([]), np.array([]))


def test_full_metrics_out_of_range_raises() -> None:
    with pytest.raises(ModelError, match=r"\[0, 1\]"):
        models.full_metrics(np.array([1, 0]), np.array([1.5, 0.2]))


# ---------------------------------------------------------------------------
# M9 — threshold analysis
# ---------------------------------------------------------------------------


def test_threshold_analysis_grid_covers_config_range() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    analysis = models.threshold_analysis(y_true, y_prob)
    thresholds = [row["threshold"] for row in analysis["rows"]]
    assert thresholds[0] == pytest.approx(config.THRESHOLD_MIN)
    assert thresholds[-1] == pytest.approx(config.THRESHOLD_MAX)
    assert len(thresholds) == int(
        round((config.THRESHOLD_MAX - config.THRESHOLD_MIN) / config.THRESHOLD_STEP)
    ) + 1
    assert thresholds == sorted(thresholds)


def test_threshold_analysis_best_f1_is_argmax() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.45, 0.4, 0.35, 0.3])
    analysis = models.threshold_analysis(y_true, y_prob)
    f1s = [row["f1"] for row in analysis["rows"]]
    assert analysis["best_f1"]["f1"] == pytest.approx(max(f1s))
    # At 0.5 everything is perfect; best_F1 should also be perfect.
    assert analysis["best_f1"]["f1"] == pytest.approx(1.0)
    assert "threshold" in analysis["best_f1"]
    assert "threshold" in analysis["best_precision"]
    assert "threshold" in analysis["best_recall"]


def test_threshold_analysis_best_recall_earliest_perfect() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    analysis = models.threshold_analysis(y_true, y_prob)
    assert analysis["best_recall"]["recall"] == pytest.approx(1.0)
    assert analysis["best_precision"]["precision"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# M9 — baselines
# ---------------------------------------------------------------------------


def test_majority_baseline_predicts_majority_class() -> None:
    y_true = np.array([1, 1, 1, 0, 0])
    preds = models.majority_baseline(y_true)
    np.testing.assert_array_equal(preds, np.ones(5))


def test_majority_baseline_zero_majority() -> None:
    y_true = np.array([1, 0, 0, 0])
    preds = models.majority_baseline(y_true)
    np.testing.assert_array_equal(preds, np.zeros(4))


def test_always_positive_baseline() -> None:
    y_true = np.array([0, 1, 0])
    np.testing.assert_array_equal(models.always_positive_baseline(y_true), np.ones(3))


def _frame_with_windows(n: int = 40) -> pd.DataFrame:
    """Synthetic frame with trailing-window features for baseline tests."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "events_w14d": rng.integers(0, 10, size=n).astype(float),
            "fatalities_w14d": rng.integers(0, 10, size=n).astype(float),
            config.LABEL_COLUMN: rng.integers(0, 2, size=n),
        }
    )


def test_persistence_baseline_uses_label_thresholds() -> None:
    frame = _frame_with_windows()
    preds = models.persistence_baseline(frame)
    expected = (
        (frame["events_w14d"] >= config.ESCALATION_MIN_EVENTS)
        | (frame["fatalities_w14d"] >= config.ESCALATION_MIN_FATALITIES)
    ).to_numpy(dtype=float)
    np.testing.assert_array_equal(preds, expected)


def test_persistence_baseline_missing_column_raises() -> None:
    with pytest.raises(ModelError, match="Persistence baseline requires"):
        models.persistence_baseline(pd.DataFrame({"events_w14d": [1.0]}))


def test_event_count_heuristic_baseline() -> None:
    frame = _frame_with_windows()
    preds = models.event_count_heuristic_baseline(frame)
    expected = (frame["events_w14d"] >= config.HEURISTIC_MIN_EVENTS).to_numpy(
        dtype=float
    )
    np.testing.assert_array_equal(preds, expected)


def test_event_count_heuristic_baseline_missing_column_raises() -> None:
    with pytest.raises(ModelError, match="Event-count baseline requires"):
        models.event_count_heuristic_baseline(pd.DataFrame({"f1": [1.0]}))


def test_baselines_score_with_full_metrics() -> None:
    frame = _frame_with_windows(30)
    y = frame[config.LABEL_COLUMN].to_numpy()
    for preds in (
        models.majority_baseline(y),
        models.always_positive_baseline(y),
        models.persistence_baseline(frame),
        models.event_count_heuristic_baseline(frame),
    ):
        m = models.full_metrics(y, preds, config.DEFAULT_THRESHOLD)
        assert set(m) >= {"precision", "recall", "f1", "auc_pr", "brier"}


# ---------------------------------------------------------------------------
# M9 — winner selection
# ---------------------------------------------------------------------------


def _scores(f1: float, auc_pr: float, brier: float) -> dict[str, float]:
    return {"f1": f1, "auc_pr": auc_pr, "brier": brier}


def test_select_winner_higher_f1_wins() -> None:
    comparison = {
        "lightgbm": _scores(0.80, 0.90, 0.15),
        "xgboost": _scores(0.85, 0.89, 0.16),
    }
    winner, reason = models.select_winner(comparison)
    assert winner == "xgboost"
    assert "validation F1" in reason


def test_select_winner_f1_tie_uses_auc_pr() -> None:
    comparison = {
        "lightgbm": _scores(0.80, 0.90, 0.15),
        "xgboost": _scores(0.80, 0.93, 0.15),
    }
    winner, _ = models.select_winner(comparison)
    assert winner == "xgboost"


def test_select_winner_f1_auc_tie_uses_brier() -> None:
    comparison = {
        "lightgbm": _scores(0.80, 0.90, 0.15),
        "xgboost": _scores(0.80, 0.90, 0.12),
    }
    winner, _ = models.select_winner(comparison)
    assert winner == "xgboost"


def test_select_winner_effective_tie_uses_simplicity_order() -> None:
    comparison = {
        "lightgbm": _scores(0.80, 0.90, 0.15),
        "xgboost": _scores(0.80 + 1e-6, 0.90, 0.15),  # within MODEL_TIE_EPSILON
    }
    winner, reason = models.select_winner(comparison)
    assert winner == "lightgbm"
    assert "simpler" in reason


def test_select_winner_requires_two_families() -> None:
    with pytest.raises(ModelError, match="at least two"):
        models.select_winner({"lightgbm": _scores(0.8, 0.9, 0.15)})


# ---------------------------------------------------------------------------
# M9 — compare_stage end to end
# ---------------------------------------------------------------------------


def _frame_for_compare(n: int = 150) -> pd.DataFrame:
    """Synthetic frame with features + windows needed by baselines."""
    frame = _frame(n)
    rng = np.random.default_rng(11)
    frame["events_w14d"] = rng.integers(0, 10, size=n).astype(float)
    frame["fatalities_w14d"] = rng.integers(0, 10, size=n).astype(float)
    return frame


def _write_compare_splits(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write synthetic splits; return (data_dir, models_dir, reports_dir)."""
    train = _frame_for_compare(150)
    val = _frame_for_compare(60)
    val[config.LABEL_COLUMN] = 1 - val[config.LABEL_COLUMN]  # both classes in val
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    train.to_parquet(data_dir / "split_train.parquet")
    val.to_parquet(data_dir / "split_val.parquet")
    return data_dir, models_dir, reports_dir


def test_compare_stage_end_to_end(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_compare_splits(tmp_path)
    pipeline.train_stage(data_dir=data_dir, models_dir=models_dir)  # M8 artifact
    summary = pipeline.compare_stage(
        data_dir=data_dir, models_dir=models_dir, reports_dir=reports_dir
    )
    assert summary["winner"] in {"lightgbm", "xgboost"}
    assert 0.0 < summary["operating_threshold"] < 1.0
    assert summary["n_features"] >= 3
    assert Path(summary["best_model_file"]).is_file()
    assert Path(summary["comparison_json"]).is_file()
    assert Path(summary["report"]).is_file()
    assert (models_dir / config.MODEL_XGB_FILE).is_file()

    document = json.loads(Path(summary["comparison_json"]).read_text(encoding="utf-8"))
    assert document["winner"] == summary["winner"]
    assert set(document["baselines"]) == {
        "majority",
        "always_positive",
        "persistence",
        "event_count_heuristic",
    }
    assert "confusion_matrix" in document["lightgbm"]["metrics_at_0_5"]
    assert "confusion_matrix" in document["xgboost"]["metrics_at_0_5"]
    assert {"precision", "recall", "f1"} <= set(
        document["winner_metrics_at_operating"]
    )

    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "## Winner" in report
    assert "## Baselines" in report
    assert summary["winner"] in report


def test_compare_stage_missing_lgbm_raises(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_compare_splits(tmp_path)
    with pytest.raises(DataLoadError, match="run the 'train' stage"):
        pipeline.compare_stage(
            data_dir=data_dir, models_dir=models_dir, reports_dir=reports_dir
        )


def test_compare_stage_detects_lgbm_drift(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_compare_splits(tmp_path)
    pipeline.train_stage(data_dir=data_dir, models_dir=models_dir)
    manifest_path = models_dir / config.MODEL_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_metrics"]["f1"] = manifest["validation_metrics"]["f1"] + 0.1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="drifted from manifest"):
        pipeline.compare_stage(
            data_dir=data_dir, models_dir=models_dir, reports_dir=reports_dir
        )


def test_compare_stage_never_touches_test_split(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_compare_splits(tmp_path)
    pipeline.train_stage(data_dir=data_dir, models_dir=models_dir)
    # A poisoned test split (wrong label) must not affect the comparison:
    # winner selection uses train+val only.
    test = _frame_for_compare(20)
    test[config.LABEL_COLUMN] = 1 - test[config.LABEL_COLUMN]
    test.to_parquet(data_dir / "split_test.parquet")
    summary = pipeline.compare_stage(
        data_dir=data_dir, models_dir=models_dir, reports_dir=reports_dir
    )
    assert summary["winner"] in {"lightgbm", "xgboost"}
    document = json.loads(
        Path(summary["comparison_json"]).read_text(encoding="utf-8")
    )
    assert "test" not in document["splits"]  # test split never entered


def test_write_model_comparison_report_has_rationale(tmp_path: Path) -> None:
    data_dir, models_dir, reports_dir = _write_compare_splits(tmp_path)
    pipeline.train_stage(data_dir=data_dir, models_dir=models_dir)
    summary = pipeline.compare_stage(
        data_dir=data_dir, models_dir=models_dir, reports_dir=reports_dir
    )
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "Why this threshold" in report
    assert "majority-positive" in report
    assert "best-F1 threshold" in report
