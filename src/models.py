"""Model training wrappers (PRD §11.4, plan M8/M9).

Provides the LightGBM and XGBoost training paths used across the project:
deterministic train/save/load/predict with configurable class-imbalance
handling (``scale_pos_weight`` — the PRD default — or per-sample
``class_weight``), the compact binary-metric helpers, the full comparison
metric set (ROC-AUC, Brier, log loss, confusion matrix), the threshold
sweep used to pick the operating point, the four PRD baselines, and the
PRD-priority winner selector for the LightGBM vs XGBoost head-to-head.

Leakage guarantees upstream (labels future-only, splits strictly
chronological) are assumed here; this module never touches dates or splits —
it only consumes numeric feature matrices and 0/1 label vectors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

import config
from src.exceptions import ModelError

logger = logging.getLogger(__name__)

# sklearn's LGBMClassifier returns [P(0), P(1)]; positive class is index 1.
_POSITIVE_INDEX = 1


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model feature columns for a frame.

    Uses ``config.FEATURE_COLUMNS`` when non-empty, otherwise derives them
    as every column except ``config.META_COLUMNS`` and the label column.

    Raises:
        ModelError: if no features remain, or a feature is missing/non-numeric.
    """
    label = config.LABEL_COLUMN
    if config.FEATURE_COLUMNS:
        features = list(config.FEATURE_COLUMNS)
    else:
        features = [
            col
            for col in df.columns
            if col not in config.META_COLUMNS and col != label
        ]
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ModelError(f"Feature columns missing from frame: {missing}.")
    if not features:
        raise ModelError("No feature columns available for training.")
    non_numeric = [
        col for col in features if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ModelError(f"Non-numeric feature columns: {non_numeric}.")
    # Note: geo_unit_code and admin1_code are identical on this dataset
    # (geo_unit == admin1); harmless for tree models, kept for generality.
    return features


def prepare_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix and label vector from a training frame.

    Args:
        df: Frame containing the features (see :func:`resolve_feature_columns`)
            and ``config.LABEL_COLUMN``.

    Returns:
        ``(X, y, features)`` with X as float64, y as int64, and the feature
        column names in training order.

    Raises:
        ModelError: on missing columns, non-numeric features, NaN features,
            or an invalid label column.
    """
    label = config.LABEL_COLUMN
    if label not in df.columns:
        raise ModelError(f"Label column {label!r} missing from frame.")
    features = resolve_feature_columns(df)
    X = df[features].to_numpy(dtype="float64")
    if not np.isfinite(X).all():
        raise ModelError("Feature matrix contains NaN or infinite values.")
    y = df[label].to_numpy(dtype="int64")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ModelError("Label vector must contain only 0 and 1.")
    if len(np.unique(y)) < 2:
        raise ModelError("Label vector must contain both classes.")
    return X, y, features


def resolve_scale_pos_weight(y: np.ndarray) -> float:
    """Compute LightGBM's ``scale_pos_weight`` from the class counts.

    Uses the formula recommended in the LightGBM docs:
    ``n_negative / n_positive``. Because the escalation label is
    majority-positive here (≈69%), the factor is < 1 — it counterbalances the
    label skew rather than amplifying it (documented in PROGRESS.md).

    Raises:
        ModelError: if either class is absent.
    """
    counts = np.bincount(y, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        raise ModelError("Cannot compute scale_pos_weight without both classes.")
    return float(counts[0]) / float(counts[1])


def class_weight_sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights from sklearn's balanced class weights.

    Each sample is weighted inversely to its class frequency, so both
    classes contribute equally to the loss.

    Raises:
        ModelError: if either class is absent.
    """
    if len(np.unique(y)) < 2:
        raise ModelError("Cannot compute class weights without both classes.")
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return weights[y].astype("float64")


def _resolve_imbalance(
    y: np.ndarray, params: dict[str, object]
) -> tuple[dict[str, object], np.ndarray | None]:
    """Materialize the configured imbalance handling into fit arguments.

    Returns ``(params, sample_weight)``. For ``scale_pos_weight`` the
    placeholder ``"auto"`` in ``params`` is replaced by the computed ratio;
    for ``class_weight`` a per-sample weight vector is returned instead.

    Raises:
        ModelError: for an unknown ``IMBALANCE_METHOD``.
    """
    method = config.IMBALANCE_METHOD
    if method == "scale_pos_weight":
        params = dict(params)
        if params.get("scale_pos_weight") == "auto":
            params["scale_pos_weight"] = resolve_scale_pos_weight(y)
        return params, None
    if method == "class_weight":
        params = dict(params)
        params.pop("scale_pos_weight", None)
        return params, class_weight_sample_weights(y)
    raise ModelError(f"Unknown IMBALANCE_METHOD: {method!r}.")


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, object] | None = None,
    seed: int | None = None,
    family: str = "lightgbm",
) -> tuple[Any, dict[str, object]]:
    """Train a deterministic LightGBM or XGBoost classifier.

    Args:
        X: Float feature matrix (from :func:`prepare_xy`).
        y: 0/1 label vector.
        params: Overrides of the family's ``config`` params (``LGBM_PARAMS``
            or ``XGB_PARAMS``); ``scale_pos_weight`` may be the placeholder
            ``"auto"``.
        seed: Random seed; defaults to ``config.RANDOM_SEED``.
        family: ``"lightgbm"`` or ``"xgboost"``.

    Returns:
        ``(model, effective_params)`` — the fitted classifier and the exact
        parameters used (with imbalance resolved), for logging and the
        manifest.

    Raises:
        ModelError: on invalid inputs, an unknown family, or a training
            failure.
    """
    if X.shape[0] != y.shape[0]:
        raise ModelError("X and y must have the same number of rows.")
    if X.shape[0] == 0:
        raise ModelError("Cannot train on an empty matrix.")
    if family not in ("lightgbm", "xgboost"):
        raise ModelError(f"Unknown model family: {family!r}.")
    seed = seed if seed is not None else config.RANDOM_SEED
    base = dict(config.XGB_PARAMS if family == "xgboost" else config.LGBM_PARAMS)
    base.update(params or {})
    base["random_state"] = seed
    effective, sample_weight = _resolve_imbalance(y, base)
    classifier = XGBClassifier if family == "xgboost" else LGBMClassifier
    try:
        model = classifier(**effective)
        model.fit(X, y, sample_weight=sample_weight)
    except Exception as exc:  # pragma: no cover - defensive wrap
        raise ModelError(f"{family} training failed: {exc}") from exc
    logger.info(
        "Trained %s: %d rows, %d features, scale_pos_weight=%s",
        family,
        X.shape[0],
        X.shape[1],
        effective.get("scale_pos_weight", "n/a"),
    )
    return model, effective


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for ``X``.

    Raises:
        ModelError: on unexpected prediction output.
    """
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ModelError(f"Unexpected predict_proba shape: {proba.shape}.")
    return np.asarray(proba[:, _POSITIVE_INDEX], dtype="float64")


def _validate_metric_inputs(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> None:
    """Shared input validation for the metric helpers.

    Raises:
        ModelError: on length mismatch, an empty vector, or out-of-range
            probabilities.
    """
    if len(y_true) != len(y_prob):
        raise ModelError("y_true and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ModelError("Cannot compute metrics on an empty vector.")
    if not (0.0 <= float(np.min(y_prob)) and float(np.max(y_prob)) <= 1.0):
        raise ModelError("Probabilities must lie in [0, 1].")


def binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None
) -> dict[str, float]:
    """Compact binary-classification metrics used by M8 and M9.

    Returns precision, recall, F1 at ``threshold`` (default
    ``config.DEFAULT_THRESHOLD``) and average precision (AUC-PR).

    Raises:
        ModelError: on length mismatch or out-of-range probabilities.
    """
    threshold = threshold if threshold is not None else config.DEFAULT_THRESHOLD
    _validate_metric_inputs(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
    }
    return metrics


def full_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None
) -> dict[str, object]:
    """Full metric set for the M9 model comparison (PRD §11.4/§11.6).

    Returns precision, recall, F1, AUC-PR, ROC-AUC, Brier score, log loss
    and the 2x2 confusion matrix at ``threshold`` (default
    ``config.DEFAULT_THRESHOLD``). ROC-AUC is ``None`` only when ``y_true``
    has a single class; constant predictions return 0.5 (sklearn 1.9
    behavior). All other metrics are always computed.

    Raises:
        ModelError: on length mismatch or out-of-range probabilities.
    """
    threshold = threshold if threshold is not None else config.DEFAULT_THRESHOLD
    _validate_metric_inputs(y_true, y_prob, threshold)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "brier": float(np.mean((y_true - y_prob) ** 2)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "confusion_matrix": cm.tolist(),
    }


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """ROC-AUC, or ``None`` when ``y_true`` has a single class."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def save_model(model: Any, path: Path) -> Path:
    """Persist a fitted model (LightGBM or XGBoost) with joblib.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(path: Path) -> Any:
    """Load a model saved by :func:`save_model`.

    Raises:
        ModelError: if the file is missing or not a valid model.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelError(f"Model file not found: {path}")
    try:
        model = joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive wrap
        raise ModelError(f"Failed to load model from {path}: {exc}") from exc
    if not hasattr(model, "predict_proba"):
        raise ModelError(f"{path} is not a fitted classifier.")
    logger.info("Loaded model from %s", path)
    return model


def write_manifest(path: Path, contents: dict[str, Any]) -> Path:
    """Write a JSON document (manifest / comparison) with deterministic keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contents, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Wrote manifest: %s", path)
    return path


# ---------------------------------------------------------------------------
# M9 — threshold analysis, baselines, winner selection
# ---------------------------------------------------------------------------


def threshold_analysis(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, object]:
    """Sweep operating thresholds and locate the best operating points.

    Sweeps ``config.THRESHOLD_MIN``..``config.THRESHOLD_MAX`` at
    ``config.THRESHOLD_STEP`` and returns the full grid plus the best-F1,
    best-precision, and best-recall rows (ties broken by the secondary
    metric, then deterministically by the sweep order).

    Returns:
        ``{"rows": [...], "best_f1": row, "best_precision": row,
        "best_recall": row}`` where each row is ``{"threshold",
        "precision", "recall", "f1"}``.
    """
    steps = int(
        round((config.THRESHOLD_MAX - config.THRESHOLD_MIN) / config.THRESHOLD_STEP)
    ) + 1
    thresholds = [
        round(config.THRESHOLD_MIN + i * config.THRESHOLD_STEP, 6)
        for i in range(steps)
    ]
    rows: list[dict[str, float]] = []
    for t in thresholds:
        m = binary_metrics(y_true, y_prob, float(t))
        rows.append(
            {"threshold": t, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]}
        )
    return {
        "rows": rows,
        "best_f1": max(rows, key=lambda r: (r["f1"], r["precision"])),
        "best_precision": max(rows, key=lambda r: (r["precision"], r["recall"])),
        "best_recall": max(rows, key=lambda r: (r["recall"], r["precision"])),
    }


def majority_baseline(y_true: np.ndarray) -> np.ndarray:
    """Always predict the majority class (hard 0/1 probabilities)."""
    counts = np.bincount(np.asarray(y_true, dtype=int), minlength=2)
    majority = int(counts.argmax())
    return np.full(len(y_true), float(majority), dtype=float)


def always_positive_baseline(y_true: np.ndarray) -> np.ndarray:
    """Always predict the positive class (hard 0/1 probabilities)."""
    return np.ones(len(y_true), dtype=float)


def persistence_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Predict escalation when the unit already escalated in the last 14 days.

    Mirrors the label thresholds on the trailing window: 1 when
    ``PERSISTENCE_EVENTS_COLUMN >= ESCALATION_MIN_EVENTS`` or
    ``PERSISTENCE_FATALITIES_COLUMN >= ESCALATION_MIN_FATALITIES``.

    Raises:
        ModelError: if a required feature column is absent.
    """
    cols = (config.PERSISTENCE_EVENTS_COLUMN, config.PERSISTENCE_FATALITIES_COLUMN)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ModelError(f"Persistence baseline requires columns: {missing}.")
    escalated = (
        (frame[config.PERSISTENCE_EVENTS_COLUMN] >= config.ESCALATION_MIN_EVENTS)
        | (frame[config.PERSISTENCE_FATALITIES_COLUMN] >= config.ESCALATION_MIN_FATALITIES)
    )
    return escalated.to_numpy(dtype=float)


def event_count_heuristic_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Predict escalation when the trailing 14-day event count is high.

    1 when ``HEURISTIC_EVENTS_COLUMN >= HEURISTIC_MIN_EVENTS``.

    Raises:
        ModelError: if the required feature column is absent.
    """
    col = config.HEURISTIC_EVENTS_COLUMN
    if col not in frame.columns:
        raise ModelError(f"Event-count baseline requires column {col!r}.")
    return (frame[col] >= config.HEURISTIC_MIN_EVENTS).to_numpy(dtype=float)


def select_winner(
    comparison: dict[str, dict[str, float]],
) -> tuple[str, str]:
    """Pick the winning family per the PRD priority (M9, PRD §11.4).

    Priority: (1) highest validation F1, (2) higher AUC-PR, (3) better
    calibration (lower Brier), (4) simpler model when all are within
    ``config.MODEL_TIE_EPSILON`` — candidates are seeded from
    ``config.MODEL_SIMPLICITY_ORDER`` so the tie-break honors the config.

    Args:
        comparison: ``{family: {"f1": float, "auc_pr": float, "brier": float}}``.

    Returns:
        ``(winner, reason)`` where the reason names the deciding criterion.

    Raises:
        ModelError: if fewer than two families are compared.
    """
    families = list(comparison)
    if len(families) < 2:
        raise ModelError("select_winner requires at least two models to compare.")
    ordered = [f for f in config.MODEL_SIMPLICITY_ORDER if f in families]
    ordered += [f for f in families if f not in ordered]
    winner = ordered[0]
    for fam in ordered[1:]:
        if _beats(comparison[fam], comparison[winner]) is not None:
            winner = fam
    return winner, _winner_reason(winner, comparison)


def _beats(candidate: dict[str, float], current: dict[str, float]) -> str | None:
    """Criterion on which ``candidate`` beats ``current``, else ``None``.

    Returns the first PRD-priority criterion (f1 -> auc_pr -> brier) that
    decisively separates the two families, or ``None`` when ``candidate``
    loses or ties on every criterion (ties keep the current family).
    """
    eps = config.MODEL_TIE_EPSILON
    for key, higher_is_better in (("f1", True), ("auc_pr", True), ("brier", False)):
        c, k = candidate[key], current[key]
        if c is None or k is None:
            continue
        if abs(c - k) > eps:
            won = c > k if higher_is_better else c < k
            return key if won else None
    return None


def _winner_reason(
    winner: str, comparison: dict[str, dict[str, float]]
) -> str:
    """Justify the winner via the criterion that actually decided it.

    Scans the PRD priority chain (F1 -> PR-AUC -> Brier) and reports only
    the first criterion on which the winner decisively differs from a rival
    — so a reader never sees a metric the winner lost presented neutrally.
    """
    eps = config.MODEL_TIE_EPSILON
    rivals = [fam for fam in comparison if fam != winner]
    for key, label, higher in (
        ("f1", "validation F1", True),
        ("auc_pr", "PR-AUC", True),
        ("brier", "Brier", False),
    ):
        w = comparison[winner][key]
        if w is None:
            continue
        differing = [
            (fam, comparison[fam][key])
            for fam in rivals
            if comparison[fam][key] is not None and abs(w - comparison[fam][key]) > eps
        ]
        if differing:
            tail = ", ".join(f"{fam} {v:.4f}" for fam, v in differing)
            direction = "higher" if higher else "lower"
            return f"{label} {w:.4f} vs {tail} ({direction} {label} decided)"
    return (
        f"effectively equal; simpler family '{winner}' chosen "
        "(MODEL_SIMPLICITY_ORDER)"
    )
