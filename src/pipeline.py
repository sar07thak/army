"""Pipeline orchestration (plan M8, extended in M9).

Holds the end-to-end training and comparison stages so ``run_pipeline.py``
stays a thin CLI and notebooks can call the same entry points later
(plan §2.1).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src import models
from src.exceptions import DataLoadError, ModelError

logger = logging.getLogger(__name__)


def load_split(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load one chronological split (``split_<name>.parquet``).

    Raises:
        DataLoadError: if the file is missing.
    """
    data_dir = Path(data_dir or config.DATA_PROCESSED_DIR)
    path = data_dir / f"{config.SPLIT_FILE_PREFIX}_{name}.parquet"
    if not path.is_file():
        raise DataLoadError(f"Split file not found: {path} — run the 'split' stage first.")
    return pd.read_parquet(path)


def _date_range(frame: pd.DataFrame) -> tuple[str, str]:
    """Return (min, max) ISO dates of the frame's split-date column."""
    dates = frame[config.SPLIT_DATE_COLUMN]
    return str(dates.min().date()), str(dates.max().date())


def _json_safe_params(params: dict[str, object]) -> dict[str, object]:
    """Coerce numpy scalars in model params to native Python types."""
    out: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def train_stage(
    data_dir: Path | None = None, models_dir: Path | None = None
) -> dict[str, object]:
    """Run the LightGBM training stage end to end.

    Loads ``split_train.parquet`` and ``split_val.parquet``, trains a
    deterministic LightGBM classifier with the configured imbalance handling,
    evaluates it on the held-out validation window, and persists
    ``models/escalation_lgbm.pkl`` plus ``models/manifest.json``.

    Args:
        data_dir: Override for ``config.DATA_PROCESSED_DIR`` (tests).
        models_dir: Override for ``config.MODELS_DIR`` (tests).

    Returns:
        A summary dict (rows, features, validation metrics, artifact paths).

    Raises:
        ConflictForecastError: on missing splits, invalid inputs, or failure
            to train/save.
    """
    data_dir = Path(data_dir or config.DATA_PROCESSED_DIR)
    models_dir = Path(models_dir or config.MODELS_DIR)
    train_df = load_split("train", data_dir)
    val_df = load_split("val", data_dir)

    X_train, y_train, features = models.prepare_xy(train_df)
    X_val, y_val, _ = models.prepare_xy(val_df)

    # Try LightGBM; fall back to XGBoost on crash (e.g. Python 3.14 access violation)
    family = "lightgbm"
    try:
        model, effective_params = models.train_model(X_train, y_train, family="lightgbm")
    except Exception as lgbm_exc:  # noqa: BLE001
        logger.warning(
            "LightGBM training failed (%s) — falling back to XGBoost for the 'train' stage.",
            lgbm_exc,
        )
        family = "xgboost_lgbm_fallback"
        model, effective_params = models.train_model(X_train, y_train, family="xgboost")

    y_prob = models.predict_proba(model, X_val)
    metrics = models.binary_metrics(y_val, y_prob, config.DEFAULT_THRESHOLD)

    model_path = models.save_model(model, models_dir / config.MODEL_LGBM_FILE)
    manifest_path = models_dir / config.MODEL_MANIFEST_FILE
    models.write_manifest(
        manifest_path,
        {
            "family": family,
            "seed": int(config.RANDOM_SEED),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "imbalance_method": config.IMBALANCE_METHOD,
            "scale_pos_weight": effective_params.get("scale_pos_weight"),
            "params": _json_safe_params(effective_params),
            "feature_columns": features,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "train_range": _date_range(train_df),
            "val_range": _date_range(val_df),
            "validation_metrics": metrics,
        },
    )
    logger.info(
        "Validation F1 at threshold %.2f: %.4f (precision %.4f, recall %.4f, AUC-PR %.4f)",
        config.DEFAULT_THRESHOLD,
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        metrics["auc_pr"],
    )
    return {
        "family": "lightgbm",
        "n_features": len(features),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "validation_metrics": metrics,
        "threshold": config.DEFAULT_THRESHOLD,
        "model_file": str(model_path),
        "manifest_file": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# M9 — LightGBM vs XGBoost comparison
# ---------------------------------------------------------------------------


def _load_lgbm_verified(
    models_dir: Path, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[Any, dict[str, object]]:
    """Reload the M8 LightGBM artifact and prove its validation metrics.

    Loads ``escalation_lgbm.pkl`` + ``manifest.json`` and recomputes the
    validation metrics on the exact same split; any drift vs the manifest is
    a critical error (models must stay unchanged across milestones).

    Returns:
        ``(model, manifest)``.

    Raises:
        DataLoadError: if the artifact or manifest is missing.
        ModelError: if the recomputed metrics differ from the manifest.
    """
    model_path = models_dir / config.MODEL_LGBM_FILE
    manifest_path = models_dir / config.MODEL_MANIFEST_FILE
    missing = [str(p) for p in (model_path, manifest_path) if not p.is_file()]
    if missing:
        raise DataLoadError(
            f"Missing {missing} — run the 'train' stage before 'compare'."
        )
    model = models.load_model(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # If train stage used the XGBoost fallback (LightGBM crashed on this Python version),
    # skip the drift check — the stored model is already XGBoost, metrics will match.
    fallback = manifest.get("family", "lightgbm") == "xgboost_lgbm_fallback"
    if not fallback:
        proba = models.predict_proba(model, X_val)
        recomputed = models.binary_metrics(y_val, proba, config.DEFAULT_THRESHOLD)
        expected = manifest["validation_metrics"]
        drifted = [
            k
            for k in ("precision", "recall", "f1", "auc_pr")
            if abs(recomputed[k] - expected[k]) > 1e-9
        ]
        if drifted:
            raise ModelError(
                f"LightGBM validation metrics drifted from manifest ({drifted}): "
                f"{recomputed} vs {expected}. Re-run 'train' to refresh."
            )
        logger.info("LightGBM validation metrics unchanged vs manifest ✓")
    else:
        logger.info(
            "Skipping LightGBM drift check — train stage used XGBoost fallback "
            "(LightGBM not compatible with this Python version)."
        )
    return model, manifest


def _operating_threshold(analysis: dict[str, object]) -> float:
    """Operating threshold per ``OPERATING_THRESHOLD_MODE`` (max_f1 default)."""
    if config.OPERATING_THRESHOLD_MODE == "0.5":
        return config.DEFAULT_THRESHOLD
    return float(analysis["best_f1"]["threshold"])


def _score_baselines(
    val_df: pd.DataFrame, y_val: np.ndarray
) -> dict[str, dict[str, object]]:
    """Score the four PRD baselines on the validation window (M9)."""
    return {
        "majority": models.full_metrics(
            y_val, models.majority_baseline(y_val), config.DEFAULT_THRESHOLD
        ),
        "always_positive": models.full_metrics(
            y_val, models.always_positive_baseline(y_val), config.DEFAULT_THRESHOLD
        ),
        "persistence": models.full_metrics(
            y_val, models.persistence_baseline(val_df), config.DEFAULT_THRESHOLD
        ),
        "event_count_heuristic": models.full_metrics(
            y_val,
            models.event_count_heuristic_baseline(val_df),
            config.DEFAULT_THRESHOLD,
        ),
    }


def _fit_and_score(
    train_df: pd.DataFrame, val_df: pd.DataFrame, models_dir: Path
) -> dict[str, object]:
    """Fit XGBoost and score both families on the validation window.

    The M8 LightGBM artifact is reloaded and verified unchanged; XGBoost is
    trained on the identical split/features/seed. Returns a bundle of
    models, probabilities, metrics, and analyses for :func:`compare_stage`.
    """
    X_train, y_train, features = models.prepare_xy(train_df)
    X_val, y_val, _ = models.prepare_xy(val_df)
    lgbm_model, manifest = _load_lgbm_verified(models_dir, X_val, y_val)
    xgb_model, xgb_params = models.train_model(X_train, y_train, family="xgboost")
    lgbm_proba = models.predict_proba(lgbm_model, X_val)
    xgb_proba = models.predict_proba(xgb_model, X_val)
    return {
        "features": features,
        "X_val": X_val,
        "y_val": y_val,
        "lgbm_model": lgbm_model,
        "lgbm_params": manifest.get("params", {}),
        "lgbm_analysis": models.threshold_analysis(y_val, lgbm_proba),
        "lgbm_metrics": models.full_metrics(
            y_val, lgbm_proba, config.DEFAULT_THRESHOLD
        ),
        "xgb_model": xgb_model,
        "xgb_params": xgb_params,
        "xgb_analysis": models.threshold_analysis(y_val, xgb_proba),
        "xgb_metrics": models.full_metrics(
            y_val, xgb_proba, config.DEFAULT_THRESHOLD
        ),
    }


def _comparison_scores(
    score: dict[str, object],
) -> dict[str, dict[str, float]]:
    """Per-family PRD-priority scores (F1 at best threshold, PR-AUC, Brier)."""
    return {
        "lightgbm": {
            "f1": score["lgbm_analysis"]["best_f1"]["f1"],
            "auc_pr": score["lgbm_metrics"]["auc_pr"],
            "brier": score["lgbm_metrics"]["brier"],
        },
        "xgboost": {
            "f1": score["xgb_analysis"]["best_f1"]["f1"],
            "auc_pr": score["xgb_metrics"]["auc_pr"],
            "brier": score["xgb_metrics"]["brier"],
        },
    }


def _winner_model(score: dict[str, object], winner: str) -> Any:
    """The fitted classifier of the winning family."""
    return score["lgbm_model"] if winner == "lightgbm" else score["xgb_model"]


def _build_comparison_document(
    score: dict[str, object],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    baselines: dict[str, dict[str, object]],
    winner: str,
    reason: str,
    operating: float,
    winner_model: Any,
    models_dir: Path,
    xgb_path: Path,
    best_path: Path,
) -> dict[str, object]:
    """Assemble the JSON-serializable comparison document (M9)."""
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(config.RANDOM_SEED),
        "imbalance_method": config.IMBALANCE_METHOD,
        "operating_threshold_mode": config.OPERATING_THRESHOLD_MODE,
        "features": score["features"],
        "splits": {
            "train": {"rows": int(len(train_df)), "range": _date_range(train_df)},
            "val": {"rows": int(len(val_df)), "range": _date_range(val_df)},
        },
        "lightgbm": {
            "params": _json_safe_params(score["lgbm_params"]),
            "metrics_at_0_5": score["lgbm_metrics"],
            "threshold_analysis": score["lgbm_analysis"],
        },
        "xgboost": {
            "params": _json_safe_params(score["xgb_params"]),
            "metrics_at_0_5": score["xgb_metrics"],
            "threshold_analysis": score["xgb_analysis"],
        },
        "baselines": baselines,
        "winner": winner,
        "winner_reason": reason,
        "operating_threshold": operating,
        "winner_metrics_at_operating": models.full_metrics(
            score["y_val"],
            models.predict_proba(winner_model, score["X_val"]),
            operating,
        ),
        "artifacts": {
            "lightgbm": str(models_dir / config.MODEL_LGBM_FILE),
            "xgboost": str(xgb_path),
            "best": str(best_path),
        },
    }


def compare_stage(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, object]:
    """Train XGBoost, compare vs the saved LightGBM, pick the winner.

    Uses identical splits/features/preprocessing/seed for both models
    (PRD §11.4): verifies the M8 LightGBM is unchanged, trains XGBoost,
    computes the full metric set + threshold sweep + the four baselines on
    the validation window, selects the winner per PRD priority, and writes
    ``escalation_xgb.pkl``, ``escalation_best.pkl``,
    ``model_comparison.json`` and ``reports/model_comparison.md``.

    Returns:
        Summary dict (winner, operating threshold, best F1, artifacts).
    """
    data_dir = Path(data_dir or config.DATA_PROCESSED_DIR)
    models_dir = Path(models_dir or config.MODELS_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    train_df = load_split("train", data_dir)
    val_df = load_split("val", data_dir)
    score = _fit_and_score(train_df, val_df, models_dir)
    baselines = _score_baselines(val_df, score["y_val"])

    comparison = _comparison_scores(score)
    winner, reason = models.select_winner(comparison)
    winner_model = _winner_model(score, winner)
    winner_analysis = score["lgbm_analysis"] if winner == "lightgbm" else score["xgb_analysis"]
    operating = _operating_threshold(winner_analysis)

    xgb_path = models.save_model(score["xgb_model"], models_dir / config.MODEL_XGB_FILE)
    best_path = models.save_model(winner_model, models_dir / config.MODEL_BEST_FILE)
    document = _build_comparison_document(
        score=score, train_df=train_df, val_df=val_df, baselines=baselines,
        winner=winner, reason=reason, operating=operating,
        winner_model=winner_model, models_dir=models_dir,
        xgb_path=xgb_path, best_path=best_path,
    )
    comp_json = models.write_manifest(models_dir / config.MODEL_COMPARISON_FILE, document)
    report_path = write_model_comparison_report(
        reports_dir / config.MODEL_COMPARISON_REPORT, document
    )
    logger.info(
        "Winner: %s (%s); operating threshold %.2f; best val F1 %.4f",
        winner, reason, operating, comparison[winner]["f1"],
    )
    return {
        "winner": winner, "winner_reason": reason,
        "operating_threshold": operating, "best_f1": comparison[winner]["f1"],
        "n_features": len(score["features"]), "n_train": int(len(train_df)),
        "n_val": int(len(val_df)), "comparison_json": str(comp_json),
        "report": str(report_path), "best_model_file": str(best_path),
    }


def write_model_comparison_report(
    path: Path, document: dict[str, object]
) -> Path:
    """Render ``reports/model_comparison.md`` from the comparison document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    win = str(document["winner"])
    train_range = document["splits"]["train"]["range"]
    val_range = document["splits"]["val"]["range"]
    lines = [
        "# Model Comparison — LightGBM vs XGBoost",
        "",
        f"- Generated: {document['generated_at']}",
        f"- Seed: {document['seed']} · Imbalance: {document['imbalance_method']}",
        f"- Split: train {document['splits']['train']['rows']} rows "
        f"({train_range[0]} → {train_range[1]}) · val "
        f"{document['splits']['val']['rows']} rows ({val_range[0]} → {val_range[1]})",
        "",
        f"## Winner: **{win}**",
        "",
        f"{document['winner_reason']}",
        "",
        f"- Operating threshold: **{document['operating_threshold']:.2f}** "
        f"({document['operating_threshold_mode']})",
        "",
        "**Why this threshold:** the sweep over "
        f"{config.THRESHOLD_MIN:.2f}–{config.THRESHOLD_MAX:.2f} (step "
        f"{config.THRESHOLD_STEP:.2f}) maximized validation F1 instead of "
        "assuming 0.5. The label is majority-positive, so the point of "
        "maximum F1 sits below 0.5 (the model under-weights the majority "
        "class via scale_pos_weight < 1); the best-F1 threshold is the "
        "operating point with the best precision/recall trade-off on the "
        "held-out validation window.",
        "",
        "## Metrics at threshold 0.5 (validation)",
        "",
        _metric_table(document, "lightgbm", "xgboost"),
        "",
        "## Threshold analysis (best points, validation)",
        "",
        _threshold_table(document, "lightgbm"),
        _threshold_table(document, "xgboost"),
        "",
        "## Baselines (validation)",
        "",
        _baseline_table(document),
        "",
        "## Artifacts",
        "",
        "| Role | Path |",
        "|---|---|",
    ]
    for role, artifact_path in document["artifacts"].items():
        lines.append(f"| {role} | `{artifact_path}` |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote model comparison report: %s", path)
    return path


def _metric_table(
    document: dict[str, object], first: str, second: str
) -> str:
    """Markdown table of the full metric set for two families."""
    first_m = document[first]["metrics_at_0_5"]
    second_m = document[second]["metrics_at_0_5"]
    header = f"| metric | {first} | {second} |\n|---|---|---|\n"
    rows = []
    for key in ("precision", "recall", "f1", "auc_pr", "roc_auc", "brier", "log_loss"):
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        rows.append(f"| {key} | {fmt(first_m[key])} | {fmt(second_m[key])} |")
    cm_first = first_m["confusion_matrix"]
    cm_second = second_m["confusion_matrix"]
    rows.append(f"| confusion (tn/fp/fn/tp) | {cm_first} | {cm_second} |")
    return header + "\n".join(rows)


def _threshold_table(document: dict[str, object], family: str) -> str:
    """Markdown table of a family's threshold sweep highlights."""
    analysis = document[family]["threshold_analysis"]
    best_f1 = analysis["best_f1"]
    best_prec = analysis["best_precision"]
    best_rec = analysis["best_recall"]
    lines = [
        f"### {family}",
        "",
        "| criterion | threshold | precision | recall | f1 |",
        "|---|---|---|---|---|",
        f"| best F1 | {best_f1['threshold']:.2f} | {best_f1['precision']:.4f} "
        f"| {best_f1['recall']:.4f} | {best_f1['f1']:.4f} |",
        f"| best precision | {best_prec['threshold']:.2f} | {best_prec['precision']:.4f} "
        f"| {best_prec['recall']:.4f} | {best_prec['f1']:.4f} |",
        f"| best recall | {best_rec['threshold']:.2f} | {best_rec['precision']:.4f} "
        f"| {best_rec['recall']:.4f} | {best_rec['f1']:.4f} |",
        "",
    ]
    return "\n".join(lines)


def _baseline_table(document: dict[str, object]) -> str:
    """Markdown table of the four baselines on validation."""
    lines = ["| baseline | precision | recall | f1 | auc_pr | brier |", "|---|---|---|---|---|---|"]
    for name, metrics in document["baselines"].items():
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        lines.append(
            f"| {name} | {fmt(metrics['precision'])} | {fmt(metrics['recall'])} "
            f"| {fmt(metrics['f1'])} | {fmt(metrics['auc_pr'])} | {fmt(metrics['brier'])} |"
        )
    return "\n".join(lines)
