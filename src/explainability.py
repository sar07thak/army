"""SHAP explainability for the winning model (PRD §12 / FR-9, plan M11 → M10).

Loads ``models/escalation_best.pkl`` (never retrains), computes TreeExplainer
SHAP values on the held-out **test window**, and produces:

- global feature importance (mean |SHAP|) with rankings
- SHAP summary (beeswarm) and bar plots
- waterfall plots for representative predictions (correct positives,
  correct negatives, borderline)
- dependence plots for the top-K features
- local explanations (top-K drivers) for the same representative rows
- ``reports/shap_summary.md`` with the top-20 features, interpretations,
  risk drivers, and model-behaviour observations

All plots are saved under ``reports/shap/`` using the Agg backend so the
module runs headless (tests, CI, server) without a display.

Leakage guarantees are inherited from upstream: the model was trained on
chronological splits and the test window is never used for training, so
SHAP explanations describe behaviour on data the model has not seen.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src import models
from src.exceptions import DataLoadError, ExplainabilityError

logger = logging.getLogger(__name__)

# Positive-class index for sklearn-style predict_proba output.
_POSITIVE_INDEX = 1

# Agg is set once at import so every matplotlib figure is headless-safe.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_winning_model(models_dir: Path | None = None) -> Any:
    """Load the winner artifact (``escalation_best.pkl``).

    Raises:
        DataLoadError: if the artifact is missing.
    """
    models_dir = Path(models_dir or config.MODELS_DIR)
    path = models_dir / config.MODEL_BEST_FILE
    if not path.is_file():
        raise DataLoadError(
            f"Winning model not found: {path} — run the 'compare' stage first."
        )
    logger.info("Loading winning model: %s", path)
    return models.load_model(path)


def load_test_window(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the held-out test split for explanation.

    Raises:
        DataLoadError: if the split is missing.
    """
    data_dir = Path(data_dir or config.DATA_PROCESSED_DIR)
    path = data_dir / f"{config.SPLIT_FILE_PREFIX}_test.parquet"
    if not path.is_file():
        raise DataLoadError(
            f"Test split not found: {path} — run the 'split' stage first."
        )
    logger.info("Loading test window: %s", path)
    return pd.read_parquet(path)


def _sample_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministically downsample to ``SHAP_SAMPLE_CAP`` evenly spaced rows."""
    if len(frame) <= config.SHAP_SAMPLE_CAP:
        return frame
    indices = np.linspace(0, len(frame) - 1, config.SHAP_SAMPLE_CAP).astype(int)
    sampled = frame.iloc[indices].reset_index(drop=True)
    logger.info(
        "SHAP sample cap applied: %d -> %d rows (evenly spaced across the window)",
        len(frame),
        len(sampled),
    )
    return sampled


def compute_shap_values(
    model: Any, X: np.ndarray, features: list[str]
) -> tuple[Any, np.ndarray, float]:
    """Compute TreeExplainer SHAP values on a feature matrix.

    Args:
        model: Fitted classifier (LightGBM or XGBoost) with ``predict_proba``.
        X: Float feature matrix.
        features: Feature column names (must match ``X`` width).

    Returns:
        ``(explainer, shap_values, base_value)`` where ``shap_values`` has
        shape ``(n_samples, n_features)`` and ``base_value`` is the expected
        model output (positive class).

    Raises:
        ExplainabilityError: on any SHAP failure or dimension mismatch.
    """
    if X.shape[1] != len(features):
        raise ExplainabilityError(
            f"Feature count mismatch: X has {X.shape[1]} columns but "
            f"{len(features)} feature names were provided."
        )
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X)
    except Exception as exc:  # pragma: no cover - defensive wrap
        raise ExplainabilityError(f"SHAP computation failed: {exc}") from exc
    values = _positive_class_values(raw)
    expected = explainer.expected_value
    if isinstance(expected, (list, tuple)):
        expected = np.asarray(expected, dtype=float)
    if isinstance(expected, np.ndarray):
        base = float(expected[_POSITIVE_INDEX]) if expected.size > 1 else float(expected[0])
    else:
        base = float(expected)
    if values.shape != (X.shape[0], X.shape[1]):
        raise ExplainabilityError(
            f"SHAP values shape {values.shape} does not match expected "
            f"{(X.shape[0], X.shape[1])}."
        )
    return explainer, values, base


def _positive_class_values(raw: Any) -> np.ndarray:
    """Extract the positive-class SHAP matrix from a TreeExplainer output."""
    if isinstance(raw, list):
        raw = raw[_POSITIVE_INDEX] if len(raw) > 1 else raw[0]
    values = np.asarray(raw, dtype=float)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        values = values[:, :, _POSITIVE_INDEX]
    return values


def mean_abs_importance(
    shap_values: np.ndarray, features: list[str]
) -> pd.DataFrame:
    """Rank features by mean |SHAP| with a share column."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = df["mean_abs_shap"].sum()
    df["share"] = df["mean_abs_shap"] / total if total > 0 else 0.0
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _render_figure(path: Path, figure: Any, dpi: int = 130) -> Path:
    """Save a matplotlib figure to ``path`` and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote plot: %s", path)
    return path


def save_summary_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    features: list[str],
    out_dir: Path,
) -> Path:
    """Beeswarm summary plot (top ``SHAP_TOP_N`` features)."""
    import shap

    path = out_dir / "summary_plot.png"
    figure = plt.figure(figsize=(10, max(5, config.SHAP_TOP_N * 0.35)))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=features,
        max_display=config.SHAP_TOP_N,
        show=False,
    )
    return _render_figure(path, figure)


def save_bar_plot(
    importance: pd.DataFrame, out_dir: Path
) -> Path:
    """Horizontal bar chart of mean |SHAP| (top ``SHAP_TOP_N``)."""
    top = importance.head(config.SHAP_TOP_N).iloc[::-1]  # ascending for barh
    path = out_dir / "bar_plot.png"
    figure, ax = plt.subplots(figsize=(9, max(5, config.SHAP_TOP_N * 0.35)))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP| (positive-class)")
    ax.set_title("Global feature importance (mean |SHAP|)")
    figure.tight_layout()
    return _render_figure(path, figure)


def save_waterfall_plot(
    base_value: float,
    values: np.ndarray,
    X_row: np.ndarray,
    features: list[str],
    path: Path,
) -> Path:
    """Waterfall plot for a single prediction row (SHAP values precomputed)."""
    import shap

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=X_row,
        feature_names=features,
    )
    figure = plt.figure(figsize=(10, max(5, config.SHAP_MAX_DISPLAY * 0.45)))
    shap.plots.waterfall(explanation, max_display=config.SHAP_MAX_DISPLAY, show=False)
    return _render_figure(path, figure)


def save_dependence_plot(
    feature: str,
    feature_index: int,
    shap_values: np.ndarray,
    X: np.ndarray,
    features: list[str],
    out_dir: Path,
) -> Path:
    """SHAP dependence plot for one feature."""
    import shap

    path = out_dir / f"dependence_{feature}.png"
    figure = plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_index,
        shap_values,
        X,
        feature_names=features,
        show=False,
    )
    return _render_figure(path, figure)


def _row_meta(
    frame: pd.DataFrame, index: int
) -> dict[str, object]:
    """Human-readable metadata for a test-window row."""
    row = frame.iloc[index]
    return {
        "geo_unit": str(row.get("geo_unit", "?")),
        "country": str(row.get("country", "?")),
        "admin1": str(row.get("admin1", "?")),
        "event_date": str(pd.Timestamp(row["event_date"]).date()),
    }


def select_representative_rows(
    proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    count: int,
) -> dict[str, np.ndarray]:
    """Indices of representative predictions per local-explanation category.

    Categories (PRD §12 demo narrative):
    - ``pos``: correctly predicted positive cases (y=1, p>=threshold),
      ranked by confidence descending.
    - ``neg``: correctly predicted negative cases (y=0, p<threshold),
      ranked by confidence descending (lowest probability first).
    - ``border``: difficult/borderline cases — smallest |p - threshold|.

    Returns:
        ``{"pos": ndarray, "neg": ndarray, "border": ndarray}``.
    """
    eps = 1e-9
    pos_idx = np.nonzero((y_true == 1) & (proba >= threshold))[0]
    neg_idx = np.nonzero((y_true == 0) & (proba < threshold - eps))[0]
    border_idx = np.argsort(np.abs(proba - threshold))[:count]
    pos_rank = np.argsort(-proba[pos_idx])[:count] if len(pos_idx) else np.array([], dtype=int)
    neg_rank = np.argsort(proba[neg_idx])[:count] if len(neg_idx) else np.array([], dtype=int)
    return {
        "pos": pos_idx[pos_rank],
        "neg": neg_idx[neg_rank],
        "border": border_idx,
    }


def _interpret_feature(feature: str) -> str:
    """Human-readable interpretation of a feature name (M5 convention)."""
    if feature.startswith("events_log1p_w"):
        days = feature.replace("events_log1p_w", "").replace("d", "")
        return f"Log1p-transformed event count in the trailing {days}-day window (skew-robust volume)."
    if feature.startswith("events_w"):
        days = feature.replace("events_w", "").replace("d", "")
        return f"Total events in the trailing {days}-day window — recent activity volume."
    if feature.startswith("fatalities_log1p_w"):
        days = feature.replace("fatalities_log1p_w", "").replace("d", "")
        return f"Log1p-transformed fatalities in the trailing {days}-day window."
    if feature.startswith("fatalities_w"):
        days = feature.replace("fatalities_w", "").replace("d", "")
        return f"Total fatalities in the trailing {days}-day window — recent lethality."
    if feature.startswith("entropy_w"):
        days = feature.replace("entropy_w", "").replace("d", "")
        return f"Shannon entropy of the event-type mix over {days} days — tactical diversity."
    if feature.startswith("velocity_events_w"):
        days = feature.replace("velocity_events_w", "").replace("d", "")
        return f"Event-count velocity: current {days}-day count minus the preceding {days}-day count."
    if feature.startswith("velocity_fatalities_w"):
        days = feature.replace("velocity_fatalities_w", "").replace("d", "")
        return f"Fatality velocity: current {days}-day count minus the preceding {days}-day count."
    if feature.startswith("fat_mean_w"):
        days = feature.replace("fat_mean_w", "").replace("d", "")
        return f"Mean daily fatalities over {days} days — average intensity."
    if feature.startswith("fat_std_w"):
        days = feature.replace("fat_std_w", "").replace("d", "")
        return f"Std-dev of daily fatalities over {days} days — volatility/spikiness."
    if feature == "persistence_w7d":
        return "Active days (events>0) in the trailing 7-day window — how sustained the activity is."
    if feature == "days_since_event":
        return "Days since the unit's last recorded event (999 sentinel = no history)."
    if feature == "spillover_w14d":
        return "Events in the trailing 14-day window across the K nearest same-country units — spatial spillover."
    if feature == "country_code":
        return "Deterministic numeric country identifier (captures cross-country baselines)."
    if feature in ("admin1_code", "geo_unit_code"):
        return "Deterministic numeric admin-1 / geo-unit identifier (unit-level baselines)."
    if feature == "month":
        return "Calendar month of the prediction date (seasonality)."
    if feature == "day_of_week":
        return "Day of week of the prediction date."
    return f"Engineered feature '{feature}' (see feature_summary.md)."


def _interpret_all(importance: pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable interpretations to the importance ranking."""
    out = importance.copy()
    out["interpretation"] = out["feature"].map(_interpret_feature)
    return out


def _observation_window_emphasis(by_name: dict[str, float]) -> str:
    """Which window length carries the most volume/fatality signal."""
    name = lambda f: by_name.get(f, 0.0)
    w30 = sum(name(f) for f in by_name if "w30d" in f and "velocity" not in f)
    w14 = sum(name(f) for f in by_name if "w14d" in f and "velocity" not in f)
    w7 = sum(name(f) for f in by_name if "w7d" in f and "velocity" not in f)
    window = "30-day" if w30 >= w14 and w30 >= w7 else ("14-day" if w14 >= w7 else "7-day")
    return (
        f"Window emphasis: **{window}** windows carry the most volume/fatality "
        f"signal (mean |SHAP| sums: 30d={w30:.3f}, 14d={w14:.3f}, 7d={w7:.3f})."
    )


def _observation_direction(importance: pd.DataFrame, feature: str) -> str:
    """One-line contribution statement for a feature family.

    Family sums use the FULL ranking (not the top-N subset) so totals are
    exact and can never contradict the data.
    """
    by_name = dict(zip(importance["feature"], importance["mean_abs_shap"]))
    if feature == "spillover_w14d":
        value = by_name.get(feature, 0.0)
        rank = (
            int(np.flatnonzero(importance["feature"].to_numpy() == feature)[0]) + 1
            if feature in by_name
            else None
        )
        clause = f"ranks #{rank} (mean |SHAP| {value:.4f})" if rank else f"mean |SHAP| {value:.4f}"
        verdict = "spatial contagion matters" if value > 0.1 else "limited neighbourhood effect here"
        return f"Spillover {clause} — {verdict} (FR-13)."
    if feature == "velocity_events_w14d":
        value = by_name.get(feature, 0.0)
        verdict = "material, so the model reacts to accelerating violence" if value > 0.1 else "secondary to absolute volume"
        return f"Event velocity (14d) contributes {value:.3f} in mean |SHAP| — {verdict}."
    if feature == "fat_std_w14d":
        value = by_name.get(feature, 0.0)
        verdict = "spiky/irregular violence is a visible risk signal" if value > 0.1 else "a minor factor here"
        return f"Fatality volatility (std) contributes {value:.3f} — {verdict}."
    if feature == "admin1_code":
        identity = sum(by_name.get(k, 0.0) for k in ("admin1_code", "geo_unit_code", "country_code"))
        calendar = sum(by_name.get(k, 0.0) for k in ("month", "day_of_week"))
        clause = "unit-level baselines shape the estimate" if identity > calendar else "temporal/seasonal structure dominates identity"
        return (
            f"Identity codes (admin1/geo-unit/country) contribute {identity:.3f} "
            f"and calendar features {calendar:.3f} — {clause}."
        )
    return ""


def _behaviour_observations(importance: pd.DataFrame) -> list[str]:
    """Data-driven model-behaviour observations from the importance ranking.

    Every statement is derived from the actual ``mean |SHAP|`` ranking so the
    report can never contradict the numbers (as a hand-written version did
    in an earlier draft).
    """
    top = importance.head(config.SHAP_TOP_N)
    by_name = dict(zip(importance["feature"], importance["mean_abs_shap"]))
    first = top.iloc[0]
    obs = [
        f"The single strongest driver is **{first['feature']}** "
        f"(mean |SHAP| {first['mean_abs_shap']:.4f}, {100 * first['share']:.1f}% of total).",
        _observation_window_emphasis(by_name),
        _observation_direction(importance, "velocity_events_w14d"),
        _observation_direction(importance, "fat_std_w14d"),
        _observation_direction(importance, "spillover_w14d"),
        _observation_direction(importance, "admin1_code"),
        "The operating threshold (max-F1, below 0.5) reflects the "
        "majority-positive label; SHAP is computed on the held-out test "
        "window, so these are out-of-sample explanations.",
    ]
    return obs


def _local_explanations_section(
    representatives: dict[str, list[dict[str, object]]],
) -> list[str]:
    """Markdown lines for the per-category local-explanations section."""
    lines: list[str] = ["## Local explanations (representative predictions)", ""]
    for category, label in (
        ("pos", "correctly predicted POSITIVE cases"),
        ("neg", "correctly predicted NEGATIVE cases"),
        ("border", "difficult / borderline predictions"),
    ):
        lines.append(f"### {label}")
        lines.append("")
        if not representatives[category]:
            lines.append("No representative rows in this category.")
            lines.append("")
            continue
        for example in representatives[category]:
            lines.append(
                f"- **{example['geo_unit']}** ({example['country']}, {example['admin1']}) "
                f"on {example['event_date']}: predicted {example['proba']:.3f}, "
                f"true label {example['label']} · waterfall `{example['waterfall']}`"
            )
            lines.append(f"  - Top drivers: {', '.join(example['drivers'])}")
        lines.append("")
    return lines


def write_shap_summary(
    importance: pd.DataFrame,
    representatives: dict[str, list[dict[str, object]]],
    threshold: float,
    path: Path,
) -> Path:
    """Render ``reports/shap_summary.md`` (top-20 + interpretations + drivers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = _interpret_all(importance.head(config.SHAP_TOP_N))
    lines = [
        "# SHAP Explainability Summary — Winning Model",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model: {config.MODEL_BEST_FILE} · operating threshold: {threshold:.2f}",
        f"- SHAP: TreeExplainer · positive class: escalation within {config.LABEL_HORIZON_DAYS} days",
        "",
        f"## Top {config.SHAP_TOP_N} features by mean |SHAP|",
        "",
        _md_table(
            ["rank", "feature", "mean |SHAP|", "share", "interpretation"],
            [
                [
                    r["rank"],
                    r["feature"],
                    f"{r['mean_abs_shap']:.4f}",
                    f"{100 * r['share']:.2f}%",
                    r["interpretation"],
                ]
                for _, r in top.iterrows()
            ],
        ),
        "",
        "## Most influential risk drivers",
        "",
        "The strongest drivers are ranked above; observations below are ",
        "computed directly from the ranking. Dependence plots live under ",
        "``reports/shap/``.",
        "",
        "## Model behaviour observations",
        "",
    ]
    lines += [f"- {obs}" for obs in _behaviour_observations(importance)]
    lines += [""] + _local_explanations_section(representatives)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote SHAP summary: %s", path)
    return path


def _md_table(header: list[str], rows: list[list[object]]) -> str:
    """Markdown table as a single string block."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(out)


def _local_explanation(
    values: np.ndarray, features: list[str]
) -> list[str]:
    """Top-K driver strings ("feature: +0.12") for one row's SHAP values."""
    order = np.argsort(-np.abs(values))[: config.SHAP_LOCAL_TOP_K]
    return [f"{features[i]}: {values[i]:+.3f}" for i in order]


def _operating_threshold(models_dir: Path) -> float:
    """Operating threshold from the M9 comparison document (fallback 0.5)."""
    path = models_dir / config.MODEL_COMPARISON_FILE
    try:
        import json

        doc = json.loads(path.read_text(encoding="utf-8"))
        return float(doc["operating_threshold"])
    except (OSError, ValueError, KeyError):
        logger.warning(
            "model_comparison.json missing/invalid — falling back to DEFAULT_THRESHOLD=%.2f",
            config.DEFAULT_THRESHOLD,
        )
        return config.DEFAULT_THRESHOLD


def _build_representatives(
    explainer: Any,
    base_value: float,
    X: np.ndarray,
    proba: np.ndarray,
    y: np.ndarray,
    features: list[str],
    frame: pd.DataFrame,
    threshold: float,
    out_dir: Path,
) -> tuple[dict[str, list[dict[str, object]]], int]:
    """Waterfall plots + local explanations for representative predictions.

    Returns ``(representatives, count)`` where ``count`` is the number of
    waterfall plots written (for the stage summary log).
    """
    representatives: dict[str, list[dict[str, object]]] = {
        "pos": [], "neg": [], "border": []
    }
    waterfall_idx = 0
    for category in ("pos", "neg", "border"):
        indices = select_representative_rows(
            proba, y, threshold, config.SHAP_WATERFALL_COUNT
        )[category]
        for row_idx in indices:
            waterfall_idx += 1
            path = out_dir / f"waterfall_{category}_{waterfall_idx:03d}.png"
            row_values = _positive_class_values(
                explainer.shap_values(X[row_idx].reshape(1, -1))
            )[0]
            save_waterfall_plot(base_value, row_values, X[row_idx], features, path)
            representatives[category].append(
                {
                    **_row_meta(frame, row_idx),
                    "proba": float(proba[row_idx]),
                    "label": int(y[row_idx]),
                    "waterfall": path.name,
                    "drivers": _local_explanation(row_values, features),
                }
            )
    return representatives, waterfall_idx


def _save_dependence_plots(
    importance: pd.DataFrame,
    shap_values: np.ndarray,
    X: np.ndarray,
    features: list[str],
    out_dir: Path,
) -> int:
    """Dependence plots for the top ``SHAP_DEPENDENCE_TOP_K`` features."""
    top_k = importance.head(config.SHAP_DEPENDENCE_TOP_K)
    for _, row in top_k.iterrows():
        feature_index = features.index(row["feature"])
        save_dependence_plot(
            row["feature"], feature_index, shap_values, X, features, out_dir
        )
    return len(top_k)


def _stage_summary(
    models_dir: Path,
    out_dir: Path,
    summary_path: Path,
    n_features: int,
    n_explained: int,
    threshold: float,
    importance: pd.DataFrame,
    representatives: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    """Compile the explain-stage result summary dict."""
    return {
        "model_file": str(models_dir / config.MODEL_BEST_FILE),
        "n_features": n_features,
        "n_explained_rows": n_explained,
        "operating_threshold": threshold,
        "top_features": [
            {"rank": int(r["rank"]), "feature": r["feature"], "mean_abs_shap": float(r["mean_abs_shap"])}
            for _, r in importance.head(config.SHAP_TOP_N).iterrows()
        ],
        "representatives": {
            category: len(examples) for category, examples in representatives.items()
        },
        "plots": sorted(str(p) for p in out_dir.glob("*.png")),
        "summary_report": str(summary_path),
    }


def explain_stage(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, object]:
    """Run the full SHAP explainability stage on the winning model.

    Loads ``escalation_best.pkl`` + the test window, computes TreeExplainer
    SHAP values, writes summary/bar/waterfall/dependence plots and local
    explanations to ``reports/shap/``, and ``reports/shap_summary.md``.

    Returns:
        Summary dict (artifact paths, top-20 features, threshold).
    """
    data_dir = Path(data_dir or config.DATA_PROCESSED_DIR)
    models_dir = Path(models_dir or config.MODELS_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    out_dir = reports_dir / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_winning_model(models_dir)
    frame = load_test_window(data_dir)
    X, y, features = models.prepare_xy(frame)
    proba = models.predict_proba(model, X)
    threshold = _operating_threshold(models_dir)

    sampled = _sample_rows(frame)
    X_sampled, _, _ = models.prepare_xy(sampled)
    explainer, shap_values, base_value = compute_shap_values(model, X_sampled, features)
    importance = mean_abs_importance(shap_values, features)

    save_summary_plot(shap_values, X_sampled, features, out_dir)
    save_bar_plot(importance, out_dir)
    representatives, waterfall_count = _build_representatives(
        explainer, base_value, X, proba, y, features, frame, threshold, out_dir
    )
    dependence_count = _save_dependence_plots(
        importance, shap_values, X_sampled, features, out_dir
    )
    summary_path = write_shap_summary(
        importance, representatives, threshold, reports_dir / config.SHAP_SUMMARY_FILE
    )
    n_local = sum(len(v) for v in representatives.values())
    logger.info(
        "SHAP stage complete: %d features, %d plots, %d local explanations -> %s",
        len(features), waterfall_count + dependence_count + 2, n_local, out_dir,
    )
    return _stage_summary(
        models_dir,
        out_dir,
        summary_path,
        len(features),
        int(X_sampled.shape[0]),
        threshold,
        importance,
        representatives,
    )
