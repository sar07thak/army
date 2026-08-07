"""Risk map and visualization layer (PRD §12 / FR-10, plan M12 → M11).

Loads ``models/escalation_best.pkl`` (never retrains) and the held-out test
window, computes predictions + full-window SHAP values, and produces the
hackathon presentation set:

- interactive risk map (folium, ``reports/maps/risk_map.html``) with one
  marker per geo unit, colored by risk category, popup with probability,
  predicted class, top SHAP drivers, recent events and fatalities
- country risk dashboard (matplotlib figure + plotly HTML)
- hotspot analysis: top-K ranking CSV, bar chart, temporal heatmap
- temporal trends: weekly / monthly average risk, risk evolution timeline,
  country-wise comparison
- feature importance dashboard: top-20 SHAP bar + category-wise contribution
- prediction distribution: histogram + KDE, risk-category distribution
- ``reports/risk_summary.md`` (highest/safest regions, country averages,
  top drivers, interpretation, observations)

Every matplotlib figure is saved at ``config.FIGURE_DPI`` (300 dpi) with the
Agg backend so the module runs headless. Leakage guarantees are inherited
from upstream: the model was trained on chronological splits and SHAP is
computed on the out-of-sample test window only.
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
from src.exceptions import DataLoadError, VisualizationError
from src.explainability import (
    _operating_threshold,
    compute_shap_values,
    load_test_window,
    load_winning_model,
    mean_abs_importance,
)

logger = logging.getLogger(__name__)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

def risk_category(prob: float) -> str:
    """Map a predicted probability to a risk category band.

    Uses ``config.RISK_LEVEL_BOUNDARIES`` with ``config.RISK_LEVEL_NAMES``:
    values below the first boundary get the first name, values between
    boundaries get the intermediate names, and values at or above the last
    boundary get the final name.

    Raises:
        VisualizationError: if ``prob`` is outside [0, 1].
    """
    if not 0.0 <= prob <= 1.0:
        raise VisualizationError(f"Probability out of range: {prob}.")
    names = config.RISK_LEVEL_NAMES
    for level, boundary in zip(names[:-1], config.RISK_LEVEL_BOUNDARIES):
        if prob < boundary:
            return level
    return names[-1]


def _save_figure(figure: Any, path: Path) -> Path:
    """Save a matplotlib figure at ``FIGUREDPI`` and close it."""
    path = Path(path)
    figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote figure: %s", path)
    return path


def _load_centroids(data_dir: Path) -> pd.DataFrame:
    """Mean latitude/longitude per geo unit from the cleaned events."""
    path = data_dir / f"{config.CLEANED_EVENTS_FILE}.parquet"
    if not path.is_file():
        raise DataLoadError(
            f"Cleaned events not found at {path} — run the 'ingest' stage first."
        )
    cleaned = pd.read_parquet(path)
    centroids = (
        cleaned.groupby("geo_unit", as_index=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "lat", "longitude": "lon"})
    )
    logger.info("Loaded centroids for %d geo units", len(centroids))
    return centroids


def _top_drivers(values: np.ndarray, features: list[str], k: int) -> list[str]:
    """Top-K driver strings (``feature: +0.12``) for one SHAP row."""
    order = np.argsort(-np.abs(values))[:k]
    return [f"{features[i]}: {values[i]:+.3f}" for i in order]


def _validate_shap_consistency(
    shap_values: np.ndarray, proba: np.ndarray, base_value: float
) -> None:
    """Verify SHAP values explain the model's predictions.

    For a TreeExplainer on a binary classifier, ``sum(shap) + base_value``
    equals the raw margin whose sigmoid is ``proba``. Checks a deterministic
    sample of rows and raises if the max reconstruction error is too large.

    Raises:
        VisualizationError: if SHAP does not reconstruct predictions.
    """
    sample = np.linspace(0, len(proba) - 1, min(50, len(proba))).astype(int)
    margin = shap_values[sample].sum(axis=1) + base_value
    expected = np.log(np.clip(proba[sample], 1e-9, 1 - 1e-9) / np.clip(1 - proba[sample], 1e-9, 1 - 1e-9))
    error = float(np.max(np.abs(margin - expected)))
    if error > 1e-2:
        raise VisualizationError(
            f"SHAP values do not reconstruct predictions (max log-odds error {error:.4f})."
        )
    logger.info("SHAP/prediction consistency verified (max log-odds error %.2e)", error)


def build_prediction_table(
    frame: pd.DataFrame,
    proba: np.ndarray,
    shap_values: np.ndarray,
    features: list[str],
    threshold: float,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """One snapshot row per geo unit at its latest date in the test window.

    Each row carries the predicted probability, predicted class (proba >=
    ``threshold``), risk category, top-3 SHAP drivers, recent event count and
    fatalities (7-day windows), plus centroid coordinates for the map.

    Args:
        frame: Test-window frame with features, meta columns, and label.
        proba: Positive-class probabilities aligned with ``frame`` rows.
        shap_values: SHAP matrix aligned with ``frame`` rows.
        features: Feature names (SHAP column order).
        threshold: Operating threshold for the predicted class.
        centroids: ``geo_unit``/``lat``/``lon`` frame from cleaned events.

    Returns:
        One-row-per-geo-unit snapshot table.
    """
    df = frame.reset_index(drop=True).copy()
    df[config.PREDICTION_PROBA_COLUMN] = proba
    df[config.PREDICTION_CLASS_COLUMN] = (proba >= threshold).astype(int)
    df[config.PREDICTION_CATEGORY_COLUMN] = df[config.PREDICTION_PROBA_COLUMN].map(
        risk_category
    )
    latest = df.loc[df.groupby("geo_unit")["event_date"].idxmax()].copy()
    positions = latest.index.to_numpy()
    latest["top_drivers"] = [
        ", ".join(_top_drivers(shap_values[p], features, 3)) for p in positions
    ]
    latest = latest.merge(centroids, on="geo_unit", how="left")
    missing = int(latest["lat"].isna().sum())
    if missing:
        logger.warning(
            "%d geo units have no centroid; they will be skipped on the map", missing
        )
    return latest


def save_risk_map(
    table: pd.DataFrame, out_dir: Path, filename: str | None = None
) -> Path:
    """Interactive folium risk map, one colored marker per geo unit.

    Color encodes the risk category; marker radius scales with predicted
    probability. Each popup shows geo unit, country, probability, predicted
    class, top SHAP drivers, recent events and recent fatalities.

    Args:
        table: Snapshot table (see :func:`build_prediction_table`).
        out_dir: Directory for the HTML file.
        filename: Output file name; defaults to ``config.RISK_MAP_FILE``.

    Raises:
        VisualizationError: if the map cannot be rendered.
    """
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - defensive wrap
        raise VisualizationError(f"folium is required for the risk map: {exc}") from exc
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = folium.Map(
        location=list(config.MAP_CENTER), zoom_start=config.MAP_ZOOM_START
    )
    for _, row in table.dropna(subset=["lat", "lon"]).iterrows():
        popup = folium.Popup(
            html=(
                f"<b>{row['geo_unit']}</b><br>"
                f"Country: {row['country']}<br>"
                f"Risk probability: {row[config.PREDICTION_PROBA_COLUMN]:.3f}<br>"
                f"Predicted class: {row[config.PREDICTION_CLASS_COLUMN]}<br>"
                f"Top SHAP: {row['top_drivers']}<br>"
                f"Recent events (7d): {row[config.RECENT_EVENTS_COLUMN]:.0f}<br>"
                f"Recent fatalities (7d): {row[config.RECENT_FATALITIES_COLUMN]:.0f}"
            ),
            max_width=320,
        )
        color = config.RISK_LEVEL_COLORS[row[config.PREDICTION_CATEGORY_COLUMN]]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6 + 14 * float(row[config.PREDICTION_PROBA_COLUMN]),
            tooltip=f"{row['geo_unit']} ({row[config.PREDICTION_PROBA_COLUMN]:.2f})",
            popup=popup,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
        ).add_to(m)
    legend = _legend_html()
    m.get_root().html.add_child(folium.Element(legend))
    path = out_dir / (filename or config.RISK_MAP_FILE)
    m.save(str(path))
    logger.info("Wrote risk map: %s", path)
    return path


def _legend_html() -> str:
    """HTML legend snippet for the risk-category colors."""
    chips = "".join(
        f"<li><span style='background:{color};width:12px;height:12px;"
        f"display:inline-block;margin-right:6px'></span>{level}</li>"
        for level, color in config.RISK_LEVEL_COLORS.items()
    )
    return (
        "<div style='position:fixed;bottom:30px;left:30px;z-index:9999;"
        "background:white;padding:8px 12px;border-radius:6px;box-shadow:0 0 6px "
        "rgba(0,0,0,.3);font-family:sans-serif;font-size:13px'>"
        f"<b>Risk level</b><ul style='list-style:none;margin:4px 0 0;padding:0'>{chips}</ul>"
        "</div>"
    )


def _country_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Per-country aggregates: avg risk, positive rate, mean events/fatalities."""
    agg = (
        table.groupby("country", as_index=False)
        .agg(
            avg_risk=(config.PREDICTION_PROBA_COLUMN, "mean"),
            positive_rate=(config.PREDICTION_CLASS_COLUMN, "mean"),
            mean_events=(config.RECENT_EVENTS_COLUMN, "mean"),
            mean_fatalities=(config.RECENT_FATALITIES_COLUMN, "mean"),
        )
        .sort_values("avg_risk", ascending=False)
    )
    return agg.reset_index(drop=True)


def save_country_dashboard(
    table: pd.DataFrame, figures_dir: Path, dashboard_dir: Path
) -> tuple[Path, Path]:
    """Country risk dashboard: 4-panel matplotlib PNG + plotly HTML."""
    metrics = _country_metrics(table)
    figures_dir = Path(figures_dir)
    dashboard_dir = Path(dashboard_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    png = _country_dashboard_png(metrics, figures_dir / "country_dashboard.png")
    html = _country_dashboard_html(metrics, dashboard_dir / "country_dashboard.html")
    return png, html


def _country_dashboard_png(metrics: pd.DataFrame, path: Path) -> Path:
    """Matplotlib 4-panel country dashboard at 300 dpi."""
    panels = [
        ("avg_risk", "Average risk probability", "#C1121F"),
        ("positive_rate", "Positive prediction rate", "#E76F51"),
        ("mean_fatalities", "Mean fatalities (7d)", "#F5A623"),
        ("mean_events", "Mean event count (7d)", "#2E86AB"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, label, color) in zip(axes.flat, panels):
        ax.bar(metrics["country"], metrics[col], color=color)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("value")
        ax.tick_params(axis="x", rotation=30)
    figure.suptitle("Country Risk Dashboard — 14-day escalation forecast", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(figure, path)


def _country_dashboard_html(metrics: pd.DataFrame, path: Path) -> Path:
    """Interactive plotly country dashboard with four bar subplots."""
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - defensive wrap
        raise VisualizationError(f"plotly is required for the dashboard: {exc}") from exc
    panels = [
        ("avg_risk", "Average risk probability", "#C1121F"),
        ("positive_rate", "Positive prediction rate", "#E76F51"),
        ("mean_fatalities", "Mean fatalities (7d)", "#F5A623"),
        ("mean_events", "Mean event count (7d)", "#2E86AB"),
    ]
    figure = make_subplots(
        rows=2, cols=2, subplot_titles=[label for _, label, _ in panels]
    )
    for i, (col, _, color) in enumerate(panels, start=1):
        figure.add_trace(
            go.Bar(x=metrics["country"], y=metrics[col], marker_color=color),
            row=(i - 1) // 2 + 1,
            col=(i - 1) % 2 + 1,
        )
    figure.update_layout(
        title_text="Country Risk Dashboard — 14-day escalation forecast",
        height=700,
        showlegend=False,
    )
    figure.write_html(str(path))
    logger.info("Wrote dashboard: %s", path)
    return path


def save_hotspot_analysis(
    table: pd.DataFrame,
    weekly_risk: pd.DataFrame,
    reports_dir: Path,
    figures_dir: Path,
) -> tuple[Path, Path, Path]:
    """Hotspot analysis: top-K ranking CSV, bar chart, and temporal heatmap."""
    reports_dir = Path(reports_dir)
    figures_dir = Path(figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    top = table.nlargest(config.HOTSPOT_TOP_K, config.PREDICTION_PROBA_COLUMN)
    ranking = top[
        [
            "geo_unit",
            "country",
            config.PREDICTION_PROBA_COLUMN,
            config.PREDICTION_CLASS_COLUMN,
            config.PREDICTION_CATEGORY_COLUMN,
            "top_drivers",
        ]
    ].copy()
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    csv_path = reports_dir / config.HOTSPOT_RANKING_FILE
    ranking.to_csv(csv_path, index=False)
    logger.info("Wrote hotspot ranking: %s", csv_path)
    bar_path = _hotspot_bar(ranking, figures_dir / "hotspots_bar.png")
    heat_path = _hotspot_heatmap(ranking, weekly_risk, figures_dir / "hotspots_heatmap.png")
    return csv_path, bar_path, heat_path


def _hotspot_bar(ranking: pd.DataFrame, path: Path) -> Path:
    """Horizontal bar of the top-K highest-risk geo units."""
    ordered = ranking.iloc[::-1]
    figure, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(ordered))))
    ax.barh(ordered["geo_unit"], ordered[config.PREDICTION_PROBA_COLUMN], color="#C1121F")
    ax.set_xlabel("predicted escalation probability")
    ax.set_title(f"Top {len(ordered)} highest-risk geo units (latest test window)")
    figure.tight_layout()
    return _save_figure(figure, path)


def _hotspot_heatmap(ranking: pd.DataFrame, weekly: pd.DataFrame, path: Path) -> Path:
    """Heatmap of the top-K units' mean risk over the trailing N weeks."""
    units = ranking["geo_unit"].tolist()
    cutoff = weekly["event_date"].max() - pd.Timedelta(days=7 * config.HOTSPOT_HEATMAP_WEEKS)
    recent = weekly[weekly["event_date"] >= cutoff].copy()
    recent["week"] = recent["event_date"].dt.to_period(config.RESAMPLE_WEEKLY).dt.start_time
    pivot = recent[recent["geo_unit"].isin(units)].pivot_table(
        index="geo_unit",
        columns="week",
        values=config.PREDICTION_PROBA_COLUMN,
        aggfunc="mean",
    )
    pivot = pivot.reindex(units).fillna(0.0)
    figure, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(units))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.strftime("%Y-%m-%d") for c in pivot.columns], rotation=45, ha="right")
    ax.set_title("Risk evolution of top-K hotspots (mean weekly probability)")
    figure.colorbar(image, ax=ax, label="predicted probability")
    figure.tight_layout()
    return _save_figure(figure, path)


def save_temporal_trends(
    full: pd.DataFrame, figures_dir: Path
) -> list[Path]:
    """Weekly/monthly average risk, risk evolution timeline, country trends."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weekly = (
        full.set_index("event_date")[config.PREDICTION_PROBA_COLUMN]
        .resample(config.RESAMPLE_WEEKLY)
        .mean()
        .dropna()
    )
    monthly = (
        full.set_index("event_date")[config.PREDICTION_PROBA_COLUMN]
        .resample(config.RESAMPLE_MONTHLY)
        .mean()
        .dropna()
    )
    paths = [
        _line_plot(weekly, "Weekly average escalation risk", figures_dir / "temporal_weekly.png"),
        _line_plot(monthly, "Monthly average escalation risk", figures_dir / "temporal_monthly.png"),
        _evolution_plot(weekly, figures_dir / "temporal_evolution.png"),
        _country_trend_plot(full, figures_dir / "temporal_country_comparison.png"),
    ]
    return paths


def _line_plot(series: pd.Series, title: str, path: Path) -> Path:
    """Simple labeled time-series line plot."""
    figure, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.to_numpy(), color="#2E86AB", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("predicted escalation probability")
    ax.grid(alpha=0.3)
    figure.tight_layout()
    return _save_figure(figure, path)


def _evolution_plot(weekly: pd.Series, path: Path) -> Path:
    """Risk evolution timeline: weekly average + rolling window overlay."""
    rolling = weekly.rolling(config.EVOLUTION_ROLLING_WEEKS).mean()
    figure, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly.index, weekly.to_numpy(), color="#F5A623", linewidth=0.8, label="weekly")
    ax.plot(rolling.index, rolling.to_numpy(), color="#C1121F", linewidth=2.0, label=f"{config.EVOLUTION_ROLLING_WEEKS}-week rolling")
    ax.set_title("Risk evolution timeline")
    ax.set_ylabel("predicted escalation probability")
    ax.legend()
    ax.grid(alpha=0.3)
    figure.tight_layout()
    return _save_figure(figure, path)


def _country_trend_plot(full: pd.DataFrame, path: Path) -> Path:
    """One monthly average-risk line per country."""
    monthly = (
        full.groupby(["country", pd.Grouper(key="event_date", freq=config.RESAMPLE_MONTHLY)])[
            config.PREDICTION_PROBA_COLUMN
        ]
        .mean()
        .dropna()
        .reset_index()
    )
    figure, ax = plt.subplots(figsize=(12, 6))
    for country, group in monthly.groupby("country"):
        ax.plot(group["event_date"], group[config.PREDICTION_PROBA_COLUMN], label=country, linewidth=1.5)
    ax.set_title("Country-wise average risk comparison")
    ax.set_ylabel("predicted escalation probability")
    ax.legend()
    ax.grid(alpha=0.3)
    figure.tight_layout()
    return _save_figure(figure, path)


def _feature_family(feature: str) -> str:
    """Feature family for the category-wise contribution chart."""
    if feature.startswith(("events_w", "events_log1p_w")):
        return "volume"
    if feature.startswith(("fatalities_w", "fatalities_log1p_w")):
        return "lethality"
    if feature.startswith("velocity_"):
        return "velocity"
    if feature.startswith(("fat_mean_w", "fat_std_w")):
        return "volatility"
    if feature.startswith("persistence"):
        return "persistence"
    if feature.startswith("days_since"):
        return "recency"
    if feature.startswith("spillover"):
        return "spillover"
    if feature.endswith("_code"):
        return "identity"
    if feature in ("month", "day_of_week"):
        return "calendar"
    return "other"


def _category_contribution(importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean |SHAP| by feature family, in config order."""
    grouped = importance.assign(family=importance["feature"].map(_feature_family)).groupby(
        "family", as_index=False
    )["mean_abs_shap"].sum()
    order = [f for f in config.FEATURE_FAMILY_ORDER if f in set(grouped["family"])]
    order += [f for f in grouped["family"] if f not in order]
    return grouped.set_index("family").loc[order].reset_index()


def save_importance_dashboard(
    importance: pd.DataFrame, figures_dir: Path
) -> tuple[Path, Path]:
    """Top-20 SHAP bar chart and category-wise contribution chart."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    top = importance.head(config.SHAP_TOP_N).iloc[::-1]
    figure, ax = plt.subplots(figsize=(10, max(5, 0.35 * config.SHAP_TOP_N)))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"Top {config.SHAP_TOP_N} features by mean |SHAP|")
    figure.tight_layout()
    bar_path = _save_figure(figure, figures_dir / "feature_importance.png")

    families = _category_contribution(importance)
    figure, ax = plt.subplots(figsize=(9, 5))
    ax.bar(families["family"], families["mean_abs_shap"], color="#E76F51")
    ax.set_xlabel("feature family")
    ax.set_ylabel("total mean |SHAP|")
    ax.set_title("Category-wise feature contribution")
    ax.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    cat_path = _save_figure(figure, figures_dir / "feature_category_contribution.png")
    return bar_path, cat_path


def save_prediction_distribution(
    table: pd.DataFrame, figures_dir: Path
) -> tuple[Path, Path]:
    """Histogram + KDE of predicted probabilities and risk-category bars."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    proba = table[config.PREDICTION_PROBA_COLUMN].to_numpy()

    figure, ax = plt.subplots(figsize=(10, 5))
    ax.hist(proba, bins=30, color="#2E86AB", alpha=0.7, density=True, label="histogram")
    try:
        from scipy.stats import gaussian_kde

        xs = np.linspace(0, 1, 300)
        ax.plot(xs, gaussian_kde(proba)(xs), color="#C1121F", linewidth=1.8, label="KDE")
    except ImportError:  # pragma: no cover - defensive wrap
        logger.warning("scipy unavailable; KDE skipped")
    ax.set_xlabel("predicted escalation probability")
    ax.set_ylabel("density")
    ax.set_title("Prediction distribution (test window)")
    ax.legend()
    figure.tight_layout()
    dist_path = _save_figure(figure, figures_dir / "prediction_distribution.png")

    counts = table[config.PREDICTION_CATEGORY_COLUMN].value_counts()
    figure, ax = plt.subplots(figsize=(8, 5))
    colors = [config.RISK_LEVEL_COLORS[level] for level in counts.index]
    ax.bar(counts.index, counts.to_numpy(), color=colors)
    ax.set_xlabel("risk category")
    ax.set_ylabel("geo units")
    ax.set_title("Risk-category distribution (latest snapshot)")
    figure.tight_layout()
    cat_path = _save_figure(figure, figures_dir / "risk_category_distribution.png")
    return dist_path, cat_path


def _md_table(header: list[str], rows: list[list[object]]) -> str:
    """Markdown table as a single string block."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(out)


def _region_rows(frame: pd.DataFrame) -> list[list[object]]:
    """Markdown rows for a highest/safest-regions section."""
    return [
        [
            i,
            r["geo_unit"],
            r["country"],
            f"{r[config.PREDICTION_PROBA_COLUMN]:.3f}",
            r[config.PREDICTION_CLASS_COLUMN],
            r[config.PREDICTION_CATEGORY_COLUMN],
        ]
        for i, (_, r) in enumerate(frame.iterrows(), start=1)
    ]


def _country_rows(metrics: pd.DataFrame) -> list[list[object]]:
    """Markdown rows for the average-risk-by-country section."""
    return [
        [
            r["country"],
            f"{r['avg_risk']:.3f}",
            f"{r['positive_rate']:.3f}",
            f"{r['mean_events']:.2f}",
            f"{r['mean_fatalities']:.2f}",
        ]
        for _, r in metrics.iterrows()
    ]


def _summary_metadata(
    table: pd.DataFrame, threshold: float
) -> list[str]:
    """Header lines of the risk summary report."""
    return [
        "# Risk Summary — 14-Day Conflict Escalation Forecast",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model: {config.MODEL_BEST_FILE} · operating threshold: {threshold:.2f}",
        f"- Scope: {len(table)} geo units · snapshot dates: "
        f"{table['event_date'].min().date()} .. {table['event_date'].max().date()}",
    ]


def _driver_rows(importance: pd.DataFrame) -> list[list[object]]:
    """Markdown rows for the top-SHAP-drivers section."""
    drivers = importance.head(10).reset_index(drop=True)
    drivers["rank"] = np.arange(1, len(drivers) + 1)
    return [
        [r["rank"], r["feature"], f"{r['mean_abs_shap']:.4f}"]
        for _, r in drivers.iterrows()
    ]


def write_risk_summary(
    table: pd.DataFrame,
    importance: pd.DataFrame,
    metrics: pd.DataFrame,
    threshold: float,
    path: Path,
) -> Path:
    """Render ``reports/risk_summary.md`` (regions, averages, drivers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = table.nlargest(10, config.PREDICTION_PROBA_COLUMN)
    safest = table.nsmallest(10, config.PREDICTION_PROBA_COLUMN)
    region_header = ["rank", "geo unit", "country", "probability", "class", "category"]
    lines = (
        _summary_metadata(table, threshold)
        + ["", "## Highest-risk regions (top 10)", ""]
        + [_md_table(region_header, _region_rows(top)), ""]
        + ["## Safest regions (bottom 10)", ""]
        + [_md_table(region_header, _region_rows(safest)), ""]
        + ["## Average risk by country", ""]
        + [
            _md_table(
                ["country", "avg risk", "positive rate", "mean events (7d)", "mean fatalities (7d)"],
                _country_rows(metrics),
            ),
            "",
        ]
        + ["## Top SHAP drivers", ""]
        + [_md_table(["rank", "feature", "mean |SHAP|"], _driver_rows(importance)), ""]
        + [
            "## Interpretation",
            "",
            "The risk categories are derived from `config.RISK_LEVEL_BOUNDARIES`; "
            "the predicted class uses the operating threshold from "
            "`model_comparison.json`. Each geo unit is shown at its latest date in "
            "the test window (the model's most recent assessment). Full SHAP "
            "interpretations live in `reports/shap_summary.md`.",
            "",
            "## Important observations",
            "",
        ]
        + [f"- {item}" for item in _observations(table, metrics, importance)]
        + [""]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote risk summary: %s", path)
    return path


def _observations(
    table: pd.DataFrame,
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
) -> list[str]:
    """Data-driven observations for the risk summary report."""
    top = table.nlargest(1, config.PREDICTION_PROBA_COLUMN).iloc[0]
    safest = table.nsmallest(1, config.PREDICTION_PROBA_COLUMN).iloc[0]
    counts = table[config.PREDICTION_CATEGORY_COLUMN].value_counts()
    risky_share = (
        counts.get("Critical", 0) + counts.get("High", 0)
    ) / len(table) if len(table) else 0.0
    top_country = metrics.iloc[0]["country"]
    top_driver = importance.iloc[0]["feature"]
    return [
        f"Highest-risk geo unit: **{top['geo_unit']}** ({top['country']}) with "
        f"probability {top[config.PREDICTION_PROBA_COLUMN]:.3f}.",
        f"Safest geo unit: **{safest['geo_unit']}** ({safest['country']}) with "
        f"probability {safest[config.PREDICTION_PROBA_COLUMN]:.3f}.",
        f"{100 * risky_share:.1f}% of geo units are High or Critical risk "
        f"({counts.to_dict()}).",
        f"Highest average risk country: **{top_country}** "
        f"(avg {metrics.iloc[0]['avg_risk']:.3f}).",
        f"Strongest overall risk driver: **{top_driver}** "
        f"(mean |SHAP| {importance.iloc[0]['mean_abs_shap']:.4f}).",
        "Observations are computed on the out-of-sample test window; the model "
        "was never retrained for visualization.",
    ]


def _save_map_and_dashboards(
    table: pd.DataFrame, reports_dir: Path
) -> tuple[Path, Path, Path]:
    """Risk map (folium) + country dashboard (PNG + plotly HTML)."""
    map_path = save_risk_map(table, reports_dir / config.MAPS_DIR.name)
    png, html = save_country_dashboard(
        table,
        reports_dir / config.FIGURES_DIR.name,
        reports_dir / config.DASHBOARD_DIR.name,
    )
    return map_path, png, html


def _save_hotspots_and_temporal(
    table: pd.DataFrame, full: pd.DataFrame, reports_dir: Path
) -> tuple[tuple[Path, Path, Path], list[Path]]:
    """Hotspot ranking/bar/heatmap + the four temporal trend plots."""
    weekly = full[["geo_unit", "event_date", config.PREDICTION_PROBA_COLUMN]]
    hotspot = save_hotspot_analysis(
        table, weekly, reports_dir, reports_dir / config.FIGURES_DIR.name
    )
    temporal = save_temporal_trends(full, reports_dir / config.FIGURES_DIR.name)
    return hotspot, temporal


def _save_importance_and_distribution(
    importance: pd.DataFrame, table: pd.DataFrame, reports_dir: Path
) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    """Importance/category charts + prediction distribution plots."""
    figures = reports_dir / config.FIGURES_DIR.name
    importance_plots = save_importance_dashboard(importance, figures)
    distribution_plots = save_prediction_distribution(table, figures)
    return importance_plots, distribution_plots


def _stage_summary(
    table: pd.DataFrame,
    importance: pd.DataFrame,
    threshold: float,
    paths: dict[str, object],
) -> dict[str, object]:
    """Compile the visualize-stage result summary dict."""
    top = table.nlargest(1, config.PREDICTION_PROBA_COLUMN).iloc[0]
    return {
        "geo_units": int(len(table)),
        "countries": int(table["country"].nunique()),
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
        **paths,
    }


def _prepare_predictions(
    model: Any, frame: pd.DataFrame, threshold: float, data_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predictions + SHAP + snapshot table for the test window.

    Returns ``(table, full, importance)`` where ``table`` is the latest-row
    snapshot per geo unit, ``full`` is the window with probabilities, and
    ``importance`` is the mean-|SHAP| ranking.
    """
    X, _, features = models.prepare_xy(frame)
    proba = models.predict_proba(model, X)
    _, shap_values, base_value = compute_shap_values(model, X, features)
    _validate_shap_consistency(shap_values, proba, base_value)
    importance = mean_abs_importance(shap_values, features)
    table = build_prediction_table(
        frame, proba, shap_values, features, threshold, _load_centroids(data_dir)
    )
    full = frame.reset_index(drop=True).copy()
    full[config.PREDICTION_PROBA_COLUMN] = proba
    return table, full, importance


def visualize_stage(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, object]:
    """Run the full M11 visualization stage on the winning model.

    Loads ``escalation_best.pkl`` + the test window, computes predictions and
    full-window SHAP values, and writes the risk map, dashboards, hotspot
    analysis, temporal trends, importance/density charts, and the risk
    summary report.

    Returns:
        Summary dict (artifact paths, geo units, categories, top driver).
    """
    data_dir = Path(data_dir or config.DATA_PROCESSED_DIR)
    models_dir = Path(models_dir or config.MODELS_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    model = load_winning_model(models_dir)
    frame = load_test_window(data_dir)
    threshold = _operating_threshold(models_dir)
    table, full, importance = _prepare_predictions(model, frame, threshold, data_dir)

    map_path, png, html = _save_map_and_dashboards(table, reports_dir)
    (csv_path, bar_path, heat_path), temporal_paths = _save_hotspots_and_temporal(
        table, full, reports_dir
    )
    (bar_imp, cat_imp), (dist, cat_dist) = _save_importance_and_distribution(
        importance, table, reports_dir
    )
    metrics = _country_metrics(table)
    summary_path = write_risk_summary(
        table, importance, metrics, threshold, reports_dir / config.RISK_SUMMARY_FILE
    )
    paths = {
        "risk_map": str(map_path),
        "dashboard_png": str(png),
        "dashboard_html": str(html),
        "hotspot_ranking": str(csv_path),
        "hotspot_bar": str(bar_path),
        "hotspot_heatmap": str(heat_path),
        "temporal_plots": [str(p) for p in temporal_paths],
        "importance_plots": [str(bar_imp), str(cat_imp)],
        "distribution_plots": [str(dist), str(cat_dist)],
        "summary_report": str(summary_path),
    }
    n_figures = len(temporal_paths) + len(paths["importance_plots"]) + len(paths["distribution_plots"]) + 3
    logger.info(
        "Visualization stage complete: map + %d figures + dashboard + summary",
        n_figures,
    )
    return _stage_summary(table, importance, threshold, paths)
