"""Central configuration — the single source of truth for the pipeline.

All paths, scope parameters, validation rules, feature/label constants,
split ratios, model hyperparameters, and evaluation settings live here so
that no other module hardcodes a value (IMPLEMENTATION_PLAN.md §2 and §5.3).
"""

from __future__ import annotations

from pathlib import Path

from src.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent
DATA_RAW_DIR: Path = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = REPO_ROOT / "data" / "processed"
MODELS_DIR: Path = REPO_ROOT / "models"
REPORTS_DIR: Path = REPO_ROOT / "reports"
LOGS_DIR: Path = REPO_ROOT / "logs"
LOGS_FILE: str = "project.log"

# ---------------------------------------------------------------------------
# Scope (PRD §9.2)
# ---------------------------------------------------------------------------
COUNTRIES: tuple[str, ...] = (
    "India",
    "Pakistan",
    "Afghanistan",
    "Myanmar",
    "Sudan",
    "South Sudan",
)
# "filter" drops out-of-scope rows with a logged per-country count; "error" raises instead.
COUNTRIES_MODE: str = "filter"
# Study window (inclusive). The real data spans 2016-12-31..2026-07-25.
DATE_START: str = "2016-01-01"
DATE_END: str = "2026-12-31"
# Geo units with fewer than this many events are dropped at feature time (PRD §9.2).
MIN_EVENTS_PER_UNIT: int = 5

# ---------------------------------------------------------------------------
# Canonical schema (plan §3.1; adaptation in §3.4)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_date",
    "country",
    "admin1",
    "event_type",
    "fatalities",
    "latitude",
    "longitude",
)
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "events",
    "sub_event_type",
    "admin2",
    "actor1",
    "actor2",
    "region",
    "disorder_type",
    "event_id",
)
# Used for the dedupe key when present; otherwise a composite key is derived.
EVENT_ID_SOURCE: str = "event_id_cnty"

# ---------------------------------------------------------------------------
# Validation rules (PRD FR-2; milestone requirements)
# ---------------------------------------------------------------------------
# "drop" removes duplicate event rows after a warning; "raise" errors instead.
DUPLICATES_MODE: str = "drop"
# Maximum fraction of rows that may be dropped for missing critical fields
# (event_date/country/admin1) before the pipeline errors instead.
MAX_DROPPED_FRACTION: float = 0.05
LAT_MIN: float = -90.0
LAT_MAX: float = 90.0
LON_MIN: float = -180.0
LON_MAX: float = 180.0
# Known admin-name fixes, e.g. {"Sind": "Sindh"}. All-caps acronyms such as
# "FATA" are legitimate and are intentionally left untouched.
ADMIN_NAME_NORMALIZATION: dict[str, str] = {}
DEFAULT_ACTOR_VALUE: str = "Unknown"

# ---------------------------------------------------------------------------
# Feature engineering (PRD §11.3 — consumed from M5)
# ---------------------------------------------------------------------------
ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 30)
VELOCITY_WINDOWS: tuple[int, ...] = (7, 14, 30)
VOLATILITY_WINDOWS: tuple[int, ...] = (14, 30)
PERSISTENCE_WINDOW_DAYS: int = 7  # active days in the trailing N days (PRD §11.3)
SPILLOVER_ENABLED: bool = True
SPILLOVER_WINDOW: int = 14
# Spillover neighbors: the K nearest same-country geo units by centroid distance.
SPILLOVER_K_NEIGHBORS: int = 3
# Days-since-last-event sentinel for units with no prior event.
RECENCY_SENTINEL: int = 999

# ---------------------------------------------------------------------------
# Labels (PRD §11.2 — consumed from M6)
# ---------------------------------------------------------------------------
LABEL_HORIZON_DAYS: int = 14
ESCALATION_MIN_EVENTS: int = 3
ESCALATION_MULTIPLIER: float = 1.5
ESCALATION_MIN_FATALITIES: int = 5
# Trailing window (days) for the escalation multiplier's baseline median.
TRAILING_MEDIAN_WINDOW_DAYS: int = 30
# Absolute fallback rule for units without trailing history.
ABSOLUTE_MIN_EVENTS: int = 5
# "drop" removes rows whose future window extends past the data end; "raise" errors.
INCOMPLETE_WINDOW: str = "drop"

# ---------------------------------------------------------------------------
# Split (PRD §11.5 — consumed from M7)
# ---------------------------------------------------------------------------
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED: int = 42
# Column used to order rows chronologically before cutting.
SPLIT_DATE_COLUMN: str = "event_date"
# Prefix for the per-split output files, e.g. split_train.parquet.
SPLIT_FILE_PREFIX: str = "split"
SPLIT_SUMMARY_FILE: str = "split_summary.md"

# ---------------------------------------------------------------------------
# Models (PRD §11.4 — consumed from M8/M9)
# ---------------------------------------------------------------------------
IMBALANCE_METHOD: str = "scale_pos_weight"  # "scale_pos_weight" | "class_weight"
# Columns that identify a row but are never model features.
META_COLUMNS: tuple[str, ...] = ("geo_unit", "admin1", "country", "event_date")
# Explicit feature list; empty tuple = derive as all columns except
# META_COLUMNS and the label column (the current behavior on the 38-col splits).
FEATURE_COLUMNS: tuple[str, ...] = ()
# Probability threshold used for the M8 validation F1 (full threshold
# analysis arrives in M10 via OPERATING_THRESHOLD_MODE).
DEFAULT_THRESHOLD: float = 0.5
LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "n_estimators": 500,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "scale_pos_weight": "auto",  # replaced by the train-set class ratio at training time
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}
XGB_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "scale_pos_weight": "auto",
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}
MODEL_LGBM_FILE: str = "escalation_lgbm.pkl"
MODEL_XGB_FILE: str = "escalation_xgb.pkl"
MODEL_BEST_FILE: str = "escalation_best.pkl"
MODEL_MANIFEST_FILE: str = "manifest.json"
# Model comparison (PRD §11.4 — consumed from M9).
MODEL_COMPARISON_FILE: str = "model_comparison.json"
MODEL_COMPARISON_REPORT: str = "model_comparison.md"
# Threshold sweep for operating-point selection (PRD §11.6 / FR-14).
THRESHOLD_MIN: float = 0.10
THRESHOLD_MAX: float = 0.90
THRESHOLD_STEP: float = 0.05
# Winner selection priority: F1 -> PR-AUC -> Brier, then simplicity order
# when all metrics are within MODEL_TIE_EPSILON.
MODEL_TIE_EPSILON: float = 1e-4
MODEL_SIMPLICITY_ORDER: tuple[str, ...] = ("lightgbm", "xgboost")
# Baseline rules (PRD §11.6 — consumed from M9).
HEURISTIC_MIN_EVENTS: int = 5  # event-count heuristic: events_w14d >= cutoff
PERSISTENCE_EVENTS_COLUMN: str = "events_w14d"
PERSISTENCE_FATALITIES_COLUMN: str = "fatalities_w14d"
HEURISTIC_EVENTS_COLUMN: str = "events_w14d"

# ---------------------------------------------------------------------------
# Evaluation (PRD §11.6 — consumed from M10)
# ---------------------------------------------------------------------------
OPERATING_THRESHOLD_MODE: str = "max_f1"  # "max_f1" | "0.5"
METRICS: tuple[str, ...] = ("precision", "recall", "f1", "auc_pr", "brier")
REPORT_TOP_K_DRIVERS: int = 10

# ---------------------------------------------------------------------------
# Processed outputs
# ---------------------------------------------------------------------------
CLEANED_EVENTS_FILE: str = "cleaned_events"
DISTRICT_MASTER_FILE: str = "district_master"
FEATURES_FILE: str = "features"
FEATURE_SUMMARY_FILE: str = "feature_summary.md"
LABELED_FEATURES_FILE: str = "labeled_features"
LABEL_SUMMARY_FILE: str = "label_summary.md"
LABEL_TIMELINE_FILE: str = "label_timeline.png"
LABEL_COLUMN: str = "escalation"
OUTPUT_FORMATS: tuple[str, ...] = ("parquet", "csv")

# ---------------------------------------------------------------------------
# Explainability (PRD §12 / FR-9 — consumed from M10)
# ---------------------------------------------------------------------------
# Deterministic row cap for global SHAP computation on the test window
# (evenly spaced across dates; logged). Local explanations always use the
# exact representative rows, never the cap.
SHAP_SAMPLE_CAP: int = 2000
SHAP_REPORT_DIR: Path = REPORTS_DIR / "shap"
SHAP_SUMMARY_FILE: str = "shap_summary.md"
# Top-N features in the summary report's ranking + interpretations table.
SHAP_TOP_N: int = 20
# Number of top features with dependence plots.
SHAP_DEPENDENCE_TOP_K: int = 10
# Representative predictions per local-explanation category (pos/neg/border).
SHAP_WATERFALL_COUNT: int = 3
# Top-K drivers shown per local explanation.
SHAP_LOCAL_TOP_K: int = 3
# Max features displayed in a single waterfall plot.
SHAP_MAX_DISPLAY: int = 12

# ---------------------------------------------------------------------------
# Visualization (PRD §12 / FR-10 — consumed from M11)
# ---------------------------------------------------------------------------
MAPS_DIR: Path = REPORTS_DIR / "maps"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DASHBOARD_DIR: Path = REPORTS_DIR / "dashboard"
RISK_MAP_FILE: str = "risk_map.html"
RISK_SUMMARY_FILE: str = "risk_summary.md"
HOTSPOT_RANKING_FILE: str = "hotspots_ranking.csv"
# Predicted-probability bands; len(RISK_LEVEL_BOUNDARIES) + 1 risk categories:
# RISK_LEVEL_NAMES[0] < b0, b0..b1 RISK_LEVEL_NAMES[1], ..., >= last -> final name.
RISK_LEVEL_BOUNDARIES: tuple[float, ...] = (0.2, 0.4, 0.6)
RISK_LEVEL_NAMES: tuple[str, ...] = ("Low", "Medium", "High", "Critical")
RISK_LEVEL_COLORS: dict[str, str] = {
    "Low": "#2E86AB",
    "Medium": "#F5A623",
    "High": "#E76F51",
    "Critical": "#C1121F",
}
FIGURE_DPI: int = 300  # publication quality (PRD §12)
HOTSPOT_TOP_K: int = 20  # top-K highest-risk geo units
HOTSPOT_HEATMAP_WEEKS: int = 12  # trailing weeks in the hotspot heatmap
RESAMPLE_WEEKLY: str = "W"  # pandas offset for weekly aggregation
RESAMPLE_MONTHLY: str = "ME"  # pandas offset for monthly aggregation
EVOLUTION_ROLLING_WEEKS: int = 4  # rolling window for the risk-evolution line
# Feature-family order for the category-wise contribution chart.
FEATURE_FAMILY_ORDER: tuple[str, ...] = (
    "volume",
    "lethality",
    "velocity",
    "volatility",
    "persistence",
    "recency",
    "spillover",
    "identity",
    "calendar",
)
# Snapshot columns consumed by the risk map / dashboards.
PREDICTION_PROBA_COLUMN: str = "proba"
PREDICTION_CLASS_COLUMN: str = "predicted_class"
PREDICTION_CATEGORY_COLUMN: str = "risk_category"
RECENT_EVENTS_COLUMN: str = "events_w7d"  # recent event count (popup)
RECENT_FATALITIES_COLUMN: str = "fatalities_w7d"  # recent fatalities (popup)
MAP_CENTER: tuple[float, float] = (23.0, 78.0)  # South-Asia focus
MAP_ZOOM_START: int = 5

# ---------------------------------------------------------------------------
# Live forecast (post-M13 --stage forecast)
# ---------------------------------------------------------------------------
# One-row-per-geo-unit "next 14 days" forecast artifacts, anchored at the
# latest available feature date per unit (NOT the test window).
FORECAST_CSV_FILE: str = "forecast_next_14_days.csv"
FORECAST_MAP_FILE: str = "forecast_risk_map.html"
FORECAST_SUMMARY_FILE: str = "forecast_summary.md"
# Top-K hotspots reported in the forecast summary.
FORECAST_TOP_K: int = 10


def validate_config() -> None:
    """Assert that the configuration is internally consistent.

    Raises:
        ConfigurationError: if any invariant is violated.
    """
    if not COUNTRIES:
        raise ConfigurationError("COUNTRIES must not be empty.")
    if COUNTRIES_MODE not in {"filter", "error"}:
        raise ConfigurationError(
            f"COUNTRIES_MODE must be 'filter' or 'error', got {COUNTRIES_MODE!r}."
        )
    if DUPLICATES_MODE not in {"drop", "raise"}:
        raise ConfigurationError(
            f"DUPLICATES_MODE must be 'drop' or 'raise', got {DUPLICATES_MODE!r}."
        )
    if not (0.0 < MAX_DROPPED_FRACTION < 1.0):
        raise ConfigurationError(
            f"MAX_DROPPED_FRACTION must be in (0, 1), got {MAX_DROPPED_FRACTION}."
        )
    if not (LAT_MIN < LAT_MAX and LON_MIN < LON_MAX):
        raise ConfigurationError("Coordinate bounds are inverted.")
    if tuple(ROLLING_WINDOWS) != tuple(sorted(ROLLING_WINDOWS)):
        raise ConfigurationError("ROLLING_WINDOWS must be sorted ascending.")
    if LABEL_HORIZON_DAYS <= 0:
        raise ConfigurationError("LABEL_HORIZON_DAYS must be positive.")
    if ESCALATION_MULTIPLIER <= 1.0:
        raise ConfigurationError("ESCALATION_MULTIPLIER must be > 1.0.")
    if TRAILING_MEDIAN_WINDOW_DAYS <= 0:
        raise ConfigurationError("TRAILING_MEDIAN_WINDOW_DAYS must be positive.")
    if min(ESCALATION_MIN_EVENTS, ESCALATION_MIN_FATALITIES, ABSOLUTE_MIN_EVENTS) < 1:
        raise ConfigurationError("Escalation thresholds must be >= 1.")
    if INCOMPLETE_WINDOW not in {"drop", "raise"}:
        raise ConfigurationError(
            f"INCOMPLETE_WINDOW must be 'drop' or 'raise', got {INCOMPLETE_WINDOW!r}."
        )
    if not all(v > 0 for v in SPLIT_RATIOS.values()):
        raise ConfigurationError("SPLIT_RATIOS values must be positive.")
    if abs(sum(SPLIT_RATIOS.values()) - 1.0) > 1e-9:
        raise ConfigurationError(
            f"SPLIT_RATIOS must sum to 1.0, got {sum(SPLIT_RATIOS.values())}."
        )
    if IMBALANCE_METHOD not in {"scale_pos_weight", "class_weight"}:
        raise ConfigurationError(f"Unknown IMBALANCE_METHOD {IMBALANCE_METHOD!r}.")
    if OPERATING_THRESHOLD_MODE not in {"max_f1", "0.5"}:
        raise ConfigurationError(
            f"Unknown OPERATING_THRESHOLD_MODE {OPERATING_THRESHOLD_MODE!r}."
        )
    for name, params in (("LGBM_PARAMS", LGBM_PARAMS), ("XGB_PARAMS", XGB_PARAMS)):
        weight = params.get("scale_pos_weight")
        if weight != "auto" and not isinstance(weight, (int, float)):
            raise ConfigurationError(
                f"{name} scale_pos_weight must be 'auto' or numeric, got {weight!r}."
            )
    if not (0.0 < THRESHOLD_MIN < THRESHOLD_MAX < 1.0):
        raise ConfigurationError(
            "THRESHOLD_MIN must be < THRESHOLD_MAX within (0, 1)."
        )
    if THRESHOLD_STEP <= 0:
        raise ConfigurationError("THRESHOLD_STEP must be positive.")
    span = THRESHOLD_MAX - THRESHOLD_MIN
    if abs(span / THRESHOLD_STEP - round(span / THRESHOLD_STEP)) > 1e-9:
        raise ConfigurationError(
            "(THRESHOLD_MAX - THRESHOLD_MIN) must be a multiple of "
            "THRESHOLD_STEP so the sweep ends exactly at THRESHOLD_MAX."
        )
    if MODEL_TIE_EPSILON <= 0:
        raise ConfigurationError("MODEL_TIE_EPSILON must be positive.")
    if HEURISTIC_MIN_EVENTS < 1:
        raise ConfigurationError("HEURISTIC_MIN_EVENTS must be >= 1.")
    if set(MODEL_SIMPLICITY_ORDER) != {"lightgbm", "xgboost"}:
        raise ConfigurationError(
            "MODEL_SIMPLICITY_ORDER must contain exactly 'lightgbm' and 'xgboost'."
        )
    if PERSISTENCE_WINDOW_DAYS <= 0:
        raise ConfigurationError("PERSISTENCE_WINDOW_DAYS must be positive.")
    if SPILLOVER_WINDOW <= 0:
        raise ConfigurationError("SPILLOVER_WINDOW must be positive.")
    if SPILLOVER_K_NEIGHBORS < 1:
        raise ConfigurationError("SPILLOVER_K_NEIGHBORS must be >= 1.")
    if LABEL_COLUMN in META_COLUMNS:
        raise ConfigurationError("LABEL_COLUMN must not appear in META_COLUMNS.")
    if set(FEATURE_COLUMNS) & set(META_COLUMNS):
        raise ConfigurationError("FEATURE_COLUMNS must be disjoint from META_COLUMNS.")
    if not (0.0 < DEFAULT_THRESHOLD < 1.0):
        raise ConfigurationError("DEFAULT_THRESHOLD must be in (0, 1).")
    if SHAP_SAMPLE_CAP < 1:
        raise ConfigurationError("SHAP_SAMPLE_CAP must be >= 1.")
    if SHAP_TOP_N < 1:
        raise ConfigurationError("SHAP_TOP_N must be >= 1.")
    if SHAP_DEPENDENCE_TOP_K < 1:
        raise ConfigurationError("SHAP_DEPENDENCE_TOP_K must be >= 1.")
    if SHAP_WATERFALL_COUNT < 1:
        raise ConfigurationError("SHAP_WATERFALL_COUNT must be >= 1.")
    if SHAP_LOCAL_TOP_K < 1:
        raise ConfigurationError("SHAP_LOCAL_TOP_K must be >= 1.")
    if not RISK_LEVEL_BOUNDARIES:
        raise ConfigurationError("RISK_LEVEL_BOUNDARIES must not be empty.")
    if tuple(RISK_LEVEL_BOUNDARIES) != tuple(sorted(RISK_LEVEL_BOUNDARIES)):
        raise ConfigurationError("RISK_LEVEL_BOUNDARIES must be sorted ascending.")
    if not all(0.0 < b < 1.0 for b in RISK_LEVEL_BOUNDARIES):
        raise ConfigurationError("RISK_LEVEL_BOUNDARIES must lie within (0, 1).")
    expected_levels = len(RISK_LEVEL_BOUNDARIES) + 1
    if len(RISK_LEVEL_NAMES) != expected_levels:
        raise ConfigurationError(
            f"RISK_LEVEL_NAMES must define exactly {expected_levels} categories, "
            f"got {len(RISK_LEVEL_NAMES)}."
        )
    if len(RISK_LEVEL_NAMES) != len(set(RISK_LEVEL_NAMES)):
        raise ConfigurationError("RISK_LEVEL_NAMES must be unique.")
    if set(RISK_LEVEL_NAMES) != set(RISK_LEVEL_COLORS):
        raise ConfigurationError(
            "RISK_LEVEL_COLORS keys must match RISK_LEVEL_NAMES exactly."
        )
    if FIGURE_DPI < 72:
        raise ConfigurationError("FIGURE_DPI must be >= 72.")
    if HOTSPOT_TOP_K < 1:
        raise ConfigurationError("HOTSPOT_TOP_K must be >= 1.")
    if HOTSPOT_HEATMAP_WEEKS < 1:
        raise ConfigurationError("HOTSPOT_HEATMAP_WEEKS must be >= 1.")
    if EVOLUTION_ROLLING_WEEKS < 1:
        raise ConfigurationError("EVOLUTION_ROLLING_WEEKS must be >= 1.")
    if FORECAST_TOP_K < 1:
        raise ConfigurationError("FORECAST_TOP_K must be >= 1.")
