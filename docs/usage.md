# Usage

This document explains how to install, configure, and run the pipeline stage by stage, and what outputs to expect.

---

## 1. Installation

### Prerequisites

- **Python 3.11+** (developed and verified on 3.14.3)
- `git`

### Setup

```bash
git clone <your-repo-url> army
cd army

python -m venv .venv
# Windows (bash / PowerShell)
.venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Optional: environment template (only needed if you add API-based tooling later)
cp .env.example .env
```

### Data placement

Put your ACLED export(s) into `data/raw/`. The loader auto-detects both supported formats:

- **Weekly aggregated** counts (week × country × admin1 × event_type with `events`, `fatalities`, centroids) — the format used for this project.
- **Event-level** rows (with `event_id_cnty`, `event_date`, `admin2`, `actor1`, per-event coordinates) — a drop-in swap.

See [data/README.md](../data/README.md) for provenance and format details.

---

## 2. CLI Overview

```bash
python run_pipeline.py --stage <stage>
```

| Stage | Purpose | Required before |
|---|---|---|
| `ingest` | Validate + canonicalize raw CSVs → cleaned events + district master | data in `data/raw/` |
| `features` | Rolling-window feature table | `ingest` |
| `labels` | 14-day escalation labels | `features` |
| `split` | Chronological train/val/test | `labels` |
| `train` | LightGBM → `escalation_lgbm.pkl` | `split` |
| `compare` | XGBoost + comparison → winner `escalation_best.pkl` | `train` |
| `explain` | SHAP explainability | `compare` |
| `visualize` | Risk map + dashboards | `compare` (+ `cleaned_events` for centroids) |
| `forecast` | **Live** next-14-days forecast per geo unit | `features` + `compare` |

Run `python run_pipeline.py --help` for the argument summary.

---

## 3. Running Each Stage

### 3.1 Ingest

```bash
python run_pipeline.py --stage ingest
```

- Reads every `*.csv` in `data/raw/`, merges, canonicalizes the schema, validates all rules, builds the district master.
- Writes `data/processed/cleaned_events.{parquet,csv}` and `data/processed/district_master.{parquet,csv}`.
- Prints a summary: cleaned rows, geo units, countries, date range.
- **Expected output:** `cleaned_rows=127052`, `geo_units=124`, `countries=6` on the bundled dataset.

### 3.2 Features

```bash
python run_pipeline.py --stage features
```

- Builds all rolling-window features per geo unit (past-only, leakage-safe).
- Writes `data/processed/features.{parquet,csv}` + `reports/feature_summary.md`.
- **Expected output:** `feature_rows=44146`, `feature_columns=37`, 0 NaN, 0 duplicates.

### 3.3 Labels

```bash
python run_pipeline.py --stage labels
```

- Attaches the 14-day escalation label (future-only) and drops incomplete-window rows.
- Writes `data/processed/labeled_features.{parquet,csv}` + `reports/label_summary.md` + `reports/label_timeline.png`.
- **Expected output:** `labeled_rows=43981`, `positives=30217` (68.7%).

### 3.4 Split

```bash
python run_pipeline.py --stage split
```

- Chronological cut over the date axis (70/15/15, no shuffle).
- Writes `data/processed/split_{train,val,test}.{parquet,csv}` + `reports/split_summary.md`.
- **Expected output:** `30,790 / 6,522 / 6,669` rows with strict temporal separation.

### 3.5 Train (LightGBM)

```bash
python run_pipeline.py --stage train
```

- Trains LightGBM with `scale_pos_weight` imbalance handling, fixed seed.
- Writes `models/escalation_lgbm.pkl` + `models/manifest.json`.
- **Expected output:** validation F1 ≈ 0.793 @ 0.5 (best-F1 threshold 0.20 → F1 0.840).

### 3.6 Compare (XGBoost + selection)

```bash
python run_pipeline.py --stage compare
```

- Verifies the saved LightGBM is unchanged, trains XGBoost on identical data, runs the full metric comparison, threshold sweep, and 4 baselines, then selects the winner per PRD priority.
- Writes `models/escalation_xgb.pkl`, `models/escalation_best.pkl`, `models/model_comparison.json`, `reports/model_comparison.md`.
- **Expected output:** winner = `xgboost`, validation F1 0.8423 @ operating threshold 0.25.

### 3.7 Explain (SHAP)

```bash
python run_pipeline.py --stage explain
```

- Computes TreeExplainer SHAP on the held-out test window (deterministic sample cap).
- Writes `reports/shap_summary.md` + `reports/shap/` (summary, bar, waterfall, dependence plots).
- **Expected output:** 21 plots; top driver `events_w30d` (mean |SHAP| 0.478).

### 3.8 Visualize

```bash
python run_pipeline.py --stage visualize
```

- Predicts on the full test window, computes SHAP, builds the latest-row snapshot per geo unit, and renders everything.
- Writes `reports/maps/risk_map.html`, `reports/dashboard/country_dashboard.html`, `reports/figures/*` (11 PNGs @ 300 dpi), `reports/hotspots_ranking.csv`, `reports/risk_summary.md`.
- **Expected output:** 122 geo units across 6 countries; highest risk = Khyber Pakhtunkhwa (0.999).

### 3.9 Forecast (live next-14-days)

```bash
python run_pipeline.py --stage forecast
```

- Loads the winning model + the **latest feature row per geo unit** (the real current state — not the test window), computes predictions + SHAP, and writes the operational forecast.
- Writes `reports/forecast_next_14_days.csv` (one row per geo unit: probability, class, risk category, top SHAP drivers, recent 7d events/fatalities), `reports/maps/forecast_risk_map.html`, `reports/forecast_summary.md`.
- **Expected output:** 124 geo units as of 2026-07-25; highest risk = Balochistan (0.998); top driver `events_w30d`.

---

## 4. Running Everything

```bash
for stage in ingest features labels split train compare explain visualize forecast; do
  python run_pipeline.py --stage "$stage" || break
done
```

> The risk map tiles (OpenStreetMap) require internet when opened in a browser; markers, popups, and the legend render offline. The plotly dashboard is fully self-contained.

---

## 5. Configuration

All constants live in **[config.py](../config.py)**. The most commonly changed settings:

| Setting | Key | Default | Notes |
|---|---|---|---|
| Countries | `COUNTRIES` | India, Pakistan, Afghanistan, Myanmar, Sudan, South Sudan | scope filter |
| Study window | `DATE_START` / `DATE_END` | 2016-01-01 / 2026-12-31 | inclusive |
| Label horizon | `LABEL_HORIZON_DAYS` | 14 | prediction window |
| Escalation thresholds | `ESCALATION_MIN_EVENTS` / `ESCALATION_MIN_FATALITIES` | 3 / 5 | label definition |
| Split ratios | `SPLIT_RATIOS` | 70/15/15 | must sum to 1 |
| Seed | `RANDOM_SEED` | 42 | reproducibility |
| Imbalance | `IMBALANCE_METHOD` | `scale_pos_weight` | or `class_weight` |
| Operating threshold | `OPERATING_THRESHOLD_MODE` | `max_f1` | or `0.5` |
| Risk bands | `RISK_LEVEL_BOUNDARIES` | (0.2, 0.4, 0.6) | Low/Medium/High/Critical |
| Figure resolution | `FIGURE_DPI` | 300 | publication quality |

`config.validate_config()` runs at every stage and raises `ConfigurationError` with a descriptive message if an invariant is broken (e.g., split ratios not summing to 1, risk bands unsorted, threshold grid not exact).

---

## 6. Tests

```bash
python -m pytest                  # 243 tests, no display required
python -m pytest -q               # quiet, fast feedback
python -m pytest --cov=src        # coverage report (96.5% overall, gate ≥80%)
python -m pytest tests/test_visualize.py   # single module
```

---

## 7. Expected Artifacts (summary)

| Artifact | Where | From stage |
|---|---|---|
| Cleaned events + district master | `data/processed/` | ingest |
| Feature table + summary | `data/processed/features.*`, `reports/feature_summary.md` | features |
| Labeled table + summary + timeline | `data/processed/labeled_features.*`, `reports/label_summary.md`, `reports/label_timeline.png` | labels |
| Splits + summary | `data/processed/split_*.parquet`, `reports/split_summary.md` | split |
| Models + manifest | `models/escalation_{lgbm,xgb,best}.pkl`, `models/manifest.json` | train / compare |
| Comparison report | `models/model_comparison.json`, `reports/model_comparison.md` | compare |
| SHAP plots + report | `reports/shap/*`, `reports/shap_summary.md` | explain |
| Risk map + dashboards + figures | `reports/maps/`, `reports/dashboard/`, `reports/figures/`, `reports/hotspots_ranking.csv`, `reports/risk_summary.md` | visualize |
| Live forecast | `reports/forecast_next_14_days.csv`, `reports/maps/forecast_risk_map.html`, `reports/forecast_summary.md` | forecast |
| Logs | `logs/project.log` | every stage |

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `DataLoadError: Cleaned dataset not found` | Run `--stage ingest` first (stages are sequential). |
| `Winning model not found` | Run `--stage compare` first. |
| `ConfigurationError: SPLIT_RATIOS must sum to 1.0` | Fix `config.SPLIT_RATIOS`. |
| `VisualizationError: Probability out of range` | Predictions must be in [0,1]; check the model artifact. |
| `ForecastError: Training features missing from the features table` | Re-run `--stage features` after any config change (features must match the model's training columns). |
| Map opens but no tiles | Internet required for OSM tiles; markers/popups work offline. |
| Tests fail with display errors | All plotting uses the Agg backend; ensure no `plt.show()` was added. |
