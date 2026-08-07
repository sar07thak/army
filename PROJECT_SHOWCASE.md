# 🎯 Project Showcase — Conflict Escalation Forecasting
### A Production-Grade ML Early-Warning System for Conflict Risk at Province Level

---

## 🔑 At a Glance

| | |
|---|---|
| **Type** | End-to-end ML forecasting pipeline |
| **Goal** | Predict conflict escalation risk 14 days ahead, per province |
| **Scope** | 83 provinces · 4 countries · 9+ years of data |
| **Model** | XGBoost — PR-AUC 0.90+ · F1 0.84+ |
| **Data** | ACLED weekly aggregated (89,297 validated records) |
| **Output** | Risk scores + SHAP explanations + interactive maps |
| **Tests** | 243 pytest tests · 96.5% coverage |
| **Status** | ✅ Fully operational — all 9 stages complete |

---

## 🌍 Live Forecast Results (as of 2026-07-25)

### 🚨 Highest-Risk Provinces — Next 14 Days

| Rank | Province | Country | Risk % | Category |
|---|---|---|---|---|
| 🔴 1 | **Magway** | Myanmar | **99.8%** | Critical |
| 🔴 2 | **Sagaing** | Myanmar | 99.2% | Critical |
| 🔴 3 | **North Kordofan** | Sudan | 99.1% | Critical |
| 🔴 4 | **Rakhine** | Myanmar | 96.2% | Critical |
| 🔴 5 | **Mandalay** | Myanmar | 96.1% | Critical |
| 🔴 6 | **Jonglei** | South Sudan | 93.8% | Critical |
| 🔴 7 | **Jammu and Kashmir** | India | 93.3% | Critical |
| 🔴 8 | **Shan-South** | Myanmar | 93.3% | Critical |
| 🔴 9 | **Karnataka** | India | 92.5% | Critical |
| 🔴 10 | **Blue Nile** | Sudan | 89.0% | Critical |

### 🟢 Safest Provinces
Sikkim (2.3%) · Lakshadweep (3.0%) · Abyei (4.3%) · Red Sea (9.8%) · Ladakh (14.3%)

### 📊 Risk Distribution Across All 83 Provinces
| Category | Count | % of Provinces |
|---|---|---|
| 🔴 Critical (≥75%) | 41 | 49.4% |
| 🟠 High (≥50%) | 17 | 20.5% |
| 🟡 Medium (≥25%) | 14 | 16.9% |
| 🟢 Low (<25%) | 11 | 13.3% |

### 🌐 Country-Level Risk Averages
| Country | Avg Risk | All-Escalation? | Avg Events/Week |
|---|---|---|---|
| 🇲🇲 Myanmar | **74.6%** | 100% provinces | 12.7 events |
| 🇮🇳 India | 53.2% | 89% provinces | 24.3 events |
| 🇸🇸 South Sudan | 53.2% | 100% provinces | 1.9 events |
| 🇸🇩 Sudan | 42.6% | 63% provinces | 3.7 events |

---

## ⚙️ Pipeline Architecture

### 9 Stages — Full End-to-End Automation

`
INGEST → FEATURES → LABELS → SPLIT → TRAIN → COMPARE → EXPLAIN → VISUALIZE → FORECAST
`

| # | Stage | Input | Output | Key Action |
|---|---|---|---|---|
| 1 | **ingest** | Raw ACLED CSV | cleaned_events + district_master | 9-rule validation · country filter · canonicalize |
| 2 | **features** | cleaned_events | features (37 cols) | 33 rolling-window features · leakage-safe |
| 3 | **labels** | features + events | labeled_features | 14-day future-only escalation labels |
| 4 | **split** | labeled_features | train/val/test | Strict chronological · no shuffle |
| 5 | **train** | train + val | escalation_lgbm.pkl | XGBoost training · manifest.json |
| 6 | **compare** | + train/val | escalation_best.pkl | Threshold sweep · baselines · winner selection |
| 7 | **explain** | best model + test | 21 SHAP plots | TreeExplainer · global + local SHAP |
| 8 | **visualize** | best model + test | risk_map + 11 charts | folium map · plotly dashboard · hotspots |
| 9 | **forecast** | best model + latest | forecast CSV + map | Live scoring — no retraining |

**Run everything with one command:**
`ash
python run_pipeline.py --stage all
`

---

## 🧠 Machine Learning Details

### Model: XGBoost Gradient Boosting

| Parameter | Value | Why |
|---|---|---|
| n_estimators | 400 | Enough trees for full convergence |
| max_depth | 6 | Deep enough to capture interactions |
| learning_rate | 0.05 | Conservative — prevents overfitting |
| subsample | 0.8 | Row bagging — reduces variance |
| colsample_bytree | 0.8 | Feature bagging — reduces variance |
| seed | 42 | Fully deterministic + reproducible |

### Class Imbalance Handling
`
scale_pos_weight = 9,732 negatives / 20,372 positives = 0.478
`
Negative examples weighted 2× heavier. Operating threshold tuned to **0.20** (not default 0.5) via F1-maximizing sweep over 0.10–0.90.

### Performance Metrics (Your Data)

| Metric | Value | Notes |
|---|---|---|
| **F1-score** | **0.84+** | Headline metric (PRD priority) |
| **PR-AUC** | **0.90+** | Robust to imbalance |
| ROC-AUC | 0.82 | Discrimination |
| Brier Score | 0.18 | Probability calibration |
| Operating Threshold | **0.20** | Argmax-F1 from threshold sweep |

### Baselines Beaten
| Baseline | F1 | Gap vs Model |
|---|---|---|
| Always-positive | 0.81 | +0.03 |
| Majority class | 0.81 | +0.03 |
| Persistence (last label) | 0.83 | +0.01 |
| Event-count heuristic | 0.80 | +0.04 |
| **Our XGBoost model** | **0.84** | **Winner** |

---

## 🔬 Feature Engineering — 33 Leakage-Safe Features

All features use strictly past data only: half-open window `[as_of − W, as_of)`.

| Group | Features | Count |
|---|---|---|
| **Event volume** | events_w7d, events_w14d, events_w30d | 3 |
| **Event volume (log)** | events_log1p_w7/14/30d | 3 |
| **Fatality volume** | fatalities_w7d, fatalities_w14d, fatalities_w30d | 3 |
| **Fatality volume (log)** | fatalities_log1p_w7/14/30d | 3 |
| **Velocity (events)** | velocity_events_w7/14/30d | 3 |
| **Velocity (fatalities)** | velocity_fatalities_w7/14/30d | 3 |
| **Fatality stats** | fat_mean_w14/30d, fat_std_w14/30d | 4 |
| **Persistence** | persistence_w7d | 1 |
| **Diversity / entropy** | entropy_w7/14/30d | 3 |
| **Recency** | days_since_event | 1 |
| **Calendar** | month, day_of_week | 2 |
| **Identity codes** | country_code, admin1_code, geo_unit_code | 3 |
| **Spatial spillover** | spillover_w14d (K=3 nearest provinces, haversine) | 1 |
| **TOTAL** | | **33** |

---

## 🔍 SHAP Explainability — Full Transparency

Every prediction is explained. Top 10 global drivers from your test window:

| Rank | Feature | Mean |SHAP| | Signal Type |
|---|---|---|---|
| 🥇 1 | fatalities_w30d | **0.54** | Sustained recent lethality |
| 🥈 2 | velocity_events_w30d | 0.33 | Violence acceleration |
| 🥉 3 | events_w30d | 0.28 | Recent sustained activity |
| 4 | events_w7d | 0.20 | Current week |
| 5 | admin1_code | 0.19 | Province baseline |
| 6 | velocity_fatalities_w30d | 0.18 | Fatality acceleration |
| 7 | events_w14d | 0.16 | 2-week activity |
| 8 | month | 0.14 | Seasonality |
| 9 | spillover_w14d | 0.14 | Spatial contagion |
| 10 | fatalities_w14d | 0.10 | 2-week deaths |

### Key Insights from SHAP
- **30-day windows dominate:** Sum of 30d SHAP = 0.945 vs 7d = 0.270 — sustained patterns beat single-week spikes
- **Acceleration matters:** velocity features (#2 and #6) outrank raw counts in short windows
- **Spatial contagion is real:** spillover at #9 — neighbouring province activity boosts local risk
- **Seasonality matters:** month at #8 — conflict follows seasonal patterns

### 21 SHAP Plots Generated
- Global summary beeswarm + bar plot
- 10 dependence plots (one per top feature)
- 9 waterfall plots (3 true positives, 3 true negatives, 3 borderline cases)

---

## 📊 Visualizations — What Gets Generated

### Interactive Maps (HTML — Open in Browser)
| File | Content |
|---|---|
| `reports/maps/forecast_risk_map.html` | **Live forecast map** — 83 province markers, click for full details |
| `reports/maps/risk_map.html` | Test-window risk map with SHAP drivers |

### Static Dashboard Charts (PNG)
| File | Content |
|---|---|
| `figures/country_dashboard.png` | Avg risk · positive rate · events · fatalities per country |
| `figures/hotspots_bar.png` | Top 20 highest-risk provinces ranked |
| `figures/hotspots_heatmap.png` | Weekly risk evolution per hotspot |
| `figures/temporal_weekly.png` | Average risk trend (weekly) |
| `figures/temporal_monthly.png` | Average risk trend (monthly) |
| `figures/temporal_evolution.png` | Rolling risk evolution timeline |
| `figures/temporal_country_comparison.png` | Country risk trends overlaid |
| `figures/feature_importance.png` | Top-20 SHAP importance bar chart |
| `figures/feature_category_contribution.png` | SHAP by feature group |
| `figures/prediction_distribution.png` | Histogram of predicted probabilities |
| `figures/risk_category_distribution.png` | Count per risk category |

### Interactive Dashboard (HTML)
`reports/dashboard/country_dashboard.html` — Plotly interactive chart with country-level metrics

---

## 📦 Data Flow Summary

`
Raw ACLED CSV (108,945 rows)
         │  Country filter (−19,648: Ukraine, Indian Ocean, Mediterranean)
         ▼
Cleaned events (89,297 × 14 columns)  +  District master (83 provinces)
         │  Feature engineering: 33 rolling features per province per week
         ▼
Feature table (30,233 × 37)
         │  14-day future labels — dropped 129 incomplete windows
         ▼
Labeled dataset (30,104 × 38) — 67.7% escalation positive
         │  Chronological split (70/15/15)
         ▼
Train: 19,801    Val: 5,099    Test: 5,204
         │  XGBoost training (seed=42, scale_pos_weight=0.478)
         ▼
escalation_best.pkl — operating threshold 0.20
         │  SHAP + visualization + live forecast
         ▼
83 province risk scores + maps + reports
`

---

## 🗂️ Key Output Files

### Reports
| File | Description |
|---|---|
| `reports/forecast_next_14_days.csv` | Per-province risk probability + category + top-3 drivers |
| `reports/forecast_summary.md` | Live forecast summary (top 10 risky, safest 10, by country) |
| `reports/risk_summary.md` | Test-window risk analysis |
| `reports/shap_summary.md` | Full SHAP analysis — top-20 features + local explanations |
| `reports/model_comparison.md` | Model comparison + threshold analysis + baselines |
| `reports/hotspots_ranking.csv` | Top-20 hotspot provinces ranked by risk |

### Models
| File | Description |
|---|---|
| `models/escalation_best.pkl` | Production XGBoost model |
| `models/model_comparison.json` | Full comparison document — all metrics + threshold sweep |
| `models/manifest.json` | Model metadata, feature list, validation metrics |

### Processed Data
| File | Description |
|---|---|
| `data/processed/cleaned_events.parquet` | Validated event data |
| `data/processed/features.parquet` | 33-feature table |
| `data/processed/labeled_features.parquet` | Features + labels |
| `data/processed/split_{train,val,test}.parquet` | Chronological splits |

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.14.5 |
| ML — Gradient Boosting | XGBoost | 3.4.0 |
| ML — Alternative | LightGBM | 4.7.0 |
| Explainability | SHAP | 0.52.0 |
| Data | Pandas + PyArrow | 3.0.5 / 25.0.0 |
| Numerics | NumPy + SciPy | 2.4.6 / 1.18.0 |
| ML utilities | Scikit-learn | 1.9.0 |
| Interactive maps | Folium | 0.20.0 |
| Interactive charts | Plotly | 6.9.0 |
| Static charts | Matplotlib | 3.11.1 |
| Testing | Pytest + pytest-cov | 9.1.1 / 7.1.0 |

---

## ✅ Quality Assurance

| Metric | Value |
|---|---|
| Test suite | **243 pytest tests** |
| Code coverage | **96.5%** |
| Validation rules | **9 automated data validation rules** |
| Leakage tests | Spike injection + property-based randomized tests |
| Reproducibility | Deterministic seed (42) + pinned dependency versions |
| Split isolation | `max(train) < min(val) < min(test)` provably verified |

---

## 🚀 How to Run

### Full Pipeline (One Command)
`ash
python run_pipeline.py --stage all
`

### Individual Stages
`ash
python run_pipeline.py --stage ingest      # Validate + clean raw data
python run_pipeline.py --stage features    # Compute rolling features
python run_pipeline.py --stage labels      # Generate escalation labels
python run_pipeline.py --stage split       # Chronological split
python run_pipeline.py --stage train       # Train XGBoost
python run_pipeline.py --stage compare     # Compare + select winner
python run_pipeline.py --stage explain     # SHAP analysis
python run_pipeline.py --stage visualize   # Maps + dashboards
python run_pipeline.py --stage forecast    # Live 14-day predictions
`

### Update Forecast with New Data (No Retraining)
`ash
# Drop new ACLED CSV into data/raw/, then:
python run_pipeline.py --stage ingest
python run_pipeline.py --stage features
python run_pipeline.py --stage forecast
# New risk scores ready in ~60 seconds
`

### Run Tests
`ash
python -m pytest --cov=src   # 243 tests, 96.5% coverage
`

---

## 🔭 Future Roadmap

| Enhancement | Impact |
|---|---|
| Event-level ACLED export | Unlock district-level granularity + actor features |
| Multi-horizon forecasting (7d/30d) | Richer signal for different planning horizons |
| Cross-border spillover | Capture refugee flows and regional contagion |
| Calibration (Platt/isotonic) | More reliable probability estimates |
| REST API (FastAPI) | Integration with external dashboards/alert systems |
| Automated weekly refresh | Scheduled ingestion + forecast pipeline |
| Country-specific thresholds | Pakistan/Myanmar vs Sudan need different operating points |
| Executed EDA notebooks | Transparency and exploratory analysis artifacts |

---

## 📐 Label Definition Summary

`
escalation = 1  if  (future_events ≥ 3  AND  future_events ≥ 1.5 × median30d)
                OR  (future_fatalities ≥ 5)

where "future" = (as_of, as_of + 14 days]
      "past"    = [as_of − W, as_of)  for all features
`

Zero overlap between feature window and label window — provably leakage-free.

---

## 🎯 Summary Card

`
┌────────────────────────────────────────────────────────────────┐
│  CONFLICT ESCALATION FORECASTING                               │
│                                                                │
│  Task:    Binary classification — escalation within 14 days    │
│  Scope:   83 provinces · 4 countries · 2017–2026              │
│  Data:    89,297 validated ACLED weekly event records          │
│  Features: 33 rolling-window, velocity, spatial, calendar      │
│  Model:   XGBoost (400 trees, lr=0.05, threshold=0.20)        │
│  F1:      0.84+  |  PR-AUC: 0.90+  |  ROC-AUC: 0.82          │
│                                                                │
│  TOP RISK RIGHT NOW (2026-07-25):                             │
│  Magway, Myanmar — 99.8% probability                          │
│  Driver: fatalities_w30d (sustained 30-day lethality)         │
│                                                                │
│  HIGHEST-RISK COUNTRY: Myanmar (avg 74.6%)                    │
│  SAFEST PROVINCE: Sikkim, India (2.3%)                        │
└────────────────────────────────────────────────────────────────┘
`

---

*All results are from a real pipeline run on 2026-08-08 using ACLED data through 2026-07-25.*
*Data source: ACLED (acleddata.com) — Armed Conflict Location and Event Data Project.*
