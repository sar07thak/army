# 📚 Learning Guide — Conflict Escalation Forecasting
### From Zero to Expert: Complete Understanding of What This Project Is, How It Works, and Where It Can Be Used

---

## 🧭 Table of Contents

1. [The Problem — Why Does This Exist?](#1-the-problem)
2. [What This Project Does — In Plain English](#2-what-this-project-does)
3. [The Data — What We Start With](#3-the-data)
4. [Core Concepts You Need to Know](#4-core-concepts)
5. [The Full Pipeline — Step by Step](#5-the-full-pipeline)
6. [Feature Engineering — Teaching the Model About Conflict](#6-feature-engineering)
7. [How the Labels Are Made](#7-label-engineering)
8. [Train/Validation/Test Split — The Right Way](#8-the-split)
9. [The Machine Learning Models](#9-the-models)
10. [SHAP — Making the Model Explain Itself](#10-shap-explainability)
11. [Risk Visualization and Output](#11-visualization-and-output)
12. [Live Forecasting](#12-live-forecasting)
13. [The Code Architecture](#13-code-architecture)
14. [Where This Can Be Used in the Real World](#14-real-world-applications)
15. [Advanced Topics](#15-advanced-topics)
16. [Common Questions and Gotchas](#16-faqs)

---

## 1. The Problem

### Why Conflict Analytics Matter

Conflict monitoring today is **reactive** — analysts see violence spike on the news and scramble to respond. By the time the response is mobilized, people are already displaced, supply chains are already broken, and opportunities to prevent escalation are already lost.

What the world needs is a system that answers: **"Where is violence about to get worse in the next two weeks — before it happens?"**

That's exactly what this project builds.

### The Core Question

> **Which provinces (admin-1 regions) across India, Myanmar, Sudan, and South Sudan are most likely to experience a significant rise in violent events or fatalities within the next 14 days?**

This question is answered for **83 provinces** every time you run the pipeline, using patterns learned from **9+ years of historical conflict data** (2017–2026).

---

## 2. What This Project Does

Think of it as a **conflict weather forecast**. Just as weather models use historical temperature, pressure, and wind data to predict tomorrow's weather, this system uses historical violence patterns to predict tomorrow's conflict risk.

### The Output — What You Get

After running the pipeline, you receive:

| Output | Description |
|---|---|
| `forecast_next_14_days.csv` | Per-province risk probability + risk category + top 3 SHAP drivers |
| `forecast_risk_map.html` | Interactive map — click any province for full risk details |
| `risk_summary.md` | Human-readable report of highest-risk and safest regions |
| `reports/shap/` | 21 explainability plots showing **why** each prediction was made |
| `reports/figures/` | 11 dashboard charts (country trends, hotspots, temporal evolution) |

### The Input — What You Start With

A single CSV file from **ACLED** (Armed Conflict Location and Event Data Project). This file contains one row per week per province per event type, with columns like:
- `event_date` — the week
- `country` / `admin1` — the location (province name)
- `event_type` — e.g., Battles, Protests, Violence against civilians
- `events` — count of incidents that week
- `fatalities` — number of people killed

---

## 3. The Data

### ACLED — The Source

**ACLED** (acleddata.com) is the world's most widely used open source conflict dataset. It records every reported political violence event globally, going back decades. Researchers at the UN, World Bank, NGOs, governments, and academic institutions all use it.

### What is Covered (Your Run)

| Country | Avg Risk Score | Typical Conflict Types |
|---|---|---|
| **Myanmar** | 0.746 (highest) | Armed conflict post-2021 coup, airstrikes, battles |
| **India** | 0.532 | Insurgencies, communal violence, Maoist activity |
| **South Sudan** | 0.532 | Inter-communal violence, political battles |
| **Sudan** | 0.426 | Civil war, Darfur conflict |

### Dataset After Cleaning (Your Actual Numbers)

| Stage | Rows | Details |
|---|---|---|
| Raw CSV | 108,945 | As loaded from ACLED file |
| After country filter | 89,297 | 19,648 removed (Ukraine, Indian Ocean, Mediterranean) |
| Features table | 30,233 | Rows × 37 features |
| Labeled features | 30,104 | 129 dropped (incomplete future windows) |
| Train split | 19,801 | Jan 2017 → Sep 2023 |
| Validation split | 5,099 | Sep 2023 → Feb 2025 |
| Test split | 5,204 | Feb 2025 → Jul 2026 |

---

## 4. Core Concepts

### 4.1 Binary Classification

The prediction task is **binary classification**: predict **1** (escalation in next 14 days) or **0** (no escalation).

The model also outputs a **probability** (e.g., 0.78) — how confident it is. This is more useful than a bare 0/1 because you can rank provinces by urgency.

### 4.2 Rolling Windows

A rolling window is a fixed time period sliding through history. The **7-day window** at a given date captures all events in the 7 days *before* that date.

`
Timeline: [Jan1 Jan2 Jan3 Jan4 Jan5 Jan6 Jan7] → [Jan8 = prediction date]
                         7-day window ─────────┘
`

Different window lengths (7d, 14d, 30d) capture different timescales of the pattern.

### 4.3 Temporal Leakage — The Critical Danger

**Leakage** means the model accidentally sees future information during training. This is catastrophically wrong:

- BAD: Feature = "events in the next 7 days" (uses future data)
- GOOD: Feature = "events in the past 30 days" (uses only past data)

This project uses half-open intervals `[date − W, date)` — current day excluded — and includes automated tests to prove no leakage.

### 4.4 SHAP Values

SHAP answers: **"Why did this province get a 0.99 risk score?"**

Example for Magway, Myanmar (risk = 0.998):
`
Base rate:              0.50
+ fatalities_w30d:      +3.23  (lots of recent deaths)
+ events_w7d:           +1.22  (active this week)
+ velocity:             +0.84  (things are getting worse)
= Final prediction:     0.998
`

Every single prediction is explainable — crucial for decision-makers who need to justify action.

---

## 5. The Full Pipeline

`
data/raw/*.csv
    │  Stage 1: INGEST  → validate, canonicalize
    ▼
cleaned_events (89,297 × 14)  +  district_master (83 × 9)
    │  Stage 2: FEATURES  → rolling windows per province per date
    ▼
features (30,233 × 37)
    │  Stage 3: LABELS  → future-only escalation labels
    ▼
labeled_features (30,104 × 38)
    │  Stage 4: SPLIT  → strict chronological cut
    ▼
train (19,801)  ·  val (5,099)  ·  test (5,204)
    │  Stage 5: TRAIN  → XGBoost (LightGBM fallback on Python 3.14)
    ▼
escalation_lgbm.pkl  +  manifest.json
    │  Stage 6: COMPARE  → XGBoost vs LGBM, winner selection
    ▼
escalation_best.pkl  +  model_comparison.json
    │  Stage 7: EXPLAIN  → SHAP on held-out test window
    ▼
reports/shap/ (21 plots)  +  shap_summary.md
    │  Stage 8: VISUALIZE  → risk map + dashboards
    ▼
risk_map.html  +  11 figures  +  risk_summary.md
    │  Stage 9: FORECAST  → live 14-day predictions
    ▼
forecast_next_14_days.csv  +  forecast_risk_map.html
`

---

## 6. Feature Engineering

All 33 features use only information from **strictly before** the prediction date — no leakage.

### Volume Features — How Much Violence?
| Feature | Window | Meaning |
|---|---|---|
| events_w7d | 7 days | Events last week |
| events_w14d | 14 days | Events last 2 weeks |
| events_w30d | 30 days | Events last month |
| fatalities_w7d/14d/30d | 7/14/30 days | Deaths per window |

### Velocity Features — Is It Getting Worse?
| Feature | Meaning |
|---|---|
| velocity_events_w30d | (events last 30d) − (events 30d before that) |
| velocity_fatalities_w30d | Same but for fatalities |

Positive velocity = violence is accelerating — a strong escalation signal.

### Statistical Features — How Volatile?
| Feature | Meaning |
|---|---|
| fat_mean_w14d/30d | Average daily fatalities (intensity) |
| fat_std_w14d/30d | Std deviation (spikiness / unpredictability) |

### Spatial Spillover — Are Neighbours Active?
| Feature | Meaning |
|---|---|
| spillover_w14d | Events across the 3 nearest same-country provinces (haversine distance) |

Violence spreads. When neighbours heat up, local risk rises too.

### Other Features
- **persistence_w7d** — days with at least 1 event in last 7 days
- **entropy_w7/14/30d** — Shannon entropy of event types (conflict diversity)
- **days_since_event** — recency sentinel (999 if no history)
- **month / day_of_week** — seasonality
- **country_code / admin1_code / geo_unit_code** — province-level baseline risk

---

## 7. Label Engineering

**escalation = 1** if, in the 14 days *after* the prediction date:

- Condition A: `future_events ≥ 3` AND `future_events ≥ 1.5 × trailing_30d_median`
- OR Condition B: `future_fatalities ≥ 5`
- Fallback: `future_events ≥ 5` (no history)

**Class balance:** 67.7% positive, 32.3% negative.

This is expected — these are high-conflict regions where escalation is the norm. The model handles imbalance via threshold tuning (operating threshold 0.20) and scale_pos_weight.

---

## 8. The Split

Strictly chronological — never shuffled:

| Split | Rows | Date Range | Positive % |
|---|---|---|---|
| Train | 19,801 | 2017-01-07 → 2023-09-02 | 64.2% |
| Validation | 5,099 | 2023-09-09 → 2025-02-01 | 74.4% |
| Test | 5,204 | 2025-02-08 → 2026-07-11 | 74.3% |

The **test set is never seen** during training or hyperparameter tuning.

---

## 9. The Machine Learning Models

### XGBoost — How It Works

1. Start with a naive prediction for all provinces
2. Build a decision tree to predict the model's current errors
3. Add that tree (scaled by learning rate 0.05) to improve predictions
4. Repeat 400 times — each tree corrects the prior ensemble's mistakes
5. Final output = sum of all 400 trees

### Key Hyperparameters
`
n_estimators = 400       # number of trees
max_depth = 6            # depth of each tree
learning_rate = 0.05     # step size per tree
subsample = 0.8          # random fraction of training rows per tree
colsample_bytree = 0.8   # random fraction of features per tree
`

### Performance on Your Data
| Metric | Value |
|---|---|
| F1-score (validation) | 0.84+ |
| PR-AUC | 0.90+ |
| Operating threshold | 0.20 (tuned from sweep) |

XGBoost beat all 4 baselines (always-1, majority, persistence, heuristic).

---

## 10. SHAP Explainability

Top 10 global feature drivers (from your run's held-out test set):

| Rank | Feature | Mean |SHAP| | Insight |
|---|---|---|---|
| 1 | fatalities_w30d | 0.54 | Sustained recent deaths = strongest escalation signal |
| 2 | velocity_events_w30d | 0.33 | Acceleration matters more than raw count |
| 3 | events_w30d | 0.28 | 30-day volume beats short-term spikes |
| 4 | events_w7d | 0.20 | Current-week activity |
| 5 | admin1_code | 0.19 | Province identity (known-risk baselines) |
| 6 | velocity_fatalities_w30d | 0.18 | Fatality acceleration |
| 7 | events_w14d | 0.16 | 2-week volume |
| 8 | month | 0.14 | Conflict has seasons |
| 9 | spillover_w14d | 0.14 | Neighbourhood contagion is real |
| 10 | fatalities_w14d | 0.10 | 2-week deaths |

**Key finding:** 30-day windows dominate the signal. Sustained patterns (month-long) matter much more than single-week spikes.

---

## 11. Visualization and Output

### Risk Map
Open `reports/maps/forecast_risk_map.html` in any browser. Click any province marker to see:
- Risk probability (0–1)
- Risk category (Low / Medium / High / Critical)
- Top 3 SHAP drivers explaining the prediction
- Recent events and fatalities

### Risk Categories
| Category | Probability | Action |
|---|---|---|
| 🔴 Critical | ≥ 0.75 | Priority alert — immediate attention |
| 🟠 High | ≥ 0.50 | Monitor closely |
| 🟡 Medium | ≥ 0.25 | Watch |
| 🟢 Low | < 0.25 | Background monitoring |

---

## 12. Live Forecasting

The forecast stage scores all 83 provinces at their most recent state without any retraining.

**To update with new ACLED data:**
`ash
# 1. Drop new CSV into data/raw/
# 2. Run only what changed:
python run_pipeline.py --stage ingest
python run_pipeline.py --stage features
python run_pipeline.py --stage forecast
# → New risk scores in ~60 seconds, no retraining needed
`

---

## 13. Code Architecture

`
src/
├── data_loader.py      → CSV loading, canonicalization, saving
├── data_validation.py  → 9 validation rules
├── feature_engineer.py → 33 rolling-window features
├── label_engineer.py   → 14-day escalation labels
├── split.py            → Chronological split
├── models.py           → XGBoost training, metrics, baselines
├── pipeline.py         → train_stage + compare_stage
├── explainability.py   → SHAP plots and summaries
├── visualization.py    → folium map + plotly dashboards
├── forecast.py         → Live scoring on latest features
├── exceptions.py       → Custom error hierarchy
└── logging_config.py   → Rotating log setup
`

**Design principles:**
- Config-driven (no magic numbers in src/)
- Leakage-safe by construction
- Deterministic + reproducible (seed 42, pinned deps)
- Fully tested (243 tests, 96.5% coverage)

---

## 14. Real-World Applications

| Domain | How to Use It |
|---|---|
| **Humanitarian ops** | Pre-position aid, staff, and supplies before escalation |
| **Media** | Deploy correspondents to where news will happen next |
| **Government / Foreign Affairs** | Issue travel advisories; adjust diplomatic pressure |
| **Military / Peacekeeping** | Prioritize patrol areas and resource allocation |
| **Academic Research** | Study spatial contagion, seasonality, and escalation dynamics |
| **Insurance** | Dynamic political risk pricing for operations in conflict zones |

### Adaptable Pattern

The same architecture (rolling features + future labels + chronological split + SHAP) applies to any time-series prediction problem: epidemics, election violence, natural disasters, supply chain disruption, cyber-attack waves.

---

## 15. Advanced Topics

### Why Threshold = 0.20?
With 68% positive labels, the optimal F1 point is below 0.5. A sweep from 0.10 to 0.90 finds that 0.20 maximizes validation F1. Using 0.5 would miss many true escalation events.

### PR-AUC vs ROC-AUC
With imbalanced classes, ROC-AUC can be misleadingly high (predicting all 1s gives good ROC). PR-AUC is stricter — it penalizes models with poor precision. That's why PR-AUC is the headline metric (0.90+).

### Spillover via Haversine Distance
The spillover feature finds each province's 3 nearest same-country neighbours using the **haversine formula** (accurate great-circle distance on a sphere). This is spatial ML at province scale.

### Scale_pos_weight
`
scale_pos_weight = 9732 (negatives) / 20372 (positives) = 0.478
`
XGBoost gives each negative sample approximately 2× the gradient weight, making the model less biased toward always predicting 1.

---

## 16. FAQs

**Q: Why XGBoost over LSTM or Transformers?**
> Tabular weekly data with ~30K rows → gradient boosted trees win consistently. Deep models need far more data and lose interpretability. XGBoost + SHAP gives better results AND full explainability.

**Q: Why 14 days?**
> Long enough to be actionable (resources can be mobilized), short enough that historical patterns still predict well. 7d and 30d are planned future extensions.

**Q: What does the model NOT capture?**
> Sudden external shocks — coups, peace treaties, foreign interventions, elections. These rupture the historical patterns the model relies on.

**Q: How often should I update?**
> ACLED publishes weekly. Drop a new CSV weekly and run ingest → features → forecast. Model retraining only needed periodically (e.g., annually) to incorporate new history.

**Q: Can I add more countries?**
> Yes — add the country to config.py COUNTRIES list and download the corresponding ACLED data. Everything else adapts automatically.

---

*Guide written based on actual pipeline run on 2026-08-08 with ACLED data covering 4 countries, 83 provinces, 89,297 validated event records.*
