# PRD — Conflict Escalation Forecasting (South Asia)

**Product Requirements Document v1.0**

| Field | Value |
|---|---|
| **Project** | Conflict Escalation Forecasting — District-Level Early-Warning System |
| **Event** | ML Bubble 2026, AIT Pune — Defense & National Security Track |
| **Version / Status** | v1.0 — Approved for build |
| **Date** | August 7, 2026 |
| **Source doc** | `Conflict_Escalation_Forecasting_Project_Guide.md` |
| **Decisions locked** | Region: **South Asia** (India, Pakistan, Afghanistan, Myanmar) · Deliverable: **Hackathon package** (notebooks + model + SHAP + maps) · Framing: **Binary classification, 14-day horizon** |

---

## 1. Executive Summary

**One-line pitch:**
> *"A district-level early-warning model that forecasts conflict escalation risk over a rolling 14-day window, using engineered temporal features from ACLED event data."*

Most student "defense ML" projects classify conflict events *after* the fact ("was this a battle or a protest?"). This project predicts **where and when** conflict activity is likely to **escalate in the next 14 days**, framed as a spatiotemporal forecasting problem rather than a one-shot classification problem. That framing, a rolling-window feature engine, a time-respecting evaluation protocol, and SHAP explainability together demonstrate the maturity expected of the "advanced" track (SE/TE-BE).

**What we are building:** A reproducible ML pipeline that ingests ACLED event data for South Asia, engineers per-district rolling temporal features, trains a gradient-boosted classifier (XGBoost/LightGBM) to predict "escalation yes/no in the next 14 days," explains predictions with SHAP, and visualizes district-level risk on a map. Delivered as a clean GitHub repo with notebooks, scripts, a trained model artifact, a metrics report, and a README that tells the full story.

---

## 2. Problem Statement

Conflict monitoring today is largely **reactive**: analysts see a spike and respond. The core problem this product solves is turning historical event patterns into a **forward-looking, district-level risk signal** that answers:

> *"Which districts in South Asia are most likely to see conflict escalation (a significant rise in violent events or fatalities) within the next 14 days?"*

Escalation is inherently **spatiotemporal**: violence clusters in space (neighboring districts spill over) and time (past violence is the strongest predictor of future violence). The product exploits exactly this structure with rolling-window temporal features and, as a stretch goal, neighboring-region spillover features.

---

## 3. Goals & Objectives

### 3.1 Primary goals
1. **G1 — Working forecast model:** A binary classifier (14-day horizon) with useful predictive skill above baselines, evaluated with a time-based split and reported with precision / recall / F1 (not accuracy alone).
2. **G2 — Credible methodology:** Rolling-window feature engineering + LightGBM/XGBoost (matching ACLED's own production approach, CAST) + SHAP explainability, so the Aug 16 explanation round is defensible.
3. **G3 — Demo-grade visualization:** A choropleth/scatter map of predicted risk per district for the next window, plus SHAP summary/waterfall plots.
4. **G4 — Reproducible, submittable package:** Clean repo (structure from the guide), `requirements.txt`, a README with problem statement, data attribution, methodology, and metrics, submitted before the deadline.

### 3.2 Non-goals (explicitly out of scope for v1)
- Real-time / daily auto-refreshing predictions
- A global or multi-region model
- Causal inference (we predict, we do not claim causes)
- Production deployment, APIs, or a hosted dashboard
- GDELT news-tension features (possible stretch only if time allows)

---

## 4. Success Metrics (KPIs)

### 4.1 Model quality (primary)
| Metric | Target | Why |
|---|---|---|
| **F1 (positive class)** | ≥ 0.55 on held-out test window | Escalation is rare; F1 is the headline metric |
| **Precision** | ≥ 0.50 | Penalize false alarms (an early-warning product must not cry wolf) |
| **Recall** | ≥ 0.50 | Penalize missed escalations (the worst failure mode) |
| **AUC-PR** | ≥ 0.65 | Better than AUC-ROC for imbalanced data |
| **Lift vs majority baseline** | ≥ 1.5× (F1 and/or precision at fixed recall) | Proves skill beyond "always predict no" |
| **Calibration** | Brier score reported; probability bins roughly match observed rates | Risk-score credibility if judges probe probabilities |

> **Reporting rule:** never report accuracy alone — with ~5–10% positive class it will look artificially high.

### 4.2 Process / product quality (secondary)
| Metric | Target |
|---|---|
| End-to-end runnable | Full pipeline re-runs from a single documented command / notebook chain |
| Reproducibility | Fixed random seeds; versioned data snapshot noted in README |
| Documentation | README covers problem, data + attribution, methodology, metrics, how to re-run |
| Demo | One risk map + one SHAP explanation ready for the PPT |

---

## 5. Stakeholders & User Personas

| Persona | Who | Needs |
|---|---|---|
| **Conflict Analyst** (primary user of the *idea*) | Early-warning desk at an NGO / defense research org | "Which districts need attention in the next 2 weeks?" — ranked list + map + reasons |
| **Hackathon Judge** (primary audience of the *submission*) | SE/TE-BE advanced-track evaluators | Clear problem framing, sound methodology, honest metrics, visible results |
| **Teammates** | 2–4 students | Clean, separable modules; clear task ownership |
| **Model Explainer** (Aug 16 round) | One teammate defends methodology | SHAP insights + a simple narrative ("rising actor diversity + fatality volatility flagged this district") |

### Key user stories
- **As a conflict analyst,** I want a ranked list of at-risk districts for the next 14 days, so I can triage monitoring resources.
- **As a conflict analyst,** I want to know *why* a district was flagged, so I can sanity-check the warning (SHAP).
- **As a judge,** I want to see a time-based evaluation with precision/recall/F1, so I trust the numbers.
- **As a teammate,** I want the pipeline split into small scripts/notebooks, so we can parallelize work.

---

## 6. Scope

### 6.1 In scope (v1)
- **Data:** ACLED events for South Asia — **India, Pakistan, Afghanistan, Myanmar** — covering **3–5 years** (recommend ~Jan 2021 – Jun 2026 or the last full 3 years available).
- **Geo unit:** Admin-2 (district) where data density allows; fall back to Admin-1 (state/province) for sparse units. **Decision locked: Admin-2 default, with a minimum-events filter** (e.g., ≥ 5 events in the study period) to drop silent districts.
- **Target:** Binary label — *will this district see escalation in the next 14 days?* (definition in §11.2).
- **Features:** Rolling-window temporal features per district (§11.3), including spillover as a stretch.
- **Model:** XGBoost vs LightGBM baseline comparison; pick better; tune lightly.
- **Evaluation:** Strict time-based split; precision/recall/F1/AUC-PR/Brier; baselines reported.
- **Explainability:** SHAP summary plot + at least one force/waterfall example.
- **Visualization:** Plotly choropleth (or Folium scatter) of next-window risk; event-trend line charts in EDA.
- **Delivery:** Repo per guide structure, `requirements.txt`, README, model artifact, metrics report, PPT input material.

### 6.2 Out of scope (v1 — explicitly)
- Multi-region/global model; country-month aggregate modeling (we are district-window level)
- Streaming updates / automated re-training on a schedule
- Web dashboard or API (switched to a stretch; revisit only if the baseline is done early)
- UCDP/GDELT feature fusion (cross-validation idea for the write-up only)
- Deep learning / neural sequence models (overkill for tabular window features)

---

## 7. Functional Requirements

Priorities: **M** = Must (blocker if absent), **S** = Should, **C** = Could (stretch).

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | **Data ingestion.** Fetch ACLED data for the 4 South Asia countries via the ACLED API (paged) or the Data Export Tool CSV; cache raw CSVs in `data/raw/`; never commit raw data (`.gitignore`). | M |
| FR-2 | **Data validation.** Schema check on required fields (`event_date`, `event_type`, `country`, `admin1`, `admin2`, `latitude`, `longitude`, `fatalities`, `actor1`); drop rows with missing country/admin1/date; parse dates; dedupe by event ID. | M |
| FR-3 | **Geo aggregation.** Aggregate events to district-day; build a district master table (district → admin1 → country). | M |
| FR-4 | **Feature engineering.** Compute rolling-window features per district per date (spec in §11.3); output `data/processed/features.parquet` (or CSV). | M |
| FR-5 | **Label construction.** Build the 14-day escalation label per district-date row (§11.2), avoiding leakage (labels must use only *future* data). | M |
| FR-6 | **Train/val/test split.** Strict chronological split; document exact cut dates; no random splitting anywhere in modeling. | M |
| FR-7 | **Model training.** Train LightGBM (primary) and XGBoost (baseline) on identical folds; save best model to `models/`. Handle class imbalance (class weight / `scale_pos_weight`). | M |
| FR-8 | **Evaluation.** Compute precision/recall/F1/AUC-PR/Brier on the held-out test window; produce `reports/model_metrics.md`; include majority-class and naive-persistence baselines. | M |
| FR-9 | **Explainability.** SHAP summary plot → `reports/shap_summary.png`; one worked example (waterfall/force) of a flagged district. | M |
| FR-10 | **Risk map.** Plotly/Folium map of predicted escalation probability by district for the final test window → `reports/risk_map.html`. | M |
| FR-11 | **Reproducibility.** `requirements.txt` (pinned or major-version-pinned), fixed seeds, and a README section "How to re-run the pipeline". | M |
| FR-12 | **EDA report.** Notebook 01 documents data coverage, event-type mix, top hotspots, and temporal trends (this feeds the PPT). | S |
| FR-13 | **Spillover features.** Adjacent-district event counts (admin-1 neighbor aggregate) as extra features. | C |
| FR-14 | **Threshold tuning.** Report precision/recall at a chosen operating threshold (e.g., max F1) vs default 0.5, and document the choice. | S |
| FR-15 | **Multi-horizon check.** Note 7-day vs 30-day results in the write-up (re-train optional). | C |

---

## 8. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | Full training run completes in ≤ 30 minutes on a standard student laptop (16 GB RAM). Data pull is a one-time cost. |
| **Usability** | A fresh teammate can run the pipeline from the README in ≤ 15 minutes. Notebooks execute top-to-bottom with a single kernel. |
| **Reliability** | Pipeline fails loudly with clear error messages on missing columns/dates; no silent NaNs in features. |
| **Determinism** | Fixed `random_state` everywhere; LightGBM/XGBoost with fixed seeds; document any platform-dependent variance. |
| **Compliance** | ACLED attribution line required in README and PPT: *"Data: ACLED, acleddata.com."* Respect the 5,000-row API response cap via pagination. |
| **Security / privacy** | No secrets in the repo — API key read from environment variable (`.env`, gitignored). |
| **Maintainability** | Separation: `data_loader.py` → `feature_engineer.py` → `train.py` → `evaluate.py`; config constants (windows, thresholds, dates) in one place (a `config.py` or top-of-file constants). |

---

## 9. Data Requirements

### 9.1 Primary source — ACLED
- **Access:** free registration at [acleddata.com](https://acleddata.com) → myACLED account.
- **Pull paths (pick one, do not do both):**
  1. **Data Export Tool** (fastest): filter *South Asia countries* → event types → date range → download CSV into `data/raw/`.
  2. **ACLED API** (more scriptable): paginated GET calls (5,000 rows/call) via `requests`; store key in env var.
- **Required fields per event:** `event_date`, `event_type`, `country`, `admin1`, `admin2`, `latitude`, `longitude`, `fatalities`, `actor1` (and `event_id_cnty` for dedupe).

### 9.2 Scope parameters
| Parameter | Value |
|---|---|
| Countries | India, Pakistan, Afghanistan, Myanmar |
| Date range | ~36–60 months (recommend latest 3 full years + current partial year; a longer range is a stretch) |
| Event types | All (battle, explosion/remote violence, violence against civilians, protest, riot, strategic development) — keep `event_type` as a feature, don't pre-filter |
| Geo unit | Admin-2 default, Admin-1 fallback for sparse units |
| Expected volume | Rough estimate: **150k–400k events** across the 4 countries over 3–5 years; **~60k–120k district-date rows** after feature construction (South Asia has ~1,600 admin-2 units) — comfortably handles XGBoost/LightGBM on a laptop |

### 9.3 Backup / supplementary sources
| Source | Use | When |
|---|---|---|
| HDX ACLED mirror ([data.humdata.org/organization/acled](https://data.humdata.org/organization/acled)) | Weekly aggregated CSVs per country | If API/export setup is slow |
| UCDP GED ([ucdp.uu.se/downloads](https://ucdp.uu.se/downloads)) | Cross-validation / write-up credibility | Stretch only |
| GDELT ([gdeltproject.org](https://www.gdeltproject.org/)) | News-tension feature | Stretch only, likely out of scope |

### 9.4 Known data-quality caveats (call out in EDA)
- **Myanmar post-2021:** coverage shifts during the civil-war period; treat as a feature (country) rather than dropping.
- **Admin-2 naming inconsistencies** across years/sources — normalize with a manual mapping table.
- Protests are much more frequent than battles → label design must not treat every event as "conflict."

---

## 10. Technical Architecture

### 10.1 Pipeline (data → decision)

```
ACLED API / Export CSV
        │
        ▼
src/data_loader.py ──► data/raw/*.csv      (FR-1, cached, gitignored)
        │
        ▼
src/feature_engineer.py ──► data/processed/features.parquet   (FR-2..FR-5)
        │                          (rolling features + labels, leakage-safe)
        ▼
src/train.py ──► models/escalation_lgbm.pkl / xgb.pkl          (FR-6, FR-7)
        │
        ├──► src/evaluate.py ──► reports/model_metrics.md      (FR-8, FR-14)
        ├──► SHAP plots ──► reports/shap_summary.png           (FR-9)
        └──► Risk map ──► reports/risk_map.html                (FR-10)
```

### 10.2 Repo structure (from the guide, extended)

```
conflict-escalation-forecasting/
├── data/
│   ├── raw/                  # ACLED CSVs (gitignored)
│   └── processed/            # features.parquet + labels
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── config.py             # single source of truth for constants
│   ├── data_loader.py        # ACLED API / CSV ingestion + validation
│   ├── feature_engineer.py   # rolling-window features + labels
│   ├── train.py              # time split, LGBM/XGB, imbalance handling
│   └── evaluate.py           # metrics, baselines, SHAP, risk map
├── models/
│   └── escalation_lgbm.pkl   # best model artifact
├── reports/
│   ├── model_metrics.md
│   ├── shap_summary.png
│   └── risk_map.html
├── .env.example              # ACLED_API_KEY=...
├── .gitignore                # data/raw/, models/, .env, __pycache__
├── README.md                 # problem, data+attribution, methodology, metrics, re-run
└── requirements.txt
```

### 10.3 Environment
- **Python 3.10–3.12.** Core deps: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `plotly`, `folium`, `requests`, `python-dotenv`, `jupyter`.
- Use a virtualenv; commit only `requirements.txt`.

---

## 11. ML Model Design

### 11.1 Task
Per district-date row: predict **P(escalation within next 14 days)** → binary label. Prediction unit: **district (admin-2), 14-day horizon**.

### 11.2 Label definition (must be leakage-safe — computed only from the *future*)
**Escalation = 1** if, over the next 14 days, **either**:
- event count ≥ 3 **and** ≥ 1.5× the district's trailing 30-day rolling median event count, **or**
- fatalities ≥ 5.

Fallback for districts with empty history: escalation = 1 if next-14-day event count ≥ 5 (absolute rule).
> Configurable constants in `src/config.py`. Document the chosen thresholds and run a quick sensitivity check in notebook 03 (the "definition of escalation" is a judge-favorite question).

### 11.3 Feature table (computed **as of each date**, using only past data)
| Feature group | Features | Notes |
|---|---|---|
| Volume | event count (7/14/30d), fatality count (7/14/30d) | Raw counts + log1p |
| Change ("velocity") | events_last_7d − events_prior_7d; same for 14/30 | Rate of change |
| Diversity | Shannon entropy over event types (7/14/30d) | Richer mix → escalation signal |
| Actors | distinct armed actors (14/30d) | Rising actor diversity = red flag |
| Recency | days since last event in district | Onset detector |
| Volatility | rolling mean/std of daily fatalities (14/30d) | Risk "volatility" |
| Persistence | count of days with events in last 7d | Chronic vs acute |
| Spillover (C) | event counts in adjacent districts (admin-1 neighbors, 14d) | Geographic diffusion |
| Calendar | month, day-of-week | Seasonality, though use with care |

### 11.4 Model selection
- **Primary:** LightGBM (matches ACLED's CAST production approach — credible talking point).
- **Baseline:** XGBoost on identical features/folds; pick the better by validation F1.
- **Why GBDT:** tabular rolling features, handles skewed counts, fast on a laptop, built-in feature importance, SHAP-friendly.
- **Class imbalance:** `scale_pos_weight`/`class_weight` from train-set class ratio; optionally a small amount of threshold tuning (FR-14).

### 11.5 Splitting protocol (non-negotiable)
1. Sort all district-date rows by date.
2. **Train** = oldest ~70–80%, **validation** = next ~10–15%, **test** = newest ~10–15% (e.g., last 6–9 months).
3. No shuffling, no random CV, no oversampling across the time boundary. (Random split = future leakage = automatic credibility loss.)
4. Optional robustness check: retrain the model on a second shifted split (walk-forward with 2 windows) if time allows.

### 11.6 Evaluation protocol
- Metrics: **precision, recall, F1, AUC-PR, Brier score** on the held-out test window.
- Baselines to beat: (a) always-predict-majority, (b) naive persistence (predict "escalation" for districts that escalated last period), (c) event-count-threshold heuristic.
- Report at default 0.5 threshold **and** at max-F1 operating threshold.

---

## 12. UI / Visualization Requirements (demo layer)

These are report artifacts, not a web app (v1):

| Artifact | Requirement | File |
|---|---|---|
| **Risk map** | Choropleth of South Asia districts colored by predicted escalation probability for the test window; hover shows district name, top-3 SHAP drivers | `reports/risk_map.html` |
| **SHAP summary** | Beeswarm of top-15 features | `reports/shap_summary.png` |
| **SHAP example** | Waterfall/force plot for one flagged district with the one-line narrative | include in notebook 03 + PPT |
| **Trend charts (EDA)** | Event counts over time per country; event-type mix; top hotspot districts | notebook 01 |

**Demo narrative for the PPT:** "Rising actor diversity + fatality volatility flagged this district" — pair one real flagged district with its SHAP waterfall.

---

## 13. Milestones & Release Plan

Deadline context: build window is tight (guide's timeline) with the online explanation round on **Aug 16**. Milestones are hours-based, not calendar-heavy.

| # | Milestone | Output / Done-when | Est. |
|---|---|---|---|
| M0 | **Setup + data pull** | ACLED registered; South Asia data (3–5 yr) in `data/raw/`; repo skeleton + venv + `.gitignore`; notebook 01 started | ~2–3 h |
| M1 | **EDA** | Notebook 01 complete: coverage, hotspots, event mix, trends; decisions confirmed (sparse-district filter, admin level) | ~2 h |
| M2 | **Features + labels** | `feature_engineer.py` produces `features.parquet`; leakage check (labels only future, features only past) | ~3 h |
| M3 | **Baseline model** | Time split; LGBM + XGB trained; validation F1/precision/recall; best model saved | ~2–3 h |
| M4 | **Evaluation + explainability** | `evaluate.py` metrics report vs baselines; SHAP summary + example | ~2 h |
| M5 | **Visualization + wrap** | `risk_map.html`; README (problem, attribution, methodology, metrics, re-run); `.gitignore` check; **submit** | ~2 h |
| M6 | **Explanation-round prep** | PPT/PDF slides; practice narrative; SHAP story per slide; attribution line on final slide | ~1–2 h |

**Hard exit criteria for M3:** if validation F1 < 0.40 by deadline, cut scope (drop spillover, simplify features, use a rule-based threshold model) rather than chase tuning.

---

## 14. Risks & Mitigations

| # | Risk | Likelihood / Impact | Mitigation |
|---|---|---|---|
| R1 | ACLED access/setup friction (registration delay, API quirks) | Med / High | Start registration immediately (M0); fall back to HDX mirror CSVs; cache raw data day 1 |
| R2 | Escalation class too rare → poor recall | High / Med | Tune `scale_pos_weight`; absolute-rule fallback label; report AUC-PR; threshold tuning; if F1 < 0.40, pivot framing to risk scoring |
| R3 | Myanmar post-2021 coverage gaps distort features | Med / Med | Keep `country` as a feature; treat gaps as a documented data caveat in EDA |
| R4 | Admin-2 naming inconsistencies across years | Med / Low | Normalization mapping table in `data_loader.py`; fall back to admin-1 for problematic units |
| R5 | Time runs out on a dependency (visualization, SHAP) | Med / Med | Build order is risk-ordered: model + metrics first (M3–M4), viz last (M5); viz is skippable without harming the core |
| R6 | Laptop resource limits | Low / Med | Keep to 3–5 yr data; `float32` + feature pruning; ≤ 30-min training budget enforced |
| R7 | Accidental leakage (labels/features using future data) | Med / High | Explicit leakage test in M2; code review by teammate; time-based split enforced in `train.py` |
| R8 | Judge questions "why should I trust this?" | High / Med | SHAP narrative + honest baseline comparison + Brier score = pre-built answer |

---

## 15. Acceptance Criteria (Definition of Done)

- [ ] Data for all 4 countries is in `data/raw/` with schema validated (FR-1, FR-2).
- [ ] `features.parquet` exists with documented features and leakage-safe labels (FR-3..FR-5).
- [ ] Strict chronological split documented with exact cut dates; no random splitting anywhere (FR-6).
- [ ] Best model saved as `models/escalation_lgbm.pkl` with fixed seed (FR-7, NFR determinism).
- [ ] `reports/model_metrics.md` lists precision/recall/F1/AUC-PR/Brier vs all three baselines at two thresholds (FR-8, FR-14).
- [ ] `reports/shap_summary.png` + one worked example exist (FR-9).
- [ ] `reports/risk_map.html` renders district risk for the test window (FR-10).
- [ ] `requirements.txt`, `.gitignore`, `.env.example` present; raw data and secrets not committed (FR-11, NFR security).
- [ ] README contains: problem statement, dataset source + ACLED attribution, methodology, metrics, and re-run instructions (FR-11).
- [ ] Full pipeline re-runs end-to-end from a fresh venv in ≤ 15 minutes of human time (NFR usability).
- [ ] One-line demo narrative + one SHAP screenshot ready for the PPT (M6).

---

## 16. Open Questions (decide during M1, before M2)

| # | Question | Default if undecided |
|---|---|---|
| Q1 | Exact escalation thresholds (counts, multipliers, fatality cutoff) | §11.2 defaults; sensitivity check in notebook 03 |
| Q2 | Admin-2 vs admin-1 for Afghanistan/Myanmar | Admin-2 where ≥ 5 events; else admin-1 |
| Q3 | Date range: 3 vs 5 years | 3 full years + current partial year |
| Q4 | Include `protest`/`riot` in "escalation" or restrict to armed violence? | Include all; label thresholds separate armed vs protest via `event_type` |
| Q5 | 7-day vs 30-day horizon mention in write-up | Report 14-day only; note others as future work |

---

## 17. Build Order (how to start — 15-minute first sprint)

1. **Register** at acleddata.com (do this first — it gates everything).
2. **Scaffold the repo** exactly as §10.2: `git init`, `venv`, `pip install` core deps, write `.gitignore` (ignore `data/raw/`, `models/`, `.env`), create `.env.example`.
3. **Pull data** via Data Export Tool for the 4 countries → save to `data/raw/`; log the exact filter parameters in a `data/README.md`.
4. **Write `src/config.py`** with all constants (countries, date range, windows, thresholds, split dates, seed).
5. **Implement `data_loader.py`** (schema validation, date parsing, dedupe, geo normalization) → verify on a 2-country subset.
6. **Prototype `feature_engineer.py`** in notebook 02 on one district, then productionize.
7. Follow M2 → M5 in §13, keeping the risk-ordered build (model before viz).

---

## 18. Appendix

### 18.1 Key references (cite in README/PPT)
- Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). *Introducing ACLED: An Armed Conflict Location and Event Dataset.* Journal of Peace Research, 47(5), 651–660.
- ACLED Conflict Alert System (CAST) — methodological reference for hierarchical LightGBM forecasting: [github.com/ACLED/cast-public](https://github.com/ACLED/cast-public)
- Explainable conflict fatality forecasting (UCDP-based, probabilistic framing): [github.com/fif911/probabilistic_conflict_modelling](https://github.com/fif911/probabilistic_conflict_modelling)
- ACLED R tooling: [github.com/chris-dworschak/acled.api](https://github.com/chris-dworschak/acled.api)

### 18.2 Attribution requirement
ACLED data is free but requires attribution: **"Data: ACLED, acleddata.com"** — include in README **and** on the PPT closing slide.

### 18.3 Glossary
| Term | Meaning |
|---|---|
| Admin-1 / Admin-2 | First/second-level administrative division (state/province vs district) |
| Rolling window | Features computed over the trailing N days as of each date |
| Escalation velocity | Change in event count between the current and previous window |
| Spillover | Effect of violence in geographically adjacent districts |
| Leakage | Using future information to predict the past; forbidden in time-series ML |
| AUC-PR | Area under the precision-recall curve; preferred over ROC for rare events |
| Brier score | Mean squared error of predicted probabilities vs outcomes |
