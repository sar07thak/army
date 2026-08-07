# Project Progress Summary

> Conflict Escalation Forecasting — 6 countries (South Asia + Sudan & South Sudan), 14-day horizon
> Generated: 2026-08-07 · Milestone 10 (SHAP Explainability) complete

---

## Overall Progress

- **Current milestone:** M10 — SHAP Explainability (complete, awaiting approval)
- **Overall completion:** **~77%** (10 of 13 milestones; M1–M10 fully built, gated, and verified)

| Milestone | Status |
|---|---|
| M1 Repository structure | ✅ Complete |
| M2 Configuration | ✅ Complete |
| M3 Data Loader | ✅ Complete |
| M4 Data Validation | ✅ Complete |
| M5 Feature Engineering | ✅ Complete |
| M6 Label Engineering | ✅ Complete |
| M7 Train/Val/Test Split | ✅ Complete |
| M8 LightGBM Training | ✅ Complete |
| M9 XGBoost + Comparison | ✅ Complete |
| M10 SHAP Explainability | ✅ Complete — this report |
| M11–M13 | ⏳ Not started |

---

## Repository Status

```
army/
├── config.py                  # Single source of truth (paths, scope, rules, thresholds, seeds)
├── run_pipeline.py            # CLI: --stage ingest | features | labels | split | train | compare | explain
├── requirements.txt           # Exact-pinned deps
├── pytest.ini                 # pythonpath + test config
├── .gitignore / .env.example  # Hygiene / secrets template
├── IMPLEMENTATION_PLAN.md     # Approved plan + adaptation notes (§3.4)
├── PROGRESS.md                # This file
├── data/
│   ├── README.md              # Provenance + attribution
│   ├── raw/                   # ACLED weekly aggregated CSV (127,353 rows)
│   └── processed/
│       ├── cleaned_events.{parquet,csv}     # 127,052 × 14
│       ├── district_master.{parquet,csv}    # 124 × 9
│       ├── features.{parquet,csv}           # 44,146 × 37
│       ├── labeled_features.{parquet,csv}   # 43,981 × 38
│       └── split_{train,val,test}.{parquet,csv}  # 30,790 / 6,522 / 6,669
├── models/
│   ├── escalation_lgbm.pkl     # Trained LightGBM (gitignored)
│   ├── escalation_xgb.pkl      # Trained XGBoost (M9)
│   ├── escalation_best.pkl     # Winner = XGBoost (M9)
│   ├── manifest.json           # LGBM params, features, cut dates, val metrics
│   └── model_comparison.json   # Full M9 comparison + threshold sweep + baselines
├── reports/
│   ├── feature_summary.md     # 37 features × dtype/missing/statistics
│   ├── label_summary.md       # Label distribution, by country/unit/month
│   ├── label_timeline.png     # Positive-rate timeline
│   ├── split_summary.md       # Cut dates, rows, class balance per split
│   ├── model_comparison.md    # Winner, metrics, threshold analysis, baselines
│   ├── shap_summary.md        # Top-20 features, interpretations, risk drivers, local explanations
│   └── shap/                  # 21 plots: summary, bar, 10 dependence, 9 waterfalls
├── notebooks/                 # (M13 — reserved)
├── src/
│   ├── __init__.py
│   ├── logging_config.py      # Rotating file + console logging → logs/project.log
│   ├── exceptions.py          # ConflictForecastError hierarchy
│   ├── data_loader.py         # Adaptive CSV discovery, merge, canonicalize, save
│   ├── data_validation.py     # All 9 validation rules + district master + orchestrator
│   ├── feature_engineer.py    # Leakage-safe rolling-window features (PRD §11.3)
│   ├── label_engineer.py      # Future-only escalation labels (PRD §11.2)
│   ├── split.py               # Strict chronological train/val/test (PRD §11.5)
│   ├── models.py              # LGBM + XGB train/save/load, imbalance, metrics, baselines, winner
│   ├── explainability.py      # SHAP: importance, summary/bar/waterfall/dependence plots, local explanations
│   └── pipeline.py            # Orchestration: train_stage + compare_stage + explain_stage
├── tests/
│   ├── conftest.py            # Deterministic fixtures (aggregated + event-level shapes)
│   ├── test_logging_config.py # 5
│   ├── test_config.py         # 15
│   ├── test_data_loader.py    # 12
│   ├── test_data_validation.py# 36
│   ├── test_feature_engineer.py # 24
│   ├── test_label_engineer.py # 19
│   ├── test_split.py          # 22
│   ├── test_models.py         # 66
│   └── test_explainability.py # 16
└── logs/project.log
```

**Files created:** 24 source/test/config files + 11 generated artifacts + 4 reports + 21 SHAP plots.
**Files modified:** `IMPLEMENTATION_PLAN.md` (twice: §3.4 adaptation + M6 note), `config.py`, `requirements.txt`, `src/exceptions.py`, `run_pipeline.py` (per milestone).

---

## Completed Milestones

### M1 — Repository structure
- **Objective:** Git-initialized, reproducible skeleton with venv.
- **Files created:** `src/__init__.py`, `src/logging_config.py`, `requirements.txt` (exact-pinned), `pytest.ini`, `.gitignore`, `.env.example`, `tests/test_logging_config.py`, `.gitkeep` placeholders.
- **Core functionality:** Central logging (rotating file → `logs/project.log`, console handler).
- **Validation:** All 6 gates (syntax, imports, deps, tests, file existence, git init). Python 3.14.3 — all wheels available (lightgbm 4.7.0, xgboost 3.4.0, shap 0.52.0, pandas 3.0.5).
- **Tests:** 5 · **Coverage:** 100% (logging_config).

### M2 — Configuration
- **Objective:** Everything configurable; nothing hardcoded.
- **Files created:** `config.py` (+ `validate_config()`).
- **Core functionality:** Paths, country scope, study window, validation rules, window sizes, label thresholds, split ratios, seeds, model hyperparameters, spillover K.
- **Validation:** Config load + sanity checks; used by every downstream module.
- **Tests:** 13 (incl. 11 parametrized rejection cases) · **Coverage:** 100%.

### M3 — Data Loader
- **Objective:** Ingest the provided ACLED data, adapt to its real shape.
- **Files created:** `src/data_loader.py`.
- **Core functionality:** CSV discovery in `data/raw/`, multi-file merge, schema-aware canonicalization (`week`→`event_date`, `centroid_*`→`lat/lon`, `events` default 1), parquet+CSV save. **No API/export code** (per instruction).
- **Validation:** 127,353 raw rows → loaded, canonicalized, saved.
- **Tests:** 12 (discovery, merge, both canonicalize paths, errors, saves) · **Coverage:** 95%.

### M4 — Data Validation
- **Objective:** Every PRD rule enforced with meaningful errors; nothing silent.
- **Files created:** `src/exceptions.py`, `src/data_validation.py`, `data/README.md`, `tests/conftest.py`, `tests/test_data_validation.py`, `tests/test_config.py`.
- **Core functionality:** Required columns; date parsing with format fallback; missing values incl. empty strings; duplicate events (composite `event_id` where `event_id_cnty` absent); country scope filter; coordinate bounds; count/type coercion; name normalization; `geo_unit` derivation; `build_district_master()`; `validate_dataset()` orchestrator.
- **Validation:** 301 out-of-scope rows (Indian Ocean region) filtered; 0 duplicates; 0 missing; `FATA` preserved (legitimate acronym).
- **Tests:** 36 · **Coverage:** 99%.

### M5 — Feature Engineering
- **Objective:** All PRD §11.3 feature groups, provably leakage-free.
- **Files created:** `src/feature_engineer.py`, `tests/test_feature_engineer.py`.
- **Core functionality:** Half-open windows `[as_of−W, as_of)` via prefix sums; event/fatality counts + log1p (7/14/30d); velocity (current − prior window); fatality mean/std (14/30d, population); persistence (active days in 7d); Shannon entropy (7/14/30d); `days_since_event` (sentinel 999); calendar (month, day_of_week); deterministic country/admin1/geo-unit codes; spillover (K=3 nearest same-country centroids, haversine).
- **Validation:** 44,146 rows × 37 cols, 0 NaN, 0 dupes; leakage **proven** by spike test + seeded randomized property test.
- **Tests:** 24 (every group hand-computed + 2 leakage proofs) · **Coverage:** 98%.

### M6 — Label Engineering (this milestone)
- **Objective:** Leakage-safe binary escalation labels, PRD §11.2.
- **Files created:** `src/label_engineer.py`, `tests/test_label_engineer.py`, `reports/label_summary.md`, `reports/label_timeline.png`.
- **Core functionality:** Labels use **only** rows in `(as_of, as_of+14d]`; trailing medians from `[as_of−30d, as_of)`; incomplete future windows dropped (134 rows); `validate_labels()`; summary report + matplotlib timeline.
- **Validation:** 43,981 × 38, labels ∈ {0,1}, 0 missing, all units chronologically ordered, 124/124 geo units consistent, last row = end−horizon (window-drop semantics confirmed).
- **Tests:** 19 (correctness, boundaries, empty window, incompleteness, chronology, geo isolation, leakage, threshold behavior, vanished units) · **Coverage:** 94%.

### M7 — Train/Validation/Test Split (this milestone)
- **Objective:** Strict chronological split; no shuffle, no random CV, no stratification across time (PRD §11.5).
- **Files created:** `src/split.py`, `tests/test_split.py`, `reports/split_summary.md`.
- **Core functionality:** Date-axis quantile cuts from `SPLIT_RATIOS` (not row counts, keeping calendars aligned); inclusive contiguous date ranges; `assert_no_leakage` (`max(train) < min(val) < min(test)`); `validate_splits` (row conservation, label values, no duplicates, NaT rejection); split summary report.
- **Validation:** 43,981 rows → train 30,790 (2016-12-31→2023-09-02) · val 6,522 (2023-09-09→2025-02-01) · test 6,669 (2025-02-08→2026-07-11); boundaries strict, no overlap, class balance 69.2% / 67.7% / 67.3%.
- **Tests:** 22 (chronology, no-shuffle, date-axis-vs-row-count, custom ratios, leakage, NaT, boundaries, determinism, conservation, report) · **Coverage:** 95%.

### M8 — LightGBM Training
- **Objective:** Trainable, savable, loadable deterministic LightGBM classifier with class-imbalance handling (PRD §11.4, FR-7).
- **Files created:** `src/models.py`, `src/pipeline.py`, `tests/test_models.py`; artifacts `models/escalation_lgbm.pkl`, `models/manifest.json`.
- **Core functionality:** Feature resolution (meta/label excluded); `prepare_xy` (numeric, NaN-free, both classes); imbalance via `scale_pos_weight` (LightGBM convention `n_neg/n_pos` = 0.445) or `class_weight` sample weights; fixed seed 42; joblib save/load; `binary_metrics` (precision/recall/F1/AUC-PR); `train_stage` end-to-end with JSON manifest (params, features, split cut dates, validation metrics).
- **Validation:** Trained on 30,790 rows × 33 features; **validation F1 0.793** (precision 0.843, recall 0.748, AUC-PR 0.900) at threshold 0.5; save/load round-trip produces identical predictions; retrain reproduces identical metrics (determinism proven).
- **Tests:** 30 (feature resolution, imbalance math, determinism, round-trip, hand-computed metrics, all error paths, end-to-end smoke) · **Coverage:** models 96%, pipeline 100%.

### M10 — SHAP Explainability (this milestone)
- **Objective:** Production-quality explainability for the winning model (PRD FR-9 / §12); never retrain.
- **Files created:** `src/explainability.py`, `tests/test_explainability.py`, `reports/shap_summary.md`, `reports/shap/` (21 plots); `src/exceptions.py` (+`ExplainabilityError`), `config.py` (SHAP constants + validation), `run_pipeline.py` (`--stage explain`), `tests/test_config.py` (+2).
- **Core functionality:** Loads `escalation_best.pkl` + held-out test window; deterministic even-spaced sampling cap (2,000 rows); TreeExplainer with positive-class extraction + base-value handling (list/ndarray); mean-|SHAP| importance ranking; summary (beeswarm) + bar plots; waterfall plots for representative correct-positive, correct-negative, and borderline predictions (confidence-ranked); dependence plots for top-10 features; pattern-based feature interpretation glossary; **data-driven** model-behaviour observations (family sums computed from the full ranking so totals are exact); local top-K driver explanations; operating threshold read from `model_comparison.json`; `reports/shap_summary.md` (top-20 + interpretations + drivers + local explanations).
- **Validation:** 21/21 PNGs verified (magic bytes + PIL verify); SHAP values shape (2000, 33) == prediction dims; feature names match training features; report observations derived from actual ranking (30-day emphasis confirmed: 30d sums 1.015 vs 14d 0.563 vs 7d 0.260); all functions ≤60 lines (PRD rule); no leakage — model never retrained, explanations on out-of-sample test window only.
- **Top-20 drivers:** `events_w30d` (0.478), `fatalities_w30d` (0.414), `velocity_events_w30d` (0.317), `events_w7d` (0.177), `spillover_w14d` (0.156), `admin1_code` (0.154), `velocity_fatalities_w30d` (0.148), `events_w14d` (0.148), `fatalities_w14d` (0.127), `month` (0.114) …
- **Tests:** 16 new (missing-artifact errors, SHAP shape/mismatch, importance ranking, representative categories incl. missing categories, interpretation patterns, PNG validity, end-to-end stage, threshold fallback) · **Coverage:** explainability 92%.

### M9 — XGBoost Training, Model Comparison & Threshold Optimization (completed M9)
- **Objective:** Fair head-to-head vs LightGBM on identical data; winner → `escalation_best.pkl`; optimized operating threshold (PRD §11.4, FR-14).
- **Files created (modified):** `src/models.py` (XGB family, `full_metrics`, `threshold_analysis`, 4 baselines, `select_winner`), `src/pipeline.py` (`compare_stage` + report writer), `run_pipeline.py` (`--stage compare`), `config.py` (threshold grid, comparison constants, baseline columns), `tests/test_models.py` (+36), `tests/test_config.py` (+5); artifacts `models/escalation_xgb.pkl`, `models/escalation_best.pkl`, `models/model_comparison.json`, `reports/model_comparison.md`.
- **Core functionality:** `train_model(family=...)` for LightGBM/XGBoost (identical seed 42, features, splits, `scale_pos_weight`); `full_metrics` (precision/recall/F1/PR-AUC/ROC-AUC/Brier/log-loss/confusion matrix); `threshold_analysis` sweep 0.10–0.90 (step 0.05) with best-F1/precision/recall points; baselines (majority, always-positive, persistence, event-count heuristic); `select_winner` per PRD priority (F1 → PR-AUC → Brier → simplicity order); `_load_lgbm_verified` proves the M8 metrics are unchanged; comparison JSON + markdown report with threshold rationale.
- **Validation:** **XGBoost wins** — validation F1 **0.8423** (best-F1 threshold 0.25) vs LightGBM 0.8400 (threshold 0.20); PR-AUC 0.9031 vs 0.9004; operating threshold **0.25** (`max_f1`); all four baselines beaten (persistence 0.8261, majority/always-positive 0.8076, event-count 0.8044); test split never touched by selection; LightGBM metrics verified unchanged vs M8 manifest.
- **Tests:** 36 new (XGB determinism/round-trip, hand-computed full metrics, grid bounds, best-point selection, baselines, winner priority incl. tie-breaks, drift detection, test-split isolation, report rationale) · **Coverage:** models 95%, pipeline 99%.

---

## Current Pipeline

```
data/raw/*.csv (ACLED weekly aggregated, 127,353 rows)
   │  data_loader.load_csv_files()      # discovery + merge + canonicalize
   ▼
canonical events frame
   │  data_validation.validate_dataset()  # 9 rules + geo_unit + event_id
   ▼
cleaned_events (127,052 × 14) ──► district_master (124 × 9)
   │  feature_engineer.build_features()   # half-open [as_of−W, as_of) windows
   ▼
features (44,146 × 37) ──► feature_summary.md
   │  label_engineer.build_labeled_dataset()  # future-only (as_of, as_of+14d]
   ▼
labeled_features (43,981 × 38) ──► label_summary.md + label_timeline.png
   │  split.chronological_split()  # date-axis quantiles, no shuffle
   ▼
split_train (30,790) · split_val (6,522) · split_test (6,669) ──► split_summary.md
   │  models.train_model(family="lightgbm")  # scale_pos_weight=0.445, seed 42
   ▼
escalation_lgbm.pkl + manifest.json ──► validation F1 0.793 @ 0.5
   │  pipeline.compare_stage()  # reloads LGBM (unchanged ✓), trains XGBoost,
   │                            # full metrics + threshold sweep + 4 baselines
   ▼
escalation_xgb.pkl · escalation_best.pkl (= XGBoost) ──► model_comparison.json + .md
   │  winner: XGBoost, val F1 0.8423 @ operating threshold 0.25
   │  pipeline.explain_stage()  # TreeExplainer on held-out test window (2,000 sampled rows)
   ▼
reports/shap_summary.md + reports/shap/ (21 plots: summary · bar · 10 dependence · 9 waterfalls)
   │  top drivers: events_w30d · fatalities_w30d · velocity_events_w30d · spillover_w14d
   │  (M11: risk map → M12: docs → M13: final audit)
   ▼
ready for risk visualization
```

---

## Dataset Status

| Dataset | Rows | Cols | Notes |
|---|---|---|---|
| raw CSV | 127,353 | 12 | 2017-01-01 → 2026-08-07 (weekly buckets 2016-12-31 → 2026-07-25) |
| cleaned_events | 127,052 | 14 | 6 countries (India 40,404 · Myanmar 28,387 · Afghanistan 26,300 · Sudan 13,211 · Pakistan 11,346 · South Sudan 7,404 raw rows); 124 geo units; 0 missing; 0 dupes |
| district_master | 124 | 9 | top hotspots incl. Sindh, Punjab, Khyber Pakhtunkhwa, Balochistan, Kabul |
| features | 44,146 | 37 | 0 NaN, 0 duplicate rows |
| labeled_features | 43,981 | 38 | 30,217 positive (68.7%) / 13,764 negative; 165 incomplete-tail rows dropped |
| split_train / val / test | 30,790 / 6,522 / 6,669 | 38 | chronological, no overlap; test = newest 15% (2025-02-08 → 2026-07-11); pos rates 69.2% / 67.7% / 67.3% |

## Model Statistics (M9 — comparison outcome)

| Item | LightGBM | XGBoost (winner) |
|---|---|---|
| Family / seed | LGBMClassifier · seed 42 | XGBClassifier · seed 42 |
| Training rows / features | 30,790 · 33 | 30,790 · 33 (identical) |
| Validation rows | 6,522 | 6,522 (identical) |
| Imbalance handling | `scale_pos_weight` = 0.4447 | `scale_pos_weight` = 0.4447 |
| **Best-F1 threshold** | **0.20** | **0.25 (operating)** |
| **Validation F1 @ best** | **0.8400** | **0.8423** |
| PR-AUC | 0.9004 | 0.9031 |
| ROC-AUC | 0.8112 | 0.8175 |
| Brier | 0.1743 | 0.1755 |
| Log loss | 0.5145 | 0.5171 |
| Metrics @ 0.5 (F1 / prec / rec) | 0.7928 / 0.8433 / 0.7480 | 0.7803 / 0.8555 / 0.7172 |
| Artifacts | `escalation_lgbm.pkl` + `manifest.json` | `escalation_xgb.pkl` + `escalation_best.pkl` |

**Baselines (validation, @0.5):** persistence F1 0.8261 · majority 0.8076 · always-positive 0.8076 · event-count heuristic 0.8044. Winner **beats all four**; the operating threshold (0.25) is the argmax-F1 point, well below 0.5 because the majority-positive label makes 0.5 suboptimal (see Risks).

---

## Feature Engineering Status (37 features)

- **Counts + log1p:** `events_{7,14,30}d`, `fatalities_{7,14,30}d`, `log_events_{7,14,30}d`, `log_fatalities_{7,14,30}d`
- **Velocity:** `velocity_events_{7,14,30}d`, `velocity_fatalities_{7,14,30}d` (= current window − prior window)
- **Fatality stats:** `fat_mean_{14,30}d`, `fat_std_{14,30}d` (population std, ddof=0)
- **Persistence:** `persistence_w7d` (active days in last 7)
- **Diversity:** `entropy_{7,14,30}d` (Shannon, natural log)
- **Recency:** `days_since_event` (sentinel 999 when no history)
- **Calendar:** `month`, `day_of_week`
- **Identity:** `country_code`, `admin1_code`, `geo_unit_code` (deterministic)
- **Spillover:** `spillover_w14d` (K=3 nearest same-country centroids, haversine)
- *Actor features:* skipped for this dataset (no actor columns in the aggregated export) — auto-enable with an event-level export.

---

## Label Engineering Status

- **Definition (PRD §11.2):** `escalation = 1` iff
  `(future_events ≥ 3 AND future_events ≥ 1.5 × trailing-30d median) OR future_fatalities ≥ 5`;
  absolute fallback `future_events ≥ 5` when the unit has no trailing history.
- **Horizon:** next 14 days; window is **exclusive** of `as_of` (`as_of < date ≤ as_of+14`).
- **Thresholds:** all from `config.py` (`LABEL_MIN_EVENTS`, `LABEL_MEDIAN_MULTIPLIER`, `LABEL_FATALITIES`, `LABEL_ABSOLUTE_EVENTS`, `PREDICTION_HORIZON_DAYS`, `TRAILING_MEDIAN_WINDOW_DAYS`).
- **Positive ratio:** 68.7% (30,217 / 43,981) — see Risks.
- **Leakage prevention:** labels computed strictly from future rows; features are never touched (strictly past by construction); incomplete future windows dropped, never partially labeled; per-unit isolation.

---

## Tests

- **Total:** 215 · **Passing:** 215 · **Coverage:** 95.98% (gate ≥80%)
  - logging_config 100% · exceptions 100% · config 100% · pipeline 99% · data_validation 99% · feature_engineer 98% · models 95% · data_loader 95% · split 95% · label_engineer 94% · explainability 92%
- **Key edge cases covered:** format-fallback date parsing, empty-string-as-missing, out-of-bounds coordinates, composite-key duplicates, mixed-format dates, min-events filter, single-row units, empty windows, sentinel recency, spillover neighbors, half-open-window boundaries, spike-injection leakage proof, seeded randomized leakage property test, empty future window, incomplete-window drop, vanished units, geo-unit isolation, threshold boundaries, XGB determinism/round-trip, full-metrics hand-computation, threshold grid bounds, best-point selection, all four baselines, winner tie-breaks (F1→PR-AUC→Brier→simplicity), LGBM drift detection, test-split isolation, SHAP shape/mismatch errors, representative-category selection (incl. missing categories), PNG validity, explain-stage end-to-end, operating-threshold fallback.

---

## Validation

| Check | Result |
|---|---|
| Imports (all modules) | ✅ |
| Config `validate_config()` | ✅ |
| Dependencies (`pip check`) | ✅ "No broken requirements found" |
| Syntax (`py_compile`) | ✅ |
| Pipeline execution (ingest → features → labels) | ✅ end-to-end on real data |
| Dataset validation (cleaned) | ✅ 127,052 × 14, unique IDs |
| Feature validation | ✅ 0 NaN, 0 dupes, 37 cols |
| Label validation | ✅ {0,1} only, 0 missing, 43,981 rows |
| Split validation | ✅ 43,981 → 30,790/6,522/6,669; max(train)<min(val)<min(test); no overlap |
| Model validation | ✅ deterministic retrain = identical metrics; save/load round-trip identical predictions |
| XGBoost training | ✅ identical features/splits/seed; `escalation_xgb.pkl` saved |
| LightGBM unchanged | ✅ `_load_lgbm_verified`: val metrics equal M8 manifest (drift → ModelError) |
| Threshold optimization | ✅ sweep 0.10–0.90 (17 points); best-F1/precision/recall per model; operating = 0.25 |
| Baselines | ✅ majority · always-positive · persistence · event-count — all scored, all beaten |
| Winner selection | ✅ XGBoost by PRD priority (F1 0.8423 > 0.8400); reason documented |
| Test-split isolation | ✅ selection never reads `split_test` (poisoned-split test proves it) |
| Leakage validation | ✅ proven (tests + window semantics + tail-drop + split boundaries) |
| SHAP computation | ✅ TreeExplainer on winner; values (2000×33) match prediction dims; base value extracted |
| SHAP visualizations | ✅ 21 PNGs (summary, bar, 10 dependence, 9 waterfalls) — all verified non-corrupt |
| Feature-name consistency | ✅ SHAP feature names == training features (mismatch raises `ExplainabilityError`) |
| Report generation | ✅ `reports/shap_summary.md`: top-20 + interpretations + risk drivers + local explanations |
| Data-driven observations | ✅ family sums computed from full ranking (identity 0.228 vs calendar 0.114; 30d > 14d > 7d) |
| Regressions M1–M9 | ✅ all prior suites still pass (215/215) |

---

## PRD Compliance

**Implemented (fully):** repo structure · config-driven everything · logging · exception hierarchy · adaptive data loader (no API per instruction) · 9 validation rules · district master · rolling windows (7/14/30d) · velocity · fatality stats · persistence · Shannon entropy · recency · calendar · spillover (FR-13) · leakage-safe labels (FR-8) · per-unit chronological processing · incomplete-window policy · label summary + timeline · chronological split (FR-6) · LightGBM training with imbalance handling + save/load + determinism (FR-7, NFR determinism) · XGBoost on identical data · full metric suite incl. ROC-AUC/Brier/log-loss/confusion (FR-15) · threshold sweep + operating-point selection (FR-14) · four baselines with comparison (FR-8) · PRD-priority winner selection · `escalation_best.pkl` + comparison report/JSON · **SHAP explainability (FR-9, PRD §12):** global importance, summary + bar plots, waterfall (pos/neg/borderline), dependence plots (top-10), local explanations, top-20 drivers, `reports/shap_summary.md`.

**Deviations (all user-approved, documented in IMPLEMENTATION_PLAN.md §3.4):**
1. **Granularity:** source is ACLED **weekly admin-1 aggregated counts** (not event-level district rows) → forecasts are **province-level, weekly**; `event_id` is a composite key (no `event_id_cnty`); actor-diversity features skipped ("if available").
2. **No API / data-export code** (explicit instruction; manual CSV only).
3. **Country scope:** six countries per the user's Data Export Tool selection — India, Pakistan, Afghanistan, Myanmar, **Sudan, South Sudan** (only the `Indian Ocean` region rows are filtered).

---

## Known Limitations

- **High positive-class rate (68.7%)** at this granularity with these thresholds — handled in M9 via threshold tuning (operating point 0.25) and PR-AUC as headline; thresholds remain config-adjustable.
- **Modest lift vs majority (≈1.04× on F1 @ best threshold):** the winner's edge over the trivial baseline is small at this granularity; PR-AUC (0.903) is the stronger claim. A re-tuned label definition or event-level data may sharpen separation (M13 revisit).
- 7-day window at weekly granularity = previous week's bucket (documented adaptation).
- No actor/event-type columns in source → actor features and event-type entropy use `event_type` diversity only where present; entropy is computed on event-type counts within windows where the column exists (aggregated file provides `event_type`).
- `days_since_event` saturates at sentinel 999.
- Weekly data ⇒ finer daily dynamics are not representable.

---

## Remaining Milestones

- **M11** — Risk map & visualization (interactive map, country trends, hotspot analysis, temporal charts)
- **M12** — Documentation & README
- **M13** — Final audit & hackathon-submission readiness (full checklist)

---

## Risks

1. **Class imbalance (68.7% positive) — resolved for operating point:** threshold tuning moved the operating threshold from 0.5 to **0.25** (max-F1), lifting validation F1 from 0.793 → **0.842** (winner) and beating all baselines. Remaining caveat: F1 lift vs majority is ≈1.04×; PR-AUC (0.903) is the strongest headline metric. Optional M13 experiment: `scale_pos_weight = n_pos/n_neg` or 1.0.
2. **XGBoost wins narrowly (F1 0.8423 vs 0.8400) and has slightly worse Brier (0.1755 vs 0.1743):** the F1→PR-AUC→Brier priority picks XGBoost, but the margin is thin; both models remain saved so downstream milestones can switch the winner with a config-only change if test-window evidence favors LightGBM.
3. **SHAP on a 2,000-row even-spaced sample** of the test window (memory cap, configurable via `SHAP_SAMPLE_CAP`); importance ranking is stable across the window, but per-row waterfalls describe sampled representative rows only.
4. **Province-level scope:** the product is "early warning" at admin-1 granularity; an event-level ACLED export would drop-in and restore district-level + actor features without pipeline changes.
5. **Python 3.14 / pandas 3.0.5 (PyArrow str dtype):** handled throughout; wheel availability verified for all pinned deps.
6. **Collinearity:** `geo_unit_code` and `admin1_code` are identical on this dataset (geo_unit == admin1); harmless for tree models, kept for generality.

---

## Final Readiness

**Ready for Milestone 11 (Risk Map & Visualization): Yes.**
M1–M10 are complete, gated, reviewed, and regression-free (215/215 tests, 95.98% coverage). The winner (XGBoost) is explained end-to-end with SHAP: global importance, summary/bar/waterfall/dependence plots (all verified non-corrupt), local explanations for positive/negative/borderline predictions, and a data-driven `reports/shap_summary.md` (top-20 drivers led by `events_w30d`/`fatalities_w30d`, 30-day windows dominate, spillover ranks #5). The model was never retrained; all explanations are on the held-out test window. M11 will turn the winner's risk scores into the interactive risk map, country trends, hotspot analysis, and temporal charts per PRD §13.
