# Implementation Plan — Conflict Escalation Forecasting

**Status: PENDING APPROVAL** — per build rules, no code is written until this plan is approved.

| Field | Value |
|---|---|
| **Source of truth** | `PRD.md` v1.0 (approved) |
| **Build rules** | The 13-milestone, validation-gated process defined by the team lead (this document operationalizes them) |
| **Environment** | Windows (win32), Python 3.11+ recommended (3.11 or 3.12; 3.13 has wheel gaps for some ML deps on Windows) |
| **Deliverable** | Complete, reproducible ML forecasting package — repo, pipeline, notebooks, trained model, evaluation, SHAP, risk map, documentation |

---

## 0. How to read this document

- **Section 1** proves no requirement is ignored: every PRD requirement and every build rule is mapped to a milestone and file.
- **Section 2** is the frozen target architecture (file inventory with responsibilities). Anything not in this tree is out of scope.
- **Sections 3–4** are the data contract and the 13 milestones, built strictly in the mandated order.
- **Section 5** is the cross-cutting engineering standard (logging, errors, config, style) applied to every file.
- **Section 6** is the testing strategy and **Section 7** the post-milestone validation protocol.
- **Section 8** lists the external prerequisites you must satisfy (mainly: ACLED data).
- **Section 9** is the requirements traceability matrix. **Section 10** risks, **Section 11** decisions to confirm, **Section 12** final acceptance checklist, **Section 13** the approval gate.

---

## 1. Build-rules compliance map

| Build rule (from your instructions) | Where it is satisfied |
|---|---|
| Never ignore a requirement | §9 traceability matrix — every FR-1…FR-15, every §11.3 feature, every NFR is assigned to a milestone/file |
| Never simplify, never invent features | Scope frozen in §2; anything not in the file tree is explicitly out of scope (§2.1) |
| Stop and ask if a requirement is unclear | §11 lists every decision point with its PRD-default; you approve them in §13 |
| Every module compiles / notebook runs / script executable / import exists | §7 validation protocol (py_compile, import checks, `nbconvert --execute`), run after **every** milestone |
| Every dependency listed | `requirements.txt` authored in M1, `pip check` + freeze-vs-file gate every milestone |
| No placeholders / TODOs / pseudo-code / fake implementations | M1–M13 done-conditions; final validation grep for `TODO/FIXME/pass # stub` (§12) |
| Python 3.11+, PEP8, type hints, docstrings | §5.4 coding standard; enforced by review gate in every milestone |
| No function > 60 lines, no class > 300 lines | §5.4; lint gate (`ruff` optional) |
| Exactly the mandated structure | §2 file tree (with 4 justified additions, each explained) |
| Milestone-by-milestone, wait for approval | §3 order + §13 gate; nothing built ahead of schedule |
| Post-milestone validation before continuing | §7 protocol executed after every milestone |
| Data validation with meaningful exceptions | M4 (`src/data_validation.py` + `src/exceptions.py`) |
| Every feature in PRD §11.3 | M5 feature table (all 9 groups, spillover config-gated) |
| Labels use ONLY future data | M6 label engine + leakage unit tests |
| LightGBM + XGBoost, imbalance, seeds, save/load | M8–M9 (`src/models.py`) |
| Only chronological splits | M7 (`src/split.py`) + tests that forbid leakage |
| Full evaluation suite + 3 baselines | M10 (`src/evaluate.py`) |
| SHAP: summary, importance, waterfall, dependence, top drivers | M11 (`src/explain.py`) |
| Risk map + country trends + hotspot + temporal charts | M12 (`src/visualize.py`) |
| pytest ≥ 80% coverage | §6 matrix; coverage gate in §7 |
| Descriptive exceptions, logging to `logs/` | §5.1–5.2; `src/logging_config.py`, `src/exceptions.py` |
| Everything configurable, no hardcoded values | M2 `config.py` (root, per your structure) |
| Professional README | M13 (10 mandated sections) |
| Final validation before declaring complete | §12 checklist run at M13 |

---

## 2. Target architecture & file inventory (frozen)

### 2.1 File tree

```
army/                                  # repo root = this folder (PRD.md + guide stay at top level)
├── config.py                          # ALL constants: paths, countries, dates, windows, thresholds,
│                                      #   split ratios, seed, LGBM/XGB params, eval settings  [M2]
├── run_pipeline.py                    # executable end-to-end orchestrator (CLI)  [M8+, final in M13]
├── requirements.txt                   # pinned runtime + dev deps  [M1]
├── pytest.ini                         # pytest config: pythonpath=., testpaths, coverage opts  [M1]
├── .gitignore                         # data/, models/, logs/, .env, caches, venv  [M1]
├── .env.example                       # ACLED_API_KEY=  [M1]
├── README.md                          # 10 mandated sections  [M13]
├── src/
│   ├── __init__.py                    # package marker  [M1]
│   ├── exceptions.py                  # ConflictForecastError hierarchy  [M3]
│   ├── logging_config.py              # rotating file logger → logs/ + console, INFO/WARNING/ERROR  [M1]
│   ├── data_loader.py                 # ACLED CSV ingestion + optional paged API fetch  [M3]
│   ├── data_validation.py             # schema, missing, dupes, dates, coords, names, types  [M4]
│   ├── feature_engineer.py            # all rolling-window features, leakage-safe  [M5]
│   ├── label_engine.py                # 14-day escalation labels, future-only  [M6]
│   ├── split.py                       # strict chronological train/val/test  [M7]
│   ├── models.py                      # LGBM + XGB wrappers, imbalance, save/load  [M8–M9]
│   ├── evaluate.py                    # full metric suite + 3 baselines + threshold analysis  [M10]
│   ├── explain.py                     # SHAP summary/waterfall/dependence/top-drivers  [M11]
│   ├── visualize.py                   # risk map, country trends, hotspot, temporal charts  [M12]
│   └── pipeline.py                    # run_pipeline() used by run_pipeline.py + notebooks  [M8+]
├── notebooks/
│   ├── 01_eda.ipynb                   # coverage, hotspots, event mix, trends  [M13]
│   ├── 02_feature_engineering.ipynb   # features + labels demo, leakage sanity  [M13]
│   └── 03_modeling.ipynb              # split→train→eval→SHAP→map, results + narrative  [M13]
├── scripts/
│   └── validate_project.py            # automated final-validation report (written complete in M13)  [M13]
├── tests/
│   ├── conftest.py                    # deterministic synthetic ACLED-schema fixtures  [M3]
│   ├── test_logging_config.py         # [M1]
│   ├── test_config.py                 # [M2]
│   ├── test_data_loader.py            # [M3]
│   ├── test_data_validation.py        # [M4]
│   ├── test_feature_engineer.py       # incl. leakage tests  [M5]
│   ├── test_label_engine.py           # incl. leakage tests  [M6]
│   ├── test_split.py                  # [M7]
│   ├── test_models.py                 # [M8–M9]
│   ├── test_evaluate.py               # [M10]
│   ├── test_explain.py                # [M11]
│   └── test_visualize.py              # [M12]
├── data/
│   ├── raw/                           # ACLED CSVs — GITIGNORED  [M1]
│   ├── processed/                     # features.parquet, splits — GITIGNORED  [M5+]
│   └── README.md                      # data provenance: filters, dates, pull date, source  [M3]
├── models/                            # .pkl artifacts — GITIGNORED (keep .gitkeep)  [M8]
├── reports/                           # model_metrics.md, *.png, risk_map.html  [M10+]
└── logs/                              # project.log (rotating) — GITIGNORED  [M1]
```

### 2.2 Justified additions to your mandated structure (each is architecture, not scope creep)

| Addition | Why |
|---|---|
| `src/exceptions.py` | "Throw meaningful exceptions / never crash without explanation" needs one exception hierarchy |
| `src/logging_config.py` | Logging rule ("store logs in `logs/`") needs a single setup point — one place, no duplication |
| `run_pipeline.py` + `src/pipeline.py` | PRD's own pipeline diagram (§10.1) + "Pipeline runs end-to-end" requirement demand an executable orchestrator; also lets notebooks call `src/` instead of duplicating logic |
| `pytest.ini` | Makes `import src.*` work from the root and pins the 80% coverage gate |
| `data/README.md` | PRD NFR: data provenance & reproducible pulls ("versioned data snapshot noted in README") |
| `scripts/validate_project.py` | The validation protocol (§7) is only trustworthy if automated; one script runs every §12 check. Home scaffolded in M1; written complete in M13 |

### 2.3 Architectural decisions (fixed, not re-openable during build)

1. **`config.py` lives at the repo root** (your mandated structure). `src/` imports it; notebooks do too. The PRD's `src/config.py` is superseded by your structure.
2. **Notebooks are thin**: they call `src/` functions and render results/plots. Zero business logic inside notebooks → no duplicated code, notebooks always in sync with source.
3. **Risk map = Plotly `scatter_geo`** (no Mapbox token required) plotting district centroids from the data, colored by risk, hover = district + probability + top-3 SHAP drivers. A Folium choropleth is only added in M12 if a district GeoJSON boundary file is available (PRD allows "choropleth **or scatter** map").
4. **Processed data format**: `features.parquet` (PRD FR-4) with `pyarrow`.
5. **Model selection rule** (PRD §11.4): best of LGBM/XGB by **validation F1**; winner is saved as `models/escalation_best.pkl` with a manifest noting which model family it is. Both artifacts (`escalation_lgbm.pkl`, `escalation_xgb.pkl`) are retained per PRD §10.1.

---

## 3. Data contract (from PRD §9)

### 3.1 Canonical schema (post-load, pre-validation)

| Column | dtype | Source field | Rule |
|---|---|---|---|
| `event_id` | str | `event_id_cnty` | unique key for dedupe |
| `event_date` | datetime64 | `event_date` | parse `%d %b %Y` / ISO; invalid → exception with row context |
| `event_type` | category | `event_type` | one of 6 ACLED types; not pre-filtered (PRD §9.2) |
| `country` | str | `country` | must be in `COUNTRIES` whitelist (IN/PK/AF/MM) |
| `admin1` | str | `admin1` | non-empty (drop-rule per FR-2) |
| `admin2` | str | `admin2` | used as geo unit; empty → falls back to `admin1` (M4 normalization) |
| `latitude` / `longitude` | float64 | `latitude` / `longitude` | lat ∈ [-90, 90], lon ∈ [-180, 180] |
| `fatalities` | int | `fatalities` | ≥ 0, integer-coerced with explicit error on non-numeric |
| `actor1` | str | `actor1` | non-null string (nulls → "Unknown") |
| `geo_unit` | str | derived | canonical district key = `admin2|admin1|country` (M4) |
| `event_day` | int | derived | day-of-month for calendar features (M5) |

### 3.2 Scope parameters (config.py, from PRD §9.2)

`COUNTRIES = [India, Pakistan, Afghanistan, Myanmar]` · date range = last 3 full years + current partial year (default; 5 years is a stretch) · all event types · admin-2 default with **min-events filter ≥ 5 events/unit** else admin-1 fallback.

### 3.3 Data acquisition (user action — see §8)

Primary: your ACLED **Data Export Tool CSV** placed in `data/raw/`. Secondary (scripted): ACLED **API** with paged pulls (5,000 rows/call), key from `.env` (`ACLED_API_KEY`). `data_loader.py` supports both; the CSV path is the primary tested path.

### 3.4 Data adaptation decision (2026-08-07)

User decision: **"do according to whatever data is provided."** `data/raw/` currently holds the ACLED **weekly aggregated admin-1 count file** (week × country × admin1 × event_type, with `events`, `fatalities`, centroids) — not the event-level export. The loader is therefore **adaptive**: it canonicalizes whichever source columns exist onto one internal schema — `event_date` (from `event_date` or `week`), `events` (from `events` or `1` per row), coordinates (`latitude/longitude` or centroids), `geo_unit` (finest of `admin2`/`admin1`), `event_id` (`event_id_cnty` if present, else a composite key). Consequences for this run:
- **Geo unit = admin-1 (province), weekly granularity** — the "district-level" pitch becomes province-level for this dataset.
- PRD items needing event-level fields (dedupe via `event_id_cnty`, actor diversity, per-event coordinates) are **adapted** (composite-key dedupe) or deferred to feature engineering.
- Countries are filtered to the configured scope with dropped counts logged — currently **six countries** (India, Pakistan, Afghanistan, Myanmar, Sudan, South Sudan) per the user's Data Export Tool selection (2026-08-07); only the `Indian Ocean` region rows (301) are filtered. `DUPLICATES_MODE` defaults to `"drop"` (user requirement: remove duplicates).
- **No API / data-export functionality is implemented** (explicit user instruction).
- Environment note: pandas 3.0.5 uses the PyArrow-backed `str` dtype — all string handling is written against it.
- **M5 note (feature engineering):** features are computed on `(geo_unit, event_date)` rows with **half-open windows `[as_of − W, as_of)`** — strictly historical, so for weekly data a 7-day window equals the previous week's bucket. Missing history defaults to 0 for window features and the `RECENCY_SENTINEL` for days-since-event (no NaNs in the output). Actor-diversity features are computed only when the export has actor columns (the aggregated file does not). Spillover uses the K nearest same-country centroids (haversine).
- **M7 note (split):** `src/split.py` cuts over the **date axis** at the `SPLIT_RATIOS` quantiles (not row counts, keeping calendars aligned across units). On the 43,981 labeled rows: train 30,790 (2016-12-31→2023-09-02), val 6,522 (2023-09-09→2025-02-01), test 6,669 (2025-02-08→2026-07-11); boundaries re-verified by `assert_no_leakage` at runtime.
- **M9 note (comparison):** the user extended M9 beyond the plan's minimal "XGBoost baseline" — it now also performs the **threshold optimization** and the **four-baseline comparison** (which the plan had assigned to M10). `src/models.py` gained `full_metrics` (incl. ROC-AUC/Brier/log-loss/confusion), `threshold_analysis` (sweep 0.10–0.90, step 0.05, config-gated), the four PRD baselines, and `select_winner` (PRD priority F1 → PR-AUC → Brier → `MODEL_SIMPLICITY_ORDER`). `src/pipeline.py` gained `compare_stage`, which **reloads the M8 LightGBM artifact and proves its validation metrics are unchanged** (drift → `ModelError`), trains XGBoost on identical data, and writes `escalation_xgb.pkl`, `escalation_best.pkl`, `model_comparison.json`, `reports/model_comparison.md`. Result: **XGBoost wins** (val F1 0.8423 @ threshold 0.25 vs LGBM 0.8400 @ 0.20); operating threshold = **0.25** (max-F1, documented rationale in the report). The test split is never read during selection (isolation test proves it).
- **M8 note (training):** `src/models.py` + `src/pipeline.py` train a deterministic LGBMClassifier (seed 42) on the chronological splits with `IMBALANCE_METHOD='scale_pos_weight'` (LightGBM convention `n_neg/n_pos` = 0.4447 — the label is majority-positive so the factor is < 1, documented as a risk). Validation F1 0.793 (precision 0.843, recall 0.748, AUC-PR 0.900) at threshold 0.5; artifacts `models/escalation_lgbm.pkl` + `models/manifest.json`; retrain reproduces identical metrics.

---

## 4. Milestones (exact mandated order — one at a time, gated, awaiting approval after each)

**Legend for every milestone:** *Goal · Files · Implementation detail · Depends on · Validation (§7) · Done-when.*

---

### M1 — Repository structure
- **Goal:** empty-but-correct skeleton; nothing to import yet except stdlib.
- **Files:** all directories (`data/raw`, `data/processed`, `models`, `reports`, `logs`, `notebooks`, `src`, `tests`), `src/__init__.py`, `requirements.txt`, `pytest.ini`, `.gitignore`, `.env.example`, `src/logging_config.py` (first real module: stdlib-only `logging` + `RotatingFileHandler`).
- **Detail:** `.gitignore` covers `data/raw/`, `data/processed/`, `models/*.pkl`, `logs/`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`, `.pytest_cache/`, `.coverage`, `venv/`. `requirements.txt` (major-version pins): `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `shap`, `plotly`, `folium`, `requests`, `python-dotenv`, `pyarrow`, `jupyter`, `pytest`, `pytest-cov`. Optionally `git init` (recommended for submission).
- **Depends on:** nothing.
- **Done-when:** tree exists; `python -m py_compile src/logging_config.py` passes; `logs/` created on first log call; tree matches §2.1.

---

### M2 — Configuration
- **Goal:** single source of truth; **zero hardcoded values anywhere else in the repo** (build rule).
- **Files:** `config.py`, `tests/test_config.py`.
- **Detail:** grouped module-level constants + a frozen `load_validation()`-style sanity check:
  - *Paths*: `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `LOGS_DIR`.
  - *Data*: `COUNTRIES`, `DATE_START`, `DATE_END`, `MIN_EVENTS_PER_UNIT = 5`, `ADMIN_FALLBACK` flag.
  - *Features*: `ROLLING_WINDOWS = [7, 14, 30]`, velocity pairs, `VOLATILITY_WINDOWS = [14, 30]`, `SPILLOVER_ENABLED`, spillover window `14`.
  - *Labels*: `LABEL_HORIZON_DAYS = 14`, `ESCALATION_MIN_EVENTS = 3`, `ESCALATION_MULTIPLIER = 1.5`, `ESCALATION_MIN_FATALITIES = 5`, `ABSOLUTE_MIN_EVENTS = 5`, `INCOMPLETE_WINDOW = "drop"`.
  - *Split*: `SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}` (PRD §11.5), `RANDOM_SEED = 42`.
  - *Model*: `LGBM_PARAMS` (incl. `n_estimators`, `num_leaves`, `learning_rate`, `scale_pos_weight` = "auto" → computed from train class ratio, `random_state=seed`, `verbosity=-1`), `XGB_PARAMS` (analogous `scale_pos_weight`, `random_state`), `MODEL_DIR`/filenames.
  - *Eval*: `OPERATING_THRESHOLD_MODE = "max_f1"`, baseline flags, `METRICS` list.
  - `validate_config()` raises `ConfigurationError` on invalid combinations (e.g., windows not sorted, split ratios ≠ 1.0).
- **Depends on:** M1.
- **Done-when:** `python -c "import config; config.validate_config()"` passes; `test_config.py` green (all constants present, typed correctly, ranges valid).

---

### M3 — Data Loader
- **Goal:** raw ACLED data in → canonical-schema DataFrame out; cache to `data/raw/`.
- **Files:** `src/exceptions.py`, `src/data_loader.py`, `tests/conftest.py`, `tests/test_data_loader.py`, `data/README.md`.
- **Detail:**
  - `exceptions.py`: `ConflictForecastError` base → `DataLoadError`, `DataValidationError`, `ConfigurationError`, `FeatureEngineeringError`, `ModelError`, `EvaluationError`, `VisualizationError`. Every module raises these; nothing ever raises a bare `Exception`.
  - `load_from_csv(path)` — reads any CSVs in `data/raw/`, applies the canonical schema (§3.1) column rename map (ACLED export column names can differ per export; map is centralized here), parses dates, returns DataFrame.
  - `fetch_from_api(api_key, countries, date_range)` — paged GET (5,000/call), exponential backoff, progress logging; never writes the key to disk; stores cache CSV.
  - `load_raw_data()` — dispatch: CSV if present, else API if key present, else raise `DataLoadError` telling the user exactly what to do (drop the CSV or set `ACLED_API_KEY`).
- **Depends on:** M2.
- **Done-when:** loader runs on the synthetic fixture (tests) and on a real 2-country subset CSV if you have provided one (§8); `test_data_loader.py` green.

---

### M4 — Data Validation
- **Goal:** every PRD §9.4 / build-rule validation rule implemented; **nothing silently dropped**; meaningful exceptions with row context.
- **Files:** `src/data_validation.py`, `tests/test_data_validation.py`.
- **Detail:** one function per rule (each < 60 lines, pure, testable):
  - `validate_required_columns(df)` → raise with list of missing columns.
  - `validate_missing(df)` → report counts per column; raise for critical columns, warn (log WARNING) for `actor1`.
  - `validate_duplicates(df)` → dupes by `event_id` flagged; config choice: `"drop"` or `"raise"` (default `"raise"` for data-integrity honesty — duplicates indicate a broken pull).
  - `validate_dates(df)` → parse failures raise `DataValidationError` naming offending rows; out-of-range dates vs `DATE_START/END` warn.
  - `validate_coordinates(df)` → bounds check; invalid rows raise with row indices and values.
  - `validate_countries(df)` → whitelist; unknown countries raise (do not silently filter).
  - `validate_admin_names(df)` → empty admin1 handled per FR-2 drop-rule; empty admin2 → geo-unit falls back to admin1 (log INFO); normalization mapping table (PRD §9.4 admin-2 inconsistencies) lives here.
  - `validate_types(df)` → `fatalities` integer ≥ 0 (raise on NaN/non-numeric), `actor1` string (fill "Unknown", log WARNING), `event_type` cast to category.
  - `validate_dataset(df) -> DataFrame` orchestrates all of the above in order and returns the clean, typed, normalized frame.
- **Depends on:** M3.
- **Done-when:** each rule has ≥ 1 passing test + ≥ 1 failing-input test asserting the right exception type; a deliberately corrupt fixture raises with a message containing the offending value/row.

---

### M5 — Feature Engineering
- **Goal:** **every feature in PRD §11.3**, computed as-of-date (past-only), output `data/processed/features.parquet`.
- **Files:** `src/feature_engineer.py`, `tests/test_feature_engineer.py`.
- **Detail (each group = one function, all take `as_of` date and never look forward):**
  - Volume: event count 7/14/30d; fatality count 7/14/30d; `log1p` transforms.
  - Velocity: `events_last_7d − events_prior_7d`; same for 14/30.
  - Diversity: Shannon entropy over event types (7/14/30d) — `-Σ p·log p`, zero events → 0.
  - Actors: distinct `actor1` count (14/30d).
  - Recency: days since last event in the district (NA if none → large sentinel, config-gated).
  - Volatility: rolling mean & std of daily fatalities (14/30d).
  - Persistence: number of days with ≥1 event in last 7d.
  - Calendar: `month`, `day_of_week` (from `event_day`), `is_weekend`.
  - Spillover (config-gated `SPILLOVER_ENABLED`, PRD FR-13): sum of event counts in same-admin1 neighbors (14d), computed from the district master table.
  - **Geo aggregation (FR-3):** district master table `geo_unit → admin1 → country` produced here and saved to `data/processed/`; min-events filter applied (drop units with < 5 events, log INFO).
  - Column naming convention: `{feature}_w{window}d` (e.g., `events_w7d`), `entropy_w14d`, `actor_div_w30d`, `days_since_event`, `fat_mean_w14d`, `fat_std_w30d`, `persistence_7d`, `spillover_w14d`.
- **Depends on:** M4.
- **Leakage guarantee:** tests build a fixture with a known event at date T and assert no feature at date T uses events after T (including across district boundaries); `validate_features()` asserts no NaN except documented sentinels (PRD NFR "no silent NaNs").
- **Done-when:** full feature set present and named per convention; leakage tests green; parquet written and re-readable.

---

### M6 — Label Creation
- **Goal:** PRD §11.2 labels, **future-only**, leakage unit-tested.
- **Files:** `src/label_engine.py`, `tests/test_label_engine.py`.
- **Detail:**
  - Per `geo_unit`, compute the future window `[as_of+1, as_of+LABEL_HORIZON_DAYS]`.
  - `escalation = 1` iff: (future events ≥ 3 AND ≥ 1.5 × trailing 30d median events) **OR** (future fatalities ≥ 5).
  - Empty-history fallback: `escalation = 1` iff future events ≥ 5.
  - Rows whose 14-day future window is incomplete (near end of dataset) are **dropped** (`INCOMPLETE_WINDOW = "drop"`) — logged INFO; this is what prevents label leakage at the tail.
  - `label_engine.create_labels(features_df) -> df` merges labels back onto features and writes `data/processed/labelled.parquet`.
- **Depends on:** M5.
- **Leakage tests:** (a) a spike only *after* the horizon does not flip a label computed earlier; (b) a spike *inside* the window flips it; (c) trailing-30d median uses only pre-as_of events; (d) incomplete-window rows excluded.
- **Done-when:** all four leakage tests green; labelled dataset row count = features minus tail rows; class balance printed to log (expected low positive rate, PRD §4.1).

---

### M7 — Train / Validation / Test Split
- **Goal:** strict chronological split; **no shuffle, no random CV, no oversampling across the boundary** (PRD §11.5 — non-negotiable).
- **Files:** `src/split.py`, `tests/test_split.py`.
- **Detail:**
  - Sort by `as_of` date; cut at the quantiles implied by `SPLIT_RATIOS` over the *date* axis (not row counts, to keep calendars aligned across districts).
  - Return `(train_df, val_df, test_df, cut_dates)` where `cut_dates` are exact and logged.
  - `assert_no_leakage(split)` helper: max train date < min val date < min test date — called in tests and at runtime.
- **Depends on:** M6.
- **Done-when:** boundary assertions green; cut dates documented in logs; test window is the newest ~15% (≈ last 6–9 months, per PRD §11.5).

---

### M8 — LightGBM (primary)
- **Goal:** trainable, savable, loadable LGBM classifier with imbalance handling — the PRD's headline model.
- **Files:** `src/models.py` (LGBM section), `tests/test_models.py` (LGBM part), `src/pipeline.py` (first skeleton), `run_pipeline.py` (first working end-to-end call).
- **Detail:**
  - `build_lgbm(config)` from `LGBM_PARAMS`; `scale_pos_weight` computed from the train-set class ratio when set to `"auto"`.
  - `train_model(X_train, y_train, config, model_family) -> fitted` — fixed `RANDOM_SEED`.
  - `save_model(model, path)` / `load_model(path)` (joblib), each logging INFO.
  - `predict_proba(model, X)` → positive-class probability.
  - `run_pipeline.py` in this milestone runs: load → validate → features → labels → split → **train LGBM** → print validation F1. (Eval/plots come in M10+; this early E2E proves wiring.)
- **Depends on:** M7.
- **Done-when:** smoke-train completes on synthetic data in < 60 s; `models/escalation_lgbm.pkl` round-trips through save/load with identical predictions; validation F1 printed.

---

### M9 — XGBoost + Model Comparison + Threshold Optimization ✅ (2026-08-07)
- **Goal:** XGBoost trained on **identical features and split**; formal head-to-head vs LightGBM; threshold optimization; baselines (PRD §11.4/§11.6, FR-14 — **expanded by the user to include the threshold sweep + 4-baseline comparison the plan had placed in M10**; see §3.4 M9 note).
- **Files:** `src/models.py` (XGB family, `full_metrics`, `threshold_analysis`, 4 baselines, `select_winner`), `src/pipeline.py` (`compare_stage`, `_load_lgbm_verified`, report writer), `tests/test_models.py` (+36), `config.py` (threshold grid, comparison constants, baseline columns), `run_pipeline.py` (`--stage compare`); artifacts `models/escalation_xgb.pkl`, `models/escalation_best.pkl`, `models/model_comparison.json`, `reports/model_comparison.md`.
- **Detail:** `train_model(..., family=...)` supports LightGBM/XGBoost on identical data (seed 42, same split, same `scale_pos_weight`); `_load_lgbm_verified` reloads the M8 artifact and **proves val metrics unchanged** (drift → `ModelError`); full metric set (precision/recall/F1/PR-AUC/ROC-AUC/Brier/log-loss/confusion) at 0.5 and at each sweep point; baselines (majority, always-positive, persistence, event-count heuristic); `select_winner` per PRD priority F1 → PR-AUC → Brier → `MODEL_SIMPLICITY_ORDER`; operating threshold = argmax-F1 (`OPERATING_THRESHOLD_MODE=max_f1`) with documented rationale; test split never read during selection.
- **Depends on:** M8.
- **Done-when:** ✅ both models trained on identical data (LGBM verified unchanged); comparison table + sweep + baselines written to JSON/MD; **winner = XGBoost** (val F1 0.8423 @ threshold 0.25 vs LGBM 0.8400 @ 0.20; PR-AUC 0.9031 vs 0.9004); `models/escalation_best.pkl` saved; all 4 baselines beaten; 199/199 tests, 96.84% coverage.

---

### M10 — Evaluation
- **Goal:** the full suite (build rule) + PRD §11.6 protocol + 3 baselines + threshold analysis → `reports/model_metrics.md`.
- **Files:** `src/evaluate.py`, `tests/test_evaluate.py`, `reports/*`.
- **Detail:**
  - `compute_metrics(y_true, y_prob, threshold)` → precision, recall, F1, **AUC-PR**, **Brier score** (+ accuracy only as a listed-but-flagged stat, PRD §4.1).
  - Plots (all `matplotlib`-based, saved to `reports/`, rendered with `show=False` for testability): confusion matrix heatmap, classification report (as table too), ROC curve, precision-recall curve.
  - `threshold_analysis(y_true, y_prob)` → F1-vs-threshold sweep; operating threshold = argmax F1 (`OPERATING_THRESHOLD_MODE`); metrics reported at **0.5 and at the chosen threshold** (FR-14).
  - Baselines (FR-8, PRD §11.6): `majority_baseline` (always majority class), `persistence_baseline` (district escalated last 14d → predict escalation), `threshold_baseline` (events_last_14d ≥ heuristic cutoff from config). Each scored with the same `compute_metrics`; **lift vs majority** computed (PRD §4.1 ≥ 1.5× target).
  - `write_metrics_report(...)` → `reports/model_metrics.md`: model cards for winner + runner-up, baseline comparison table, threshold analysis, cut dates, seed, data provenance pointer.
- **Depends on:** M9.
- **Done-when:** all metrics/baselines/plots produced on synthetic + real data; md report exists and contains every mandated metric; tests assert metric correctness on hand-computable fixtures.

---

### M11 — SHAP Explainability ✅ (2026-08-07; delivered as the user's M10)
- **Goal:** PRD FR-9 + build-rule full set: summary, feature importance, waterfall, dependence, top drivers.
- **Files (delivered):** `src/explainability.py`, `tests/test_explainability.py` (16), `reports/shap_summary.md`, `reports/shap/` (21 plots: summary, bar, 10 dependence, 9 waterfalls), `src/exceptions.py` (+`ExplainabilityError`), `config.py` (SHAP constants + checks), `run_pipeline.py` (`--stage explain`).
- **Detail:**
  - `shap_values(model, X_sample)` — TreeExplainer (works for both LGBM/XGB); sample cap config-gated for speed (e.g., 2,000 rows, logged).
  - Artifacts: beeswarm **summary** (top-15 features), **feature importance** bar (mean |SHAP|), **waterfall** for the highest-risk correctly-flagged district in the test window (config: pick by predicted probability where y_true=1), **dependence plots** for top 3 features, **top_drivers** → markdown table of top-10 features with mean |SHAP|.
  - Produces the PPT narrative hook: one flagged district + its waterfall (PRD §12 demo narrative).
- **Depends on:** M10.
- **Done-when:** all SHAP artifacts exist and are non-empty; test asserts SHAP values shape matches `(n_samples, n_features)` and artifact files exist.

---

### M12 — Risk Map & Visualization ✅ (2026-08-07)
- **Goal:** FR-10 + build-rule visual set: interactive risk map, country trend plots, hotspot analysis, temporal charts — implemented as **M11** in the user's milestone numbering.
- **Files:** `src/visualization.py`, `tests/test_visualize.py`; artifacts `reports/maps/risk_map.html`, `reports/dashboard/country_dashboard.html`, `reports/hotspots_ranking.csv`, `reports/risk_summary.md`, 11 figures in `reports/figures/`; `config.py` (risk bands/colors/dirs/DPI/hotspot/temporal constants + validation), `run_pipeline.py` (`--stage visualize`), `tests/test_config.py` (+8).
- **Detail:**
  - **Risk map:** folium CircleMarkers at geo-unit centroids (from `cleaned_events`), colored by config-driven risk category (Low/Medium/High/Critical from `RISK_LEVEL_BOUNDARIES`), radius ∝ predicted probability, popup = geo unit · country · probability · predicted class · top-3 SHAP drivers · recent 7d events · recent 7d fatalities + HTML legend → `reports/maps/risk_map.html` (PRD §13).
  - **Country dashboard:** 4-metric matplotlib figure (avg risk, positive rate, mean fatalities, mean events) @300 dpi + interactive plotly HTML.
  - **Hotspot analysis:** top-20 ranking CSV + horizontal bar + weekly risk heatmap (trailing `HOTSPOT_HEATMAP_WEEKS`).
  - **Temporal trends:** weekly + monthly avg risk, rolling evolution timeline, country-wise monthly comparison — 4 PNGs.
  - **Importance dashboard:** top-20 SHAP bar (full-window SHAP) + category-wise family contribution.
  - **Prediction distribution:** histogram + scipy KDE + risk-category bars.
  - All figures saved at `FIGURE_DPI=300` via Agg; headless-safe; every function ≤60 lines.
  - **Leakage:** model never retrained; predictions + SHAP on the held-out test window; SHAP↔prediction log-odds reconstruction checked (tol 1e-2).
- **Depends on:** M10 (winner + SHAP).
- **Done-when:** 11/11 PNGs @300 dpi verified (PIL + dpi metadata); `risk_map.html` (leaflet) + `country_dashboard.html` (plotly) open-valid; ranking CSV sorted desc; `risk_summary.md` sections present; **243/243 tests, 96.52% coverage**; M1–M10 suites unaffected.

---

### M13 — Documentation & Final Validation ✅ (2026-08-07, COMPLETE)
- **Goal:** PRD FR-11/FR-12/FR-15 + professional README + **final validation checklist (§12) fully green**. Implemented as **M12 + M13** in the user's milestone numbering.
- **Files (all done):** `README.md` (all 29 mandated sections incl. ACLED attribution — PRD §18.2), `docs/{architecture,model,usage,results}.md`, `scripts/generate_diagrams.py`, `docs/images/` (4 diagrams + 12 screenshots from real outputs), `notebooks/01_EDA.ipynb` / `02_Feature_Engineering.ipynb` / `03_Modeling.ipynb` (executed via nbconvert, 0 errors, plots embedded), `scripts/validate_project.py` (automates §12 checks), `FINAL_AUDIT.md`, `LICENSE` (MIT).
- **Detail:**
  - README mandated sections: **Installation · Architecture · Pipeline · Usage · Folder Structure · Model · Evaluation · Results · Future Work · Attribution** (ACLED attribution line mandatory — PRD §18.2) — all present, metrics verified against `model_comparison.json`.
  - Notebooks: `01_EDA` (coverage, event-type mix, hotspots, trends — FR-12), `02_Feature_Engineering` (features/labels, distribution plots, leakage sanity), `03_Modeling` (split → winner evaluation → SHAP → risk snapshot). All three import from `src/`; executed via `jupyter nbconvert --to notebook --execute` during validation; `%matplotlib inline` re-asserted per cell because `src/explainability.py` forces the Agg backend at import.
  - `scripts/validate_project.py` runs 29 checks in §12 and prints a PASS/FAIL report — **29/29 PASS**.
  - `FINAL_AUDIT.md` documents the full acceptance checklist and repository readiness.
- **Depends on:** M12 (user numbering M11).
- **Done-when:** validator 29/29 PASS; notebooks executed error-free with embedded plots; GitHub scan clean (no secrets, `.gitignore` correct); **243/243 tests, 96.46% coverage**; no source code changed in M13 (no critical bugs found).

---

## 5. Cross-cutting engineering standards (applied to every file)

### 5.1 Logging
- `src/logging_config.py` — `setup_logging()`: console handler (INFO) + `RotatingFileHandler` (`logs/project.log`, 5 MB × 3 backups, WARNING/ERROR also mirrored). Module loggers via `logging.getLogger(__name__)`; every major step logs INFO, every recovered issue WARNING, every raised error ERROR (with traceback) before re-raise.

### 5.2 Error handling
- All domain errors derive from `src/exceptions.py::ConflictForecastError`. Messages must name the offending file/column/value/row where applicable. **Never** `except: pass`; never crash bare — always raise a descriptive exception after logging.

### 5.3 Configuration discipline
- `config.py` is the only place constants live. Modules receive `config` (module import is fine since it's root-level and stable). Tests may monkeypatch config for edge cases; no literal paths/thresholds in `src/` code.

### 5.4 Code style
- Python 3.11+, PEP8, full type hints on every function signature (incl. return types), docstrings on every function (one-line + args/returns for non-trivial), functions ≤ 60 lines, classes ≤ 300 lines, composition over inheritance, no duplicated logic (shared helpers in the owning module, imported where needed), no TODOs/placeholders (grep gate).

---

## 6. Testing strategy

### 6.1 Fixtures (`tests/conftest.py`)
- `make_events_fixture()` — deterministic, seeded synthetic ACLED-schema frame (a few districts across ~400 days, known event spikes at known dates, known actors/types/fatalities) used to make hand-computable expectations possible in feature/label tests.
- `make_corrupt_fixture()` — for validation exception tests (bad coords, dupes, unknown country, non-numeric fatalities, unparsable date).
- `make_features_fixture()` — small pre-computed feature frame for split/model/eval tests (keeps model tests fast, < 60 s).

### 6.2 Coverage gate
- **≥ 80% per `src/` module** via `pytest --cov=src.<module> --cov-fail-under=80` in each milestone's gate; notebooks and `run_pipeline.py` are excluded from coverage (they are thin orchestrators).

### 6.3 Test matrix (module → milestone → what is proven)
| Test file | M | Proves |
|---|---|---|
| `test_config.py` | 2 | all constants exist, typed, valid ranges |
| `test_data_loader.py` | 3 | CSV load, schema rename map, API-arg validation, missing-file error |
| `test_data_validation.py` | 4 | each rule + exception type/message for corrupt inputs |
| `test_feature_engineer.py` | 5 | every feature present/valued; **no-future-leakage** |
| `test_label_engine.py` | 6 | PRD §11.2 logic; **future-only**; incomplete-window drop |
| `test_split.py` | 7 | chronological order, boundary assertion, no shuffle |
| `test_models.py` | 8–9 | train/save/load round-trip, seed determinism, winner selection |
| `test_evaluate.py` | 10 | metric correctness vs hand-computed values, baselines, threshold sweep |
| `test_explain.py` | 11 | SHAP shapes, artifact existence, top-driver table |
| `test_visualize.py` | 12 | files created, HTML structurally valid, headless-safe |

---

## 7. Post-milestone validation protocol (executed after EVERY milestone, before continuing)

1. **Syntax:** `python -m py_compile config.py src/*.py run_pipeline.py`
2. **Imports:** `python -c "import config; import src.<milestone modules>"` (each module individually)
3. **Dependencies:** `pip check` + `pip freeze | diff` against `requirements.txt` (no unlisted imports — checked via `python -c` import of every dependency used by the milestone)
4. **Unit tests:** `pytest tests/test_<module>.py --cov=src.<module> --cov-fail-under=80 -q` → all green
5. **Path/file existence:** automated check that every file the milestone claims to produce exists and is non-empty (`scripts/validate_project.py` grows with each milestone)
6. **Configuration:** `python -c "import config; config.validate_config()"`

Only when all six pass do I proceed to the next milestone and report back for approval.

---

## 8. External prerequisites (blocking items you must provide)

| # | Item | Needed by | What I need from you |
|---|---|---|---|
| P1 | **Python 3.11 or 3.12** installed on this machine | M1 | Confirm `python --version` (I will check) |
| P2 | **ACLED account + data** | M3 real-data validation | Either (a) a **Data Export Tool CSV** for the 4 South Asia countries placed in `data/raw/`, or (b) an **ACLED API key** for `.env` (never committed). Until this arrives, the pipeline is fully developed and tested on synthetic fixtures that match the ACLED schema — a drop-in swap when real data lands |
| P3 | Internet access for `pip install` | M1 | — |
| P4 | `git` (if you want version control / GitHub submission) | M1 | Optional; I'll `git init` only with your OK |

---

## 9. Requirements traceability matrix

| PRD requirement | Milestone | File(s) |
|---|---|---|
| FR-1 Data ingestion (CSV + paged API, cache, gitignored) | M3 | `src/data_loader.py` |
| FR-2 Data validation (schema, missing, dupes, dates, coords, names, types) | M4 | `src/data_validation.py` |
| FR-3 Geo aggregation + district master | M5 | `src/feature_engineer.py` |
| FR-4 Feature engineering → parquet | M5 | `src/feature_engineer.py` |
| FR-5 Label construction (future-only) | M6 | `src/label_engine.py` |
| FR-6 Chronological split, documented cut dates | M7 | `src/split.py` |
| FR-7 LGBM + XGB, imbalance, save/load | M8–M9 | `src/models.py` |
| FR-8 Evaluation + baselines → `model_metrics.md` | M10 | `src/evaluate.py` |
| FR-9 SHAP summary + worked example | M11 | `src/explain.py` |
| FR-10 Risk map → `risk_map.html` | M12 | `src/visualize.py` |
| FR-11 Reproducibility (requirements, seeds, README re-run) | M1, M13 | `requirements.txt`, `config.py`, `README.md` |
| FR-12 EDA notebook | M13 | `notebooks/01_eda.ipynb` |
| FR-13 Spillover features (config-gated) | M5 | `src/feature_engineer.py` |
| FR-14 Threshold tuning (0.5 vs max-F1) | M10 | `src/evaluate.py` |
| FR-15 Multi-horizon note (7d/30d) | M13 | `reports/multi_horizon_notes.md` |
| §11.2 Label definition (exact thresholds) | M6 | `src/label_engine.py` |
| §11.3 All 9 feature groups | M5 | `src/feature_engineer.py` |
| §11.4 LightGBM primary / XGB baseline | M8–M9 | `src/models.py` |
| §11.5 Split protocol (no shuffle, no random CV) | M7 | `src/split.py` |
| §11.6 Eval protocol (5 metrics + 3 baselines + 2 thresholds) | M10 | `src/evaluate.py` |
| §12 SHAP summary + waterfall + demo narrative | M11 | `src/explain.py` |
| §12 Risk map hover = district + top-3 drivers | M12 | `src/visualize.py` |
| NFR performance (≤ 30 min training) | M8–M9 | config params, tests |
| NFR determinism (seeds everywhere) | M2, M8–M9 | `config.py`, `src/models.py` |
| NFR compliance (attribution, 5k pagination) | M3, M13 | `data_loader.py`, `README.md` |
| NFR security (key in `.env`, gitignored) | M1 | `.gitignore`, `.env.example` |
| NFR maintainability (separation, one config source) | M2 | `config.py` |
| §16 Q1–Q5 open questions | M1/M5 | resolved via config defaults (see §11) |

---

## 10. Risks & mitigations (build-phase)

| # | Risk | Mitigation |
|---|---|---|
| R1 | Real ACLED data not available yet | Pipeline fully validated on schema-faithful synthetic fixtures; real CSV/API key is a drop-in swap at M3; data/README.md documents exactly where to drop it |
| R2 | Python 3.13 wheel gaps on Windows (xgboost/shap) | Pin to 3.11/3.12; check `pip install` in M1; fallback: older compatible pins in `requirements.txt` |
| R3 | SHAP runtime on full data | Config-gated sample cap (2,000 rows) for SHAP; only test-window rows needed for waterfall |
| R4 | Class imbalance → low recall | `scale_pos_weight` from train ratio; threshold tuning (M10); PRD's absolute-rule label fallback (M6); PRD §13 hard-exit rule respected |
| R5 | Notebook drift vs `src/` | Notebooks only call `src/`; executed end-to-end in M13 validation; any code change re-executes notebooks before commit |
| R6 | Coverage < 80% on viz/explain modules | Plot/HTML generation functions are pure (save-to-dir, `show=False`) → trivially testable; files checked for existence/non-emptiness |
| R7 | Windows path/encoding quirks | All paths via `config.py` + `pathlib`; UTF-8 reads explicit; logs to `logs/` not stdout-only |
| R8 | Scope creep mid-build | Anything not in §2 tree is out of scope; new ideas go to §12 "Future Work" in the README only |

---

## 11. Decisions to confirm at approval (defaults in bold — all from PRD)

| # | Decision | Default |
|---|---|---|
| D1 | **Escalation thresholds** (§11.2) | **events ≥ 3 AND ≥ 1.5× trailing-30d median, OR fatalities ≥ 5; absolute fallback ≥ 5 events** |
| D2 | Geo unit | **Admin-2 default, admin-1 fallback for units < 5 events** |
| D3 | Date range | **Last 3 full years + current partial year** (5 years = stretch) |
| D4 | Protest/riot in label | **Included** (per PRD §16 Q4 default) |
| D5 | Risk map tech | **Plotly `scatter_geo` (no token)**; Folium choropleth only if a district GeoJSON is available |
| D6 | Build location | **Repo root = this folder** (`army/`), PRD.md + guide stay at top level |
| D7 | Data path (P2) | **CSV export primary**; API secondary; I proceed on synthetic fixtures until real data is provided |
| D8 | `git init` | **Yes** (recommended for submission) — confirm if you want it |

---

## 12. Final acceptance checklist (run at M13, every box must be green)

- [ ] Every file in §2.1 exists
- [ ] Every import works (`python -c` import sweep of `config.py`, all `src/` modules, `run_pipeline.py`)
- [ ] Every notebook executes top-to-bottom (`jupyter nbconvert --execute`, no errors, outputs committed)
- [ ] No missing dependency (`pip check` green; imports match `requirements.txt`)
- [ ] No circular imports (import sweep catches this)
- [ ] No syntax errors (`py_compile` sweep)
- [ ] No undefined variables / dead code (pytest collection + `ruff` if enabled)
- [ ] No placeholders: grep for `TODO`, `FIXME`, `XXX`, `pass  # stub`, `NotImplementedError` → zero hits
- [ ] Config loads and validates (`config.validate_config()`)
- [ ] Pipeline runs end-to-end (`python run_pipeline.py` on real or synthetic data — success log)
- [ ] Model trains successfully; artifacts exist (`models/escalation_best.pkl` + manifest)
- [ ] Reports generated: `model_metrics.md`, all PNGs, `risk_map.html` (non-empty)
- [ ] Risk map opens (HTML structure valid; manual visual check by you)
- [ ] SHAP plots generate (summary, waterfall, dependence, top-drivers)
- [ ] README matches implementation (structure, commands, filenames, metrics all truthful)
- [ ] ACLED attribution present in README
- [ ] Logs written to `logs/project.log`; `logs/` gitignored
- [ ] Secrets safe: no `.env`/key in git, `.gitignore` correct
- [ ] `pytest --cov=src --cov-fail-under=80` green
- [ ] Coverage report generated (`coverage html`/`xml` in `reports/`)

---

## 13. Approval gate

This plan is ready for your review. On approval I will:

1. Start **M1 (Repository structure)** only.
2. After M1's six-point validation (§7) passes, report what was built, files changed, tests passed, what remains — and **stop for your approval**.
3. Proceed one milestone at a time, never ahead of schedule.

To change anything: tell me which section/decision to amend **before** I start M1 — no code has been written yet, so the plan is fully malleable at this point.
