# Running the Project — Complete Guide for New Users

This guide walks you through running the **Conflict Escalation Forecasting** project from a completely fresh clone. No prior knowledge of the project is assumed. Every command in this guide has been executed and verified on a clean machine (Windows 11, Python 3.14.3).

> **Time to complete a full run:** roughly 5–10 minutes on a typical laptop.

---

## 1. Project Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **Operating system** | Windows 10/11, macOS 12+, Ubuntu 20.04+ (any OS with Python) | Same |
| **Python version** | 3.11 | 3.11–3.14 (verified on 3.14.3) |
| **Disk space** | ~500 MB free | 1 GB free |
| **RAM** | 4 GB | 8 GB+ |
| **Internet** | Required for `pip install` and for the risk map's OpenStreetMap tiles when opened | Same |
| **Optional** | Git (to clone); a browser (to view HTML dashboards) | — |

**Project size when fully built:** raw data ~17 MB, processed data ~60 MB, models ~7 MB, reports ~8 MB.

---

## 2. Clone Repository

```bash
git clone <your-repo-url> army
cd army
```

If you do not use Git, download the repository as a ZIP and extract it, then open a terminal in the extracted folder.

> **Important:** from here on, every command runs from inside the project root (the folder containing `run_pipeline.py` and `config.py`).

---

## 3. Create Virtual Environment

A virtual environment keeps this project's dependencies isolated from your system Python.

### Windows (Command Prompt / PowerShell / Git Bash)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Windows (Git Bash)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation you should see `(.venv)` at the start of your prompt.

---

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**What this installs:**

- **Core data stack:** `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`
- **Machine learning:** `lightgbm`, `xgboost`, `shap`
- **Visualization:** `matplotlib`, `plotly`, `folium`
- **Utilities:** `requests`, `python-dotenv`
- **Notebooks (optional but recommended):** `jupyter`
- **Testing:** `pytest`, `pytest-cov`

All versions are **exact-pinned** in `requirements.txt` so every machine gets the identical environment. If a specific wheel is unavailable for your Python version, `pip` will report which package failed — see [Troubleshooting](#10-troubleshooting).

---

## 5. Configure Environment

### `.env` file (optional — not required to run)

This project **does not call any external API**, so the `.env` file is **not required**. The template `ACLED_API_KEY=` in `.env.example` is reserved for future API-based data refresh. If you want to prepare it anyway:

```bash
cp .env.example .env
```

### `config.py` (the real configuration)

Everything the pipeline needs is in `config.py` — there are **no hardcoded values** anywhere else. You normally do **not** need to edit anything to run the project.

| Setting | Default | What it controls |
|---|---|---|
| `COUNTRIES` | India, Pakistan, Afghanistan, Myanmar, Sudan, South Sudan | Scope of the analysis |
| `DATE_START` / `DATE_END` | 2016-01-01 / 2026-12-31 | Study window (inclusive) |
| `LABEL_HORIZON_DAYS` | 14 | Forecast horizon for the escalation label |
| `SPLIT_RATIOS` | train 0.70 / val 0.15 / test 0.15 | Chronological split |
| `RANDOM_SEED` | 42 | Reproducibility |
| `RISK_LEVEL_BOUNDARIES` | (0.2, 0.4, 0.6) | Risk category bands (Low/Medium/High/Critical) |
| `FIGURE_DPI` | 300 | Publication-quality figures |

`config.validate_config()` runs at the start of every stage and raises a descriptive error if anything is misconfigured.

---

## 6. Dataset Setup

### Where to put the data

Place your ACLED export **CSV file(s) into `data/raw/`**:

```
army/
└── data/
    └── raw/
        └── ACLED Data_2026-08-07_event_date_from_2017-01-01_event_date_to_2026-08-07.csv
```

### Expected filename

Any `.csv` file works — the loader auto-discovers all CSVs in `data/raw/` and merges them. The file used in development is the ACLED weekly aggregated export:

```
ACLED Data_2026-08-07_event_date_from_2017-01-01_event_date_to_2026-08-07.csv
```

### Supported formats (auto-detected)

1. **Weekly aggregated counts** — `week × country × admin1 × event_type` with `events`, `fatalities`, and centroid coordinates (the format used in development).
2. **Event-level rows** — with `event_id_cnty`, `event_date`, `admin2`, `actor1`, per-event coordinates (a drop-in swap; no code changes needed).

### Get the data from ACLED

1. Go to [acleddata.com](https://acleddata.com) → **Data Export Tool**.
2. Select the countries you want (the six supported by default are India, Pakistan, Afghanistan, Myanmar, Sudan, South Sudan).
3. Choose **"Weekly aggregated"** format and your date range (e.g., from 2017-01-01 to today).
4. Download and move the CSV into `data/raw/`.

> Data is used under ACLED's academic-use terms; see `data/README.md` for provenance and attribution.

### Folder structure after setup

```
army/
├── config.py
├── run_pipeline.py
├── requirements.txt
├── data/
│   ├── README.md
│   └── raw/
│       └── ACLED Data_....csv          ← your file goes here
├── src/            # pipeline modules (do not edit)
├── tests/          # test suite
├── notebooks/      # EDA / feature / modeling notebooks
└── docs/           # documentation
```

`data/raw/` and the generated `data/processed/`, `models/*.pkl`, and `logs/` are git-ignored (never committed).

---

## 7. Run the Project

Run the pipeline **one stage at a time** (each stage reads the previous stage's outputs):

```bash
python run_pipeline.py --stage ingest     # 1. raw CSVs → validated cleaned events
python run_pipeline.py --stage features   # 2. → rolling-window feature table
python run_pipeline.py --stage labels     # 3. → 14-day escalation labels
python run_pipeline.py --stage split      # 4. → chronological train/val/test
python run_pipeline.py --stage train      # 5. → train LightGBM
python run_pipeline.py --stage compare    # 6. → train XGBoost, compare, pick winner
python run_pipeline.py --stage explain    # 7. → SHAP explainability
python run_pipeline.py --stage visualize  # 8. → risk map + dashboards
python run_pipeline.py --stage forecast   # 9. → LIVE next-14-days forecast CSV + risk map
```

Or run **everything at once**:

```bash
python run_pipeline.py --stage all
```

`--stage all` runs the 9 stages above in dependency order and stops at the first failure with a descriptive error.

### What each stage does

| Stage | Reads | Writes | Purpose |
|---|---|---|---|
| `ingest` | `data/raw/*.csv` | `cleaned_events`, `district_master` | Discover + merge CSVs, validate 9 data-quality rules, build the district master table |
| `features` | `cleaned_events.parquet` | `features` (+ `feature_summary.md`) | Compute leakage-safe rolling-window features (7/14/30-day counts, velocity, volatility, entropy, spillover…) |
| `labels` | `features.parquet` | `labeled_features` (+ `label_summary.md`) | Attach the future-only 14-day escalation label |
| `split` | `labeled_features.parquet` | `split_train/val/test` (+ `split_summary.md`) | Chronological split (no shuffle, no leakage) |
| `train` | `split_train/val` | `escalation_lgbm.pkl`, `manifest.json` | Train the LightGBM model |
| `compare` | `splits` + LGBM model | `escalation_xgb.pkl`, `escalation_best.pkl`, `model_comparison.json/.md` | Train XGBoost, full metric comparison, threshold sweep, 4 baselines, winner selection |
| `explain` | `split_test` + winner | `reports/shap/*`, `shap_summary.md` | SHAP importance, waterfall, dependence plots |
| `visualize` | `split_test` + winner + centroids | `reports/maps/*`, `reports/figures/*`, `reports/dashboard/*`, `risk_summary.md` | Interactive risk map + dashboards |
| `forecast` | `features.parquet` (latest rows) + winner | `forecast_next_14_days.csv`, `maps/forecast_risk_map.html`, `forecast_summary.md` | **Live** next-14-days forecast per geo unit (not the test window) |

---

## 8. Expected Outputs

After a full run, you should see the following generated files:

### `data/processed/` (14 files)

```
cleaned_events.{parquet,csv}      # validated event rows (127,052 × 14)
district_master.{parquet,csv}     # one row per geo unit (124 × 9)
features.{parquet,csv}            # engineered features (44,146 × 37)
labeled_features.{parquet,csv}    # features + labels (43,981 × 38)
split_train.{parquet,csv}         # 30,790 rows
split_val.{parquet,csv}           # 6,522 rows
split_test.{parquet,csv}          # 6,669 rows
```

### `models/`

```
escalation_lgbm.pkl      # trained LightGBM
escalation_xgb.pkl       # trained XGBoost
escalation_best.pkl      # winning model (XGBoost)
manifest.json            # training params + metrics
model_comparison.json    # full comparison + threshold sweep + baselines
```

### `reports/`

```
feature_summary.md       label_summary.md          label_timeline.png
split_summary.md         model_comparison.md       shap_summary.md
risk_summary.md          hotspots_ranking.csv
forecast_next_14_days.csv          # LIVE 14-day forecast per geo unit
forecast_summary.md                # forecast report (hotspots, country averages, drivers)
figures/                 # 11 PNGs @ 300 dpi (dashboards, hotspots, trends, distributions)
shap/                    # 21 PNGs (summary, bar, waterfall, dependence plots)
maps/risk_map.html       # interactive folium risk map
maps/forecast_risk_map.html       # interactive forecast risk map
dashboard/country_dashboard.html   # interactive plotly dashboard
```

### `docs/` (static documentation — always present)

```
README.md  architecture.md  model.md  usage.md  results.md  RUN_PROJECT.md
images/    # diagrams + screenshots
```

### `logs/`

```
project.log   # rotating log of every stage (INFO/WARNING/ERROR)
```

---

## 9. Verify Successful Execution

After the run, check these markers:

| Check | Expected result |
|---|---|
| Exit code | `0` (or the final print says `Pipeline stage ... complete`) |
| `data/processed/cleaned_events.parquet` exists | ~127,052 rows, 124 geo units, 6 countries |
| `models/escalation_best.pkl` exists | winner = XGBoost |
| `models/model_comparison.json` | `"winner": "xgboost"`, `"operating_threshold": 0.25` |
| `reports/model_comparison.md` | validation F1 **0.8423** @ 0.25, PR-AUC **0.9031** |
| `reports/shap/` | **21** PNG files |
| `reports/maps/risk_map.html` | exists; opens in a browser (tiles need internet) |
| `reports/dashboard/country_dashboard.html` | exists; opens in a browser |
| `reports/figures/` | **11** PNG files @ 300 dpi |
| `reports/risk_summary.md` | highest-risk unit = Khyber Pakhtunkhwa (Pakistan) |
| `reports/forecast_next_14_days.csv` | 124 rows (one per geo unit), proba in [0,1] |
| `reports/maps/forecast_risk_map.html` | exists; opens in a browser |
| `logs/project.log` | no ERROR lines |

**Quick automated check** (runs the full 30-check audit):

```bash
python scripts/validate_project.py
# expected: RESULT: 30/30 checks passed
```

**Run the test suite** (243 tests):

```bash
python -m pytest
```

**Run the executed notebooks** (optional — re-executes and saves outputs):

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_Feature_Engineering.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_Modeling.ipynb
```

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'config'`

- **Cause:** you are running Python from outside the project root.
- **Fix:** `cd` into the project root (the folder with `config.py`) and re-run. Make sure you activated the venv first.

### `ModuleNotFoundError: No module named 'numpy'` (or similar)

- **Cause:** dependencies not installed (or wrong interpreter).
- **Fix:** activate the venv and run `pip install -r requirements.txt`. Verify with `python -c "import numpy"`.

### Missing dataset: `DataLoadError: Cleaned dataset not found ... run the 'ingest' stage first`

- **Cause:** you skipped a stage, or `data/raw/` is empty.
- **Fix:** put the ACLED CSV(s) in `data/raw/`, then run `python run_pipeline.py --stage ingest` first. Stages are strictly sequential.

### `DataLoadError: No raw data found in data/raw`

- **Cause:** no `.csv` file in `data/raw/`.
- **Fix:** download an ACLED export and place it in `data/raw/` (see [Dataset Setup](#6-dataset-setup)).

### `Winning model not found: models/escalation_best.pkl`

- **Cause:** you ran `explain` / `visualize` before `compare`.
- **Fix:** run `python run_pipeline.py --stage compare` first, or just `python run_pipeline.py --stage all`.

### Wrong Python version

- **Cause:** the system Python is older than 3.11, or `python` points elsewhere.
- **Fix:** install Python 3.11+ and create the venv with it explicitly: `python3.12 -m venv .venv`. Check with `python --version`.

### `pip install` fails on a pinned package (e.g., numpy 2.4.6)

- **Cause:** no wheel for your exact Python/OS combination (rare; the pins are verified on 3.14).
- **Fix:** use Python 3.11–3.14, or relax that one pin (`pip install numpy`). Report the exact failing line.

### LightGBM issues

- **Symptom:** `OSError: could not find library "lib_lightgbm"` (Windows, rare).
- **Fix:** reinstall: `pip uninstall lightgbm && pip install lightgbm`. Ensure the venv is active.

### XGBoost issues

- **Symptom:** DLL/`libgomp` errors on Windows.
- **Fix:** reinstall `pip install --force-reinstall xgboost`. On Linux, `apt install libgomp1` if missing.

### SHAP issues

- **Symptom:** `ExplainabilityError: SHAP computation failed` or slow runs.
- **Fix:** this is usually a version mismatch — `pip install shap==0.52.0`. If the model file is corrupt, re-run `python run_pipeline.py --stage compare` to regenerate it.

### File path issues on Windows

- **Symptom:** errors mentioning `C:Users...` or backslashes.
- **Fix:** always run commands from the project root. The pipeline builds all paths from `config.py` (relative to the repo), so paths work on any OS.

### Risk map opens but shows a blank background

- **Cause:** OpenStreetMap tiles require internet.
- **Fix:** connect to the internet. Markers, popups, and the legend still render offline.

### `ConfigurationError: SPLIT_RATIOS must sum to 1.0` (or similar)

- **Cause:** `config.py` was edited inconsistently.
- **Fix:** restore the default values or fix the invalid setting — the error message names the exact problem.

---

## 11. Clean Re-run

To remove **all generated artifacts** and rebuild from scratch (raw data in `data/raw/` is kept):

```bash
# Remove processed datasets, models, reports, and logs
rm -rf data/processed models/escalation_lgbm.pkl models/escalation_xgb.pkl \
       models/escalation_best.pkl models/manifest.json models/model_comparison.json \
       reports/figures reports/maps reports/dashboard reports/shap \
       reports/hotspots_ranking.csv reports/forecast_next_14_days.csv \
       reports/*.md reports/label_timeline.png logs
```

> **Windows (PowerShell):**
> ```powershell
> Remove-Item -Recurse -Force data/processed, reports/figures, reports/maps, reports/dashboard, reports/shap, logs
> Remove-Item -Force models/escalation_*.pkl, models/manifest.json, models/model_comparison.json, reports/*.md, reports/hotspots_ranking.csv, reports/forecast_next_14_days.csv, reports/label_timeline.png
> ```

Then run the whole project again:

```bash
python run_pipeline.py --stage all
```

**Determinism guarantee:** because the pipeline is fully deterministic (fixed seed 42, chronological split, no shuffle), a clean re-run reproduces **byte-identical** model artifacts and identical metrics (validation F1 0.8423, PR-AUC 0.9031, operating threshold 0.25). This was verified during development by hashing the model files before and after a clean re-run.

---

## Next Steps

- Read `README.md` for the full project overview.
- Explore `docs/results.md` for the key findings and screenshots.
- Open `reports/maps/risk_map.html` and `reports/dashboard/country_dashboard.html` in a browser.
- Run the notebooks for a guided walk-through: `python -m jupyter notebook` (or `jupyter lab`).
