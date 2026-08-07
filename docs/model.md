# Model

This document details the modeling approach: features, labels, model selection, threshold optimization, metrics, baselines, and SHAP interpretation. All numbers come from the implemented pipeline (see `reports/`, `models/model_comparison.json`).

---

## 1. Task Framing

- **Problem type:** binary classification — "will this geo unit experience conflict escalation within the next 14 days?"
- **Unit of analysis:** `(geo_unit, as_of_date)` — one row per province per week.
- **Horizon:** 14 days ahead of the feature date.
- **Evaluation protocol:** strict chronological split; the test window (2025-02-08 → 2026-07-11, 6,669 rows) is never used for training or selection.

---

## 2. Features

**33 model features** derived from the raw event counts (see `src/feature_engineer.py` and `reports/feature_summary.md`):

| Group | Features | Rationale |
|---|---|---|
| Event volume | `events_{7,14,30}d` + `events_log1p_{7,14,30}d` | Recent activity level; log1p tames heavy tails |
| Fatality volume | `fatalities_{7,14,30}d` + `fatalities_log1p_{7,14,30}d` | Recent lethality |
| Velocity | `velocity_events_{7,14,30}d`, `velocity_fatalities_{7,14,30}d` | Acceleration/deceleration (current minus prior window) |
| Volatility | `fat_mean_{14,30}d`, `fat_std_{14,30}d` | Average intensity and spikiness of daily fatalities |
| Persistence | `persistence_w7d` | How sustained the activity is (active days) |
| Diversity | `entropy_{7,14,30}d` | Shannon entropy of the event-type mix |
| Recency | `days_since_event` | Time since the last event (sentinel 999 = no history) |
| Calendar | `month`, `day_of_week` | Seasonality |
| Identity | `country_code`, `admin1_code`, `geo_unit_code` | Unit-level baselines |
| Spillover | `spillover_w14d` | Events across the K=3 nearest same-country units (spatial contagion) |

**Leakage guarantee:** every feature at date `as_of` is computed from rows in `[as_of − W, as_of)` only — the current and future weeks are never visible. Verified by a spike-injection test and a seeded randomized property test.

---

## 3. Labels

Binary target built by `src/label_engineer.py` from **future observations only**:

```
escalation = 1  iff  (future_events ≥ 3  AND  future_events ≥ 1.5 × trailing-30d median)
                    OR  future_fatalities ≥ 5
fallback: future_events ≥ 5 when the unit has no trailing history
```

- Future window: `(as_of, as_of + 14d]` — exclusive of the feature date.
- Trailing median: `[as_of − 30d, as_of)` — past-only baseline for the multiplier.
- Rows with an incomplete future window (near the data tail) are **dropped** (134 rows), never partially labeled.
- Distribution: **30,217 positive (68.7%) / 13,764 negative** — a majority-positive, imbalanced target.

All thresholds are configurable in `config.py` (`LABEL_HORIZON_DAYS`, `ESCALATION_MIN_EVENTS`, `ESCALATION_MULTIPLIER`, `ESCALATION_MIN_FATALITIES`, `ABSOLUTE_MIN_EVENTS`, `TRAILING_MEDIAN_WINDOW_DAYS`, `INCOMPLETE_WINDOW`).

---

## 4. Train / Validation / Test

- **Chronological split** at date-axis quantiles 70/15/15 — **no shuffle, no random CV** (time-series integrity).
- Boundary invariant: `max(train) < min(val) < min(test)` (validated per run).

| Split | Rows | Date range | Positive % |
|---|---|---|---|
| Train | 30,790 | 2016-12-31 → 2023-09-02 | 69.2% |
| Validation | 6,522 | 2023-09-09 → 2025-02-01 | 67.7% |
| Test | 6,669 | 2025-02-08 → 2026-07-11 | 67.3% |

---

## 5. Models & Imbalance Handling

Both models are gradient-boosted decision-tree classifiers trained on **identical** features, splits, seed (42), and imbalance handling:

- **Imbalance:** `scale_pos_weight = n_neg / n_pos = 0.4447` (computed from the training set; `IMBALANCE_METHOD="scale_pos_weight"`, configurable).
- **LightGBM:** `n_estimators=500`, `num_leaves=63`, `learning_rate=0.05`, `objective="binary"`, seed 42.
- **XGBoost:** `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `objective="binary:logistic"`, seed 42.
- **Determinism:** fixed seed + no randomness in the pipeline → retraining reproduces identical metrics (proven by test).

Artifacts: `models/escalation_lgbm.pkl`, `models/escalation_xgb.pkl`, `models/manifest.json` (params + features + split cuts + validation metrics).

---

## 6. Threshold Optimization

A probability threshold sweep over **0.10 → 0.90 (step 0.05)** is run on the validation set for each model (`THRESHOLD_MIN/MAX/STEP`, `OPERATING_THRESHOLD_MODE="max_f1"`):

| Point | LightGBM | XGBoost |
|---|---|---|
| Best-F1 threshold | 0.20 | **0.25** |
| Best-F1 (F1 / P / R) | 0.8400 / 0.753 / 0.950 | **0.8423 / 0.771 / 0.928** |
| Best precision point (thr 0.9) | 0.954 | 0.974 |
| Best recall point (thr 0.1) | 0.987 | 0.986 |

**Why 0.25 and not 0.5?** The target is majority-positive (68.7%), so the argmax-F1 operating point sits well below the default 0.5. At 0.5, F1 drops to 0.793 (LGBM) / 0.780 (XGB). The chosen operating threshold maximizes validation F1 — the rationale is documented in `reports/model_comparison.md`.

---

## 7. Metrics (validation)

| Metric | LightGBM | **XGBoost (winner)** |
|---|---|---|
| F1 @ operating threshold | 0.8400 @ 0.20 | **0.8423 @ 0.25** |
| Precision / Recall @ operating | 0.753 / 0.950 | **0.771 / 0.928** |
| PR-AUC | 0.9004 | **0.9031** |
| ROC-AUC | 0.8112 | **0.8175** |
| Brier | **0.1743** | 0.1755 |
| Log loss | **0.5145** | 0.5171 |
| F1 @ 0.5 | 0.7928 | 0.7803 |

**Winner selection (PRD priority):** validation F1 → PR-AUC → Brier → simplicity order. XGBoost wins on F1 (0.8423 > 0.8400) and PR-AUC; it is saved as `escalation_best.pkl`. The selection never reads the test split (verified by a poisoned-split test).

---

## 8. Baselines

All baselines are scored on the validation split at threshold 0.5 (F1 / PR-AUC):

| Baseline | F1 | PR-AUC | Description |
|---|---|---|---|
| Majority | 0.8076 | 0.6772 | Predict the majority class always |
| Always-positive | 0.8076 | 0.6772 | Predict positive always |
| Persistence | **0.8261** | 0.7557 | Repeat the current escalation state |
| Event-count heuristic | 0.8044 | 0.7741 | `events_w14d ≥ 5` |
| **XGBoost (winner @ 0.25)** | **0.8423** | **0.9031** | ML model |

The winner **beats all four baselines** on F1 and PR-AUC. The persistence baseline is the strongest heuristic — sensible for an escalation-persistence signal — but the ML model still improves on it.

---

## 9. SHAP Interpretation

SHAP (TreeExplainer) values are computed on the **held-out test window** (2,000-row even-spaced sample in M10; full window in the visualization stage). The model is **never retrained** for explanation.

**Top-10 drivers (mean |SHAP|):**

| Rank | Feature | mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | `events_w30d` | 0.478 | 30-day event volume — sustained recent activity |
| 2 | `fatalities_w30d` | 0.414 | 30-day fatalities — sustained lethality |
| 3 | `velocity_events_w30d` | 0.309 | 30-day event acceleration |
| 4 | `events_w7d` | 0.177 | Immediate (last-week) activity |
| 5 | `spillover_w14d` | 0.155 | Neighbourhood contagion (FR-13) |
| 6 | `admin1_code` | 0.152 | Unit-level baseline |
| 7 | `events_w14d` | 0.148 | Two-week activity |
| 8 | `velocity_fatalities_w30d` | 0.145 | Fatality acceleration |
| 9 | `fatalities_w14d` | 0.127 | Two-week lethality |
| 10 | `month` | 0.114 | Seasonality |

**Data-driven behaviour observations** (from `reports/shap_summary.md`):
- **30-day windows dominate** the signal (mean-|SHAP| sums: 30d ≈ 1.02 vs 14d ≈ 0.56 vs 7d ≈ 0.26) — escalation is driven by sustained multi-week violence, not just last week's spike.
- **Spatial spillover ranks #5** — neighbouring-province activity is a material risk factor (spatial contagion).
- **Identity codes outweigh calendar** — unit-level baselines (0.23) matter more than seasonality (0.11).

Outputs: `reports/shap_summary.md` + `reports/shap/` (summary/bar/waterfall/dependence plots).
