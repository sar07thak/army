# Final Audit — Conflict Escalation Forecasting

> **Date:** 2026-08-07 · **Milestone:** M13 (Final Audit & Submission Readiness) · **Status:** ✅ COMPLETE

This document is the final acceptance report for the repository. Every number is produced by the implemented pipeline (`run_pipeline.py` + `src/`) and verified by `scripts/validate_project.py` and the test suite.

---

## 1. Project Overview

A reproducible machine-learning pipeline that forecasts **district/province-level conflict escalation risk 14 days ahead** across six countries (India, Pakistan, Afghanistan, Myanmar, Sudan, South Sudan) using ACLED weekly aggregated event data. The product is the ML pipeline itself: ingest → validate → features → labels → split → train → compare → explain → visualize.

**One-line pitch:** *"A geo-unit-level early-warning model that forecasts conflict escalation risk over a rolling 14-day window using engineered temporal + spatial features from ACLED data."*

---

## 2. Architecture Summary

- **Layered, stage-based design** — each of the 9 pipeline stages is an independent module in `src/`, orchestrated by `run_pipeline.py` and connected through validated artifacts on disk.
- **Cross-cutting:** `config.py` (single source of truth), `src/logging_config.py` (rotating file + console), `src/exceptions.py` (`ConflictForecastError` hierarchy).
- **No leakage by construction:** features use half-open `[as_of−W, as_of)` windows; labels use `(as_of, as_of+14d]`; split is chronological with no shuffle; test split is never read during training or selection.
- See `docs/architecture.md` and `docs/images/*` for the full diagrams.

---

## 3. Dataset Summary

| Dataset | Rows | Columns | Notes |
|---|---|---|---|
| raw CSV (ACLED) | 127,353 | 12 | 2017-01-01 → 2026-08-07, weekly buckets |
| cleaned_events | 127,052 | 14 | 6 countries · 124 geo units · 0 missing · 0 duplicates |
| district_master | 124 | 9 | admin1, country, centroids, totals |
| features | 44,146 | 37 | 0 NaN · 0 duplicates |
| labeled_features | 43,981 | 38 | 30,217 positive (68.7%) / 13,764 negative |
| split_train / val / test | 30,790 / 6,522 / 6,669 | 38 | chronological, no overlap |

---

## 4. Feature Summary

**33 model features** per row, all past-only:
- **Volume:** `events_{7,14,30}d` + log1p; **Lethality:** `fatalities_{7,14,30}d` + log1p
- **Velocity:** `velocity_events_{7,14,30}d`, `velocity_fatalities_{7,14,30}d`
- **Volatility:** `fat_mean_{14,30}d`, `fat_std_{14,30}d`
- **Persistence:** `persistence_w7d` · **Diversity:** `entropy_{7,14,30}d` (Shannon)
- **Recency:** `days_since_event` (sentinel 999) · **Calendar:** `month`, `day_of_week`
- **Identity:** `country_code`, `admin1_code`, `geo_unit_code` · **Spillover:** `spillover_w14d`

Leakage proven by spike-injection + seeded randomized property tests.

---

## 5. Label Summary

- **Definition:** `escalation = 1` iff `(future_events ≥ 3 AND future_events ≥ 1.5 × trailing-30d median) OR future_fatalities ≥ 5`; absolute fallback `future_events ≥ 5`.
- **Horizon:** next 14 days, future-only (`(as_of, as_of+14d]`); incomplete windows dropped (134 rows).
- **Balance:** 68.7% positive (majority-positive — handled by threshold tuning).

---

## 6. Model Summary

- **Candidates:** LightGBM & XGBoost on identical features/splits/seed (42)/imbalance handling (`scale_pos_weight = 0.4447`).
- **Winner:** **XGBoost** — `models/escalation_best.pkl`, selected by PRD priority F1 → PR-AUC → Brier.
- **Threshold:** operating threshold **0.25** (argmax-F1 from the 0.10–0.90 sweep), not the default 0.5.
- **Determinism:** retraining reproduces identical metrics (test-proven).

---

## 7. Performance Summary

| Metric | LightGBM | **XGBoost (winner)** |
|---|---|---|
| F1 @ operating threshold | 0.8400 @ 0.20 | **0.8423 @ 0.25** |
| Precision / Recall | 0.753 / 0.950 | **0.771 / 0.928** |
| PR-AUC | 0.9004 | **0.9031** |
| ROC-AUC | 0.8112 | **0.8175** |
| Brier | 0.1743 | 0.1755 |
| Log loss | 0.5145 | 0.5171 |

**Baselines (validation @ 0.5):** persistence F1 0.8261 · majority 0.8076 · always-positive 0.8076 · event-count heuristic 0.8044 — **winner beats all four**.

**Test-window risk snapshot (122 geo units):** Critical 42 · High 23 · Medium 21 · Low 36. Highest-risk **Khyber Pakhtunkhwa (Pakistan, 0.999)**; safest **Zabul (Afghanistan, 0.026)**. Country average risk: Pakistan 0.749 · Myanmar 0.688 · India 0.527 · South Sudan 0.505 · Sudan 0.445 · Afghanistan 0.176.

**Top SHAP drivers:** `events_w30d` (0.478) · `fatalities_w30d` (0.414) · `velocity_events_w30d` (0.309) · `events_w7d` (0.177) · `spillover_w14d` (0.155) — sustained violence and spatial contagion drive escalation.

---

## 8. Visualization Summary

- **Interactive risk map:** `reports/maps/risk_map.html` (folium, 122 markers, risk-category colors, popups with probability/class/top-SHAP drivers/recent events/fatalities).
- **Country dashboard:** `reports/figures/country_dashboard.png` + `reports/dashboard/country_dashboard.html` (plotly).
- **Hotspots:** `reports/hotspots_ranking.csv` (top-20) + bar + weekly heatmap.
- **Temporal:** weekly/monthly average risk, evolution timeline, country-wise comparison.
- **SHAP:** `reports/shap/` (21 plots) + `reports/shap_summary.md`.
- **Figures:** 11 PNGs @ 300 dpi in `reports/figures/`; `reports/risk_summary.md`.

---

## 9. Documentation Summary

| Doc | Contents |
|---|---|
| `README.md` | Full professional README (29 sections incl. attribution) |
| `docs/architecture.md` | Architecture, modules, data flow, design decisions |
| `docs/model.md` | Features, labels, selection, threshold, metrics, baselines, SHAP |
| `docs/usage.md` | Install, CLI, config, expected outputs, troubleshooting |
| `docs/results.md` | Metrics, hotspots, screenshots, key findings |
| `docs/images/` | 4 diagrams + 12 screenshots (from real outputs) |
| `notebooks/` | 3 executed notebooks (EDA, feature engineering, modeling) |

---

## 10. Testing Summary

- **Total tests:** **243** · **Passing:** 243 · **Coverage:** **96.46%** (gate ≥ 80%)
- Per-module coverage: logging_config 100% · exceptions 100% · config 100% · pipeline 99% · data_validation 99% · visualization 99% · feature_engineer 98% · models 95% · data_loader 95% · split 95% · label_engineer 94% · explainability 92%
- Key edge cases: leakage proofs (spike-injection + randomized property), incomplete-window drop, chronological boundaries, threshold grid exactness, winner tie-breaks, drift detection, test-split isolation, SHAP shape checks, PNG/HTML artifact validity, risk-band boundaries.

---

## 11. Repository Statistics

- **Milestones:** M1–M13 complete (13/13).
- **Source modules:** 11 in `src/` + `config.py` + `run_pipeline.py`.
- **Test files:** 12 (`tests/`) · **Docs:** README + 5 docs + FINAL_AUDIT + IMPLEMENTATION_PLAN + PROGRESS.
- **Reports:** 6 markdown summaries + 11 figures + 21 SHAP plots + 2 interactive HTML + 1 ranking CSV.
- **Notebooks:** 3 executed (with embedded plots).
- **Generated artifacts:** models (3 .pkl + manifest + comparison JSON), processed datasets (6 parquet + 6 CSV).

---

## 12. Known Limitations

1. **Province-level granularity** (admin-1 weekly aggregates) — district-level + actor features require an event-level ACLED export (drop-in swap).
2. **Majority-positive labels (68.7%)** — handled via threshold tuning (0.25) and PR-AUC as headline; F1 lift vs majority ≈ 1.04×.
3. **7-day windows at weekly granularity** = previous week's bucket.
4. No actor columns in source → actor-diversity features skipped.
5. Weekly data ⇒ finer daily dynamics not representable.
6. Risk map tiles require internet when opened; markers/popups render offline.

---

## 13. Future Improvements

- Event-level ACLED export (district units, actor features, per-event coords).
- Multi-horizon forecasting (7d/30d) + calibration curves.
- Richer spillover graphs (distance-decay, cross-border).
- Label-definition & threshold sensitivity ablations.
- Daily refresh via fresh ACLED exports (no retraining needed for scoring).

---

## 14. Final PRD Compliance Checklist

| Requirement | Status |
|---|---|
| Complete repository with clean architecture | ✅ |
| Reproducible pipeline (deterministic, config-driven) | ✅ |
| Notebooks (3, executed) | ✅ |
| Trained model + evaluation | ✅ |
| SHAP explainability | ✅ |
| Risk visualization | ✅ |
| Documentation (README + docs + attribution) | ✅ |
| 9 validation rules with meaningful exceptions | ✅ |
| Leakage-safe features & labels | ✅ |
| Chronological split, no shuffle | ✅ |
| LightGBM + XGBoost comparison | ✅ |
| Threshold optimization + 4 baselines | ✅ |
| 80%+ test coverage | ✅ (96.46%) |
| Logging + exception hierarchy | ✅ |
| No TODO / placeholder / debug code | ✅ |

---

## 15. Final Acceptance Checklist

| Check | Status |
|---|---|
| Imports (all modules) | ✅ |
| Configuration validation | ✅ |
| Dependency integrity (`pip check`) | ✅ |
| Python syntax (`py_compile`) | ✅ |
| Pipeline execution (all 9 stages) | ✅ |
| Deterministic outputs | ✅ |
| Model reproducibility | ✅ |
| Save/load consistency | ✅ |
| Dataset / feature / label / split integrity | ✅ |
| Model metrics match reports | ✅ |
| Threshold optimization | ✅ |
| SHAP outputs (21 plots) | ✅ |
| Risk map / dashboards / reports / figures | ✅ |
| Documentation + README + links + screenshots | ✅ |
| Folder structure + file naming | ✅ |
| CLI commands | ✅ |
| No broken references / TODO / debug / dead code | ✅ |
| No duplicated logic | ✅ |
| Logging + exceptions + type hints | ✅ |
| No secrets committed (`.env` gitignored, `.pkl` gitignored) | ✅ |
| License + ACLED attribution | ✅ |
| Automated validator `scripts/validate_project.py` | ✅ (PASS) |

---

## 16. Overall Repository Readiness

**✅ READY FOR SUBMISSION.**

The repository is a complete, self-contained, deterministic ML forecasting package: 13/13 milestones complete, 243/243 tests passing at 96.46% coverage, a verified end-to-end pipeline, executed notebooks, SHAP explainability, an interactive risk map + dashboard set, and documentation whose every metric is verified against the implementation. A fresh clone following `README.md` can reproduce the full pipeline.
