# SHAP Explainability Summary — Winning Model

- Generated: 2026-08-08T00:13:21
- Model: escalation_best.pkl · operating threshold: 0.20
- SHAP: TreeExplainer · positive class: escalation within 14 days

## Top 20 features by mean |SHAP|

| rank | feature | mean |SHAP| | share | interpretation |
|---|---|---|---|---|
| 1 | fatalities_w30d | 0.5386 | 18.88% | Total fatalities in the trailing 30-day window — recent lethality. |
| 2 | velocity_events_w30d | 0.3251 | 11.40% | Event-count velocity: current 30-day count minus the preceding 30-day count. |
| 3 | events_w30d | 0.2755 | 9.66% | Total events in the trailing 30-day window — recent activity volume. |
| 4 | events_w7d | 0.2036 | 7.14% | Total events in the trailing 7-day window — recent activity volume. |
| 5 | admin1_code | 0.1927 | 6.76% | Deterministic numeric admin-1 / geo-unit identifier (unit-level baselines). |
| 6 | velocity_fatalities_w30d | 0.1840 | 6.45% | Fatality velocity: current 30-day count minus the preceding 30-day count. |
| 7 | events_w14d | 0.1599 | 5.61% | Total events in the trailing 14-day window — recent activity volume. |
| 8 | month | 0.1424 | 4.99% | Calendar month of the prediction date (seasonality). |
| 9 | spillover_w14d | 0.1405 | 4.93% | Events in the trailing 14-day window across the K nearest same-country units — spatial spillover. |
| 10 | fatalities_w14d | 0.1014 | 3.55% | Total fatalities in the trailing 14-day window — recent lethality. |
| 11 | fat_std_w30d | 0.0965 | 3.38% | Std-dev of daily fatalities over 30 days — volatility/spikiness. |
| 12 | velocity_events_w14d | 0.0847 | 2.97% | Event-count velocity: current 14-day count minus the preceding 14-day count. |
| 13 | fatalities_w7d | 0.0664 | 2.33% | Total fatalities in the trailing 7-day window — recent lethality. |
| 14 | velocity_events_w7d | 0.0644 | 2.26% | Event-count velocity: current 7-day count minus the preceding 7-day count. |
| 15 | velocity_fatalities_w14d | 0.0556 | 1.95% | Fatality velocity: current 14-day count minus the preceding 14-day count. |
| 16 | fat_mean_w14d | 0.0531 | 1.86% | Mean daily fatalities over 14 days — average intensity. |
| 17 | velocity_fatalities_w7d | 0.0530 | 1.86% | Fatality velocity: current 7-day count minus the preceding 7-day count. |
| 18 | country_code | 0.0505 | 1.77% | Deterministic numeric country identifier (captures cross-country baselines). |
| 19 | fat_mean_w30d | 0.0345 | 1.21% | Mean daily fatalities over 30 days — average intensity. |
| 20 | fat_std_w14d | 0.0268 | 0.94% | Std-dev of daily fatalities over 14 days — volatility/spikiness. |

## Most influential risk drivers

The strongest drivers are ranked above; observations below are 
computed directly from the ranking. Dependence plots live under 
``reports/shap/``.

## Model behaviour observations

- The single strongest driver is **fatalities_w30d** (mean |SHAP| 0.5386, 18.9% of total).
- Window emphasis: **30-day** windows carry the most volume/fatality signal (mean |SHAP| sums: 30d=0.945, 14d=0.482, 7d=0.270).
- Event velocity (14d) contributes 0.085 in mean |SHAP| — secondary to absolute volume.
- Fatality volatility (std) contributes 0.027 — a minor factor here.
- Spillover ranks #9 (mean |SHAP| 0.1405) — spatial contagion matters (FR-13).
- Identity codes (admin1/geo-unit/country) contribute 0.243 and calendar features 0.142 — unit-level baselines shape the estimate.
- The operating threshold (max-F1, below 0.5) reflects the majority-positive label; SHAP is computed on the held-out test window, so these are out-of-sample explanations.

## Local explanations (representative predictions)

### correctly predicted POSITIVE cases

- **Mandalay** (Myanmar, Mandalay) on 2026-01-10: predicted 1.000, true label 1 · waterfall `waterfall_pos_001.png`
  - Top drivers: fatalities_w30d: +3.149, velocity_fatalities_w30d: +0.790, events_w30d: +0.737
- **Magway** (Myanmar, Magway) on 2026-02-14: predicted 1.000, true label 1 · waterfall `waterfall_pos_002.png`
  - Top drivers: fatalities_w30d: +3.231, velocity_fatalities_w30d: +0.740, events_w7d: +0.715
- **Magway** (Myanmar, Magway) on 2026-01-24: predicted 1.000, true label 1 · waterfall `waterfall_pos_003.png`
  - Top drivers: fatalities_w30d: +3.173, events_w7d: +1.224, velocity_fatalities_w30d: +0.835

### correctly predicted NEGATIVE cases

- **Northern Bahr el Ghazal** (South Sudan, Northern Bahr el Ghazal) on 2025-04-26: predicted 0.021, true label 0 · waterfall `waterfall_neg_004.png`
  - Top drivers: events_w30d: -1.381, admin1_code: -0.603, events_w14d: -0.490
- **Ladakh** (India, Ladakh) on 2026-06-20: predicted 0.026, true label 0 · waterfall `waterfall_neg_005.png`
  - Top drivers: events_w30d: -1.401, events_w7d: -0.517, events_w14d: -0.449
- **Assam** (India, Assam) on 2026-03-07: predicted 0.026, true label 0 · waterfall `waterfall_neg_006.png`
  - Top drivers: velocity_events_w14d: -0.971, velocity_events_w7d: -0.935, velocity_events_w30d: -0.757

### difficult / borderline predictions

- **Yangon** (Myanmar, Yangon) on 2026-07-11: predicted 0.200, true label 0 · waterfall `waterfall_border_007.png`
  - Top drivers: velocity_events_w30d: -0.554, admin1_code: -0.451, fatalities_w30d: -0.259
- **Northern** (Sudan, Northern) on 2026-03-14: predicted 0.200, true label 0 · waterfall `waterfall_border_008.png`
  - Top drivers: events_w30d: -0.575, admin1_code: -0.500, events_w14d: -0.275
- **Gedaref** (Sudan, Gedaref) on 2025-09-20: predicted 0.200, true label 0 · waterfall `waterfall_border_009.png`
  - Top drivers: events_w30d: -0.594, spillover_w14d: -0.387, velocity_fatalities_w30d: +0.283
