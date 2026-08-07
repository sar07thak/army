# SHAP Explainability Summary — Winning Model

- Generated: 2026-08-07T23:31:14
- Model: escalation_best.pkl · operating threshold: 0.25
- SHAP: TreeExplainer · positive class: escalation within 14 days

## Top 20 features by mean |SHAP|

| rank | feature | mean |SHAP| | share | interpretation |
|---|---|---|---|---|
| 1 | events_w30d | 0.4778 | 16.67% | Total events in the trailing 30-day window — recent activity volume. |
| 2 | fatalities_w30d | 0.4140 | 14.44% | Total fatalities in the trailing 30-day window — recent lethality. |
| 3 | velocity_events_w30d | 0.3173 | 11.07% | Event-count velocity: current 30-day count minus the preceding 30-day count. |
| 4 | events_w7d | 0.1766 | 6.16% | Total events in the trailing 7-day window — recent activity volume. |
| 5 | spillover_w14d | 0.1561 | 5.45% | Events in the trailing 14-day window across the K nearest same-country units — spatial spillover. |
| 6 | admin1_code | 0.1540 | 5.37% | Deterministic numeric admin-1 / geo-unit identifier (unit-level baselines). |
| 7 | velocity_fatalities_w30d | 0.1484 | 5.18% | Fatality velocity: current 30-day count minus the preceding 30-day count. |
| 8 | events_w14d | 0.1481 | 5.17% | Total events in the trailing 14-day window — recent activity volume. |
| 9 | fatalities_w14d | 0.1267 | 4.42% | Total fatalities in the trailing 14-day window — recent lethality. |
| 10 | month | 0.1137 | 3.97% | Calendar month of the prediction date (seasonality). |
| 11 | fat_std_w14d | 0.1025 | 3.58% | Std-dev of daily fatalities over 14 days — volatility/spikiness. |
| 12 | fatalities_w7d | 0.0831 | 2.90% | Total fatalities in the trailing 7-day window — recent lethality. |
| 13 | fat_std_w30d | 0.0775 | 2.71% | Std-dev of daily fatalities over 30 days — volatility/spikiness. |
| 14 | velocity_events_w14d | 0.0773 | 2.70% | Event-count velocity: current 14-day count minus the preceding 14-day count. |
| 15 | country_code | 0.0745 | 2.60% | Deterministic numeric country identifier (captures cross-country baselines). |
| 16 | velocity_fatalities_w14d | 0.0481 | 1.68% | Fatality velocity: current 14-day count minus the preceding 14-day count. |
| 17 | fat_mean_w30d | 0.0460 | 1.61% | Mean daily fatalities over 30 days — average intensity. |
| 18 | velocity_events_w7d | 0.0451 | 1.57% | Event-count velocity: current 7-day count minus the preceding 7-day count. |
| 19 | velocity_fatalities_w7d | 0.0443 | 1.55% | Fatality velocity: current 7-day count minus the preceding 7-day count. |
| 20 | fat_mean_w14d | 0.0300 | 1.05% | Mean daily fatalities over 14 days — average intensity. |

## Most influential risk drivers

The strongest drivers are ranked above; observations below are 
computed directly from the ranking. Dependence plots live under 
``reports/shap/``.

## Model behaviour observations

- The single strongest driver is **events_w30d** (mean |SHAP| 0.4778, 16.7% of total).
- Window emphasis: **30-day** windows carry the most volume/fatality signal (mean |SHAP| sums: 30d=1.015, 14d=0.563, 7d=0.260).
- Event velocity (14d) contributes 0.077 in mean |SHAP| — secondary to absolute volume.
- Fatality volatility (std) contributes 0.103 — spiky/irregular violence is a visible risk signal.
- Spillover ranks #5 (mean |SHAP| 0.1561) — spatial contagion matters (FR-13).
- Identity codes (admin1/geo-unit/country) contribute 0.228 and calendar features 0.114 — unit-level baselines shape the estimate.
- The operating threshold (max-F1, below 0.5) reflects the majority-positive label; SHAP is computed on the held-out test window, so these are out-of-sample explanations.

## Local explanations (representative predictions)

### correctly predicted POSITIVE cases

- **Sagaing** (Myanmar, Sagaing) on 2025-03-29: predicted 1.000, true label 1 · waterfall `waterfall_pos_001.png`
  - Top drivers: fatalities_w30d: +1.463, events_w30d: +1.064, events_w7d: +0.885
- **Balochistan** (Pakistan, Balochistan) on 2026-03-07: predicted 1.000, true label 1 · waterfall `waterfall_pos_002.png`
  - Top drivers: fatalities_w30d: +1.457, events_w7d: +1.029, events_w30d: +1.015
- **Sagaing** (Myanmar, Sagaing) on 2026-02-14: predicted 1.000, true label 1 · waterfall `waterfall_pos_003.png`
  - Top drivers: fatalities_w30d: +1.309, events_w30d: +1.198, events_w14d: +0.722

### correctly predicted NEGATIVE cases

- **Samangan** (Afghanistan, Samangan) on 2026-05-30: predicted 0.017, true label 0 · waterfall `waterfall_neg_004.png`
  - Top drivers: events_w30d: -1.619, events_w7d: -0.432, spillover_w14d: -0.429
- **Nuristan** (Afghanistan, Nuristan) on 2025-04-05: predicted 0.019, true label 0 · waterfall `waterfall_neg_005.png`
  - Top drivers: events_w30d: -1.555, events_w7d: -0.421, events_w14d: -0.390
- **Laghman** (Afghanistan, Laghman) on 2026-06-13: predicted 0.020, true label 0 · waterfall `waterfall_neg_006.png`
  - Top drivers: events_w30d: -1.633, events_w7d: -0.443, events_w14d: -0.417

### difficult / borderline predictions

- **Gedaref** (Sudan, Gedaref) on 2025-08-09: predicted 0.250, true label 0 · waterfall `waterfall_border_007.png`
  - Top drivers: spillover_w14d: -0.437, month: -0.316, events_w30d: -0.309
- **Kunduz** (Afghanistan, Kunduz) on 2026-02-28: predicted 0.250, true label 0 · waterfall `waterfall_border_008.png`
  - Top drivers: events_w30d: -0.577, spillover_w14d: -0.257, velocity_events_w30d: -0.253
- **Andaman and Nicobar Islands** (India, Andaman and Nicobar Islands) on 2025-08-02: predicted 0.250, true label 1 · waterfall `waterfall_border_009.png`
  - Top drivers: admin1_code: -0.451, velocity_events_w30d: -0.418, fatalities_w30d: -0.222
