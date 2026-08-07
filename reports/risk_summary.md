# Risk Summary — 14-Day Conflict Escalation Forecast

- Generated: 2026-08-07T21:20:16
- Model: escalation_best.pkl · operating threshold: 0.25
- Scope: 122 geo units · snapshot dates: 2026-05-02 .. 2026-07-11

## Highest-risk regions (top 10)

| rank | geo unit | country | probability | class | category |
|---|---|---|---|---|---|
| 1 | Khyber Pakhtunkhwa | Pakistan | 0.999 | 1 | Critical |
| 2 | Sagaing | Myanmar | 0.998 | 1 | Critical |
| 3 | Balochistan | Pakistan | 0.993 | 1 | Critical |
| 4 | North Kordofan | Sudan | 0.991 | 1 | Critical |
| 5 | Magway | Myanmar | 0.991 | 1 | Critical |
| 6 | Sindh | Pakistan | 0.986 | 1 | Critical |
| 7 | Rakhine | Myanmar | 0.970 | 1 | Critical |
| 8 | Mandalay | Myanmar | 0.943 | 1 | Critical |
| 9 | Punjab | India | 0.938 | 1 | Critical |
| 10 | Manipur | India | 0.935 | 1 | Critical |

## Safest regions (bottom 10)

| rank | geo unit | country | probability | class | category |
|---|---|---|---|---|---|
| 1 | Zabul | Afghanistan | 0.026 | 0 | Low |
| 2 | Kunduz | Afghanistan | 0.028 | 0 | Low |
| 3 | Lakshadweep | India | 0.032 | 0 | Low |
| 4 | Abyei | Sudan | 0.037 | 0 | Low |
| 5 | Samangan | Afghanistan | 0.038 | 0 | Low |
| 6 | Wardak | Afghanistan | 0.042 | 0 | Low |
| 7 | Nuristan | Afghanistan | 0.050 | 0 | Low |
| 8 | Baghlan | Afghanistan | 0.052 | 0 | Low |
| 9 | Ghor | Afghanistan | 0.055 | 0 | Low |
| 10 | Helmand | Afghanistan | 0.061 | 0 | Low |

## Average risk by country

| country | avg risk | positive rate | mean events (7d) | mean fatalities (7d) |
|---|---|---|---|---|
| Pakistan | 0.749 | 0.833 | 44.67 | 36.17 |
| Myanmar | 0.688 | 0.889 | 14.94 | 8.28 |
| India | 0.527 | 0.857 | 15.26 | 0.46 |
| South Sudan | 0.505 | 0.700 | 3.00 | 17.80 |
| Sudan | 0.445 | 0.684 | 3.89 | 9.79 |
| Afghanistan | 0.176 | 0.176 | 0.68 | 0.38 |

## Top SHAP drivers

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | events_w30d | 0.4772 |
| 2 | fatalities_w30d | 0.4209 |
| 3 | velocity_events_w30d | 0.3096 |
| 4 | events_w7d | 0.1789 |
| 5 | spillover_w14d | 0.1547 |
| 6 | admin1_code | 0.1515 |
| 7 | events_w14d | 0.1481 |
| 8 | velocity_fatalities_w30d | 0.1454 |
| 9 | fatalities_w14d | 0.1284 |
| 10 | month | 0.1144 |

## Interpretation

The risk categories are derived from `config.RISK_LEVEL_BOUNDARIES`; the predicted class uses the operating threshold from `model_comparison.json`. Each geo unit is shown at its latest date in the test window (the model's most recent assessment). Full SHAP interpretations live in `reports/shap_summary.md`.

## Important observations

- Highest-risk geo unit: **Khyber Pakhtunkhwa** (Pakistan) with probability 0.999.
- Safest geo unit: **Zabul** (Afghanistan) with probability 0.026.
- 53.3% of geo units are High or Critical risk ({'Critical': 42, 'Low': 36, 'High': 23, 'Medium': 21}).
- Highest average risk country: **Pakistan** (avg 0.749).
- Strongest overall risk driver: **events_w30d** (mean |SHAP| 0.4772).
- Observations are computed on the out-of-sample test window; the model was never retrained for visualization.
