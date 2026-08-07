# Live 14-Day Forecast Summary

- Generated: 2026-08-08T00:13:43
- Model: escalation_best.pkl · operating threshold: 0.20
- As-of date: 2026-07-25 (prediction window: next 14 days)
- Scope: 83 geo units across 4 countries

## Highest-risk regions (top 10)

| rank | geo unit | country | probability | class | category |
|---|---|---|---|---|---|
| 1 | Magway | Myanmar | 0.998 | 1 | Critical |
| 2 | Sagaing | Myanmar | 0.992 | 1 | Critical |
| 3 | North Kordofan | Sudan | 0.991 | 1 | Critical |
| 4 | Rakhine | Myanmar | 0.962 | 1 | Critical |
| 5 | Mandalay | Myanmar | 0.961 | 1 | Critical |
| 6 | Jonglei | South Sudan | 0.938 | 1 | Critical |
| 7 | Jammu and Kashmir | India | 0.933 | 1 | Critical |
| 8 | Shan-South | Myanmar | 0.933 | 1 | Critical |
| 9 | Karnataka | India | 0.925 | 1 | Critical |
| 10 | Blue Nile | Sudan | 0.890 | 1 | Critical |

## Safest regions (bottom 10)

| rank | geo unit | country | probability | class | category |
|---|---|---|---|---|---|
| 1 | Sikkim | India | 0.023 | 0 | Low |
| 2 | Lakshadweep | India | 0.030 | 0 | Low |
| 3 | Dadra and Nagar Haveli and Daman and Diu | India | 0.037 | 0 | Low |
| 4 | Abyei | Sudan | 0.043 | 0 | Low |
| 5 | Red Sea | Sudan | 0.098 | 0 | Low |
| 6 | Sennar | Sudan | 0.106 | 0 | Low |
| 7 | Gedaref | Sudan | 0.114 | 0 | Low |
| 8 | Ladakh | India | 0.143 | 0 | Low |
| 9 | East Darfur | Sudan | 0.147 | 0 | Low |
| 10 | River Nile | Sudan | 0.175 | 0 | Low |

## Average risk by country

| country | avg risk | positive rate | mean events (7d) | mean fatalities (7d) |
|---|---|---|---|---|
| Myanmar | 0.746 | 1.000 | 12.67 | 6.89 |
| South Sudan | 0.532 | 1.000 | 1.90 | 3.10 |
| India | 0.532 | 0.889 | 24.33 | 0.11 |
| Sudan | 0.426 | 0.632 | 3.68 | 4.53 |

## Top SHAP drivers

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | fatalities_w30d | 0.3412 |
| 2 | events_w30d | 0.3281 |
| 3 | velocity_events_w30d | 0.2987 |
| 4 | events_w7d | 0.2524 |
| 5 | admin1_code | 0.1969 |
| 6 | events_w14d | 0.1811 |
| 7 | velocity_fatalities_w30d | 0.1651 |
| 8 | spillover_w14d | 0.1358 |
| 9 | month | 0.0854 |
| 10 | velocity_events_w14d | 0.0819 |

## Interpretation

Each geo unit is scored at its most recent available feature date; the predicted probability is the model's assessment that escalation occurs within the next 14 days. The risk category comes from `config.RISK_LEVEL_BOUNDARIES`; the predicted class uses the operating threshold from `model_comparison.json`.

## Important observations

- Highest-risk geo unit over the next 14 days: **Magway** (Myanmar) with probability 0.998.
- 69.9% of geo units are High or Critical risk ({'Critical': 41, 'High': 17, 'Medium': 14, 'Low': 11}).
- Highest average risk country: **Myanmar** (avg 0.746).
- Strongest overall risk driver: **fatalities_w30d** (mean |SHAP| 0.3412).
- This is a live forecast on each unit's most recent state — the model was never retrained for forecasting.
