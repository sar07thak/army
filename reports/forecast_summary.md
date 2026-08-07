# Live 14-Day Forecast Summary

- Generated: 2026-08-07T21:55:27
- Model: escalation_best.pkl · operating threshold: 0.25
- As-of date: 2026-07-25 (prediction window: next 14 days)
- Scope: 124 geo units across 6 countries

## Highest-risk regions (top 10)

| rank | geo unit | country | probability | class | category |
|---|---|---|---|---|---|
| 1 | Balochistan | Pakistan | 0.998 | 1 | Critical |
| 2 | Magway | Myanmar | 0.997 | 1 | Critical |
| 3 | Khyber Pakhtunkhwa | Pakistan | 0.996 | 1 | Critical |
| 4 | Sagaing | Myanmar | 0.990 | 1 | Critical |
| 5 | North Kordofan | Sudan | 0.971 | 1 | Critical |
| 6 | Mandalay | Myanmar | 0.957 | 1 | Critical |
| 7 | Sindh | Pakistan | 0.955 | 1 | Critical |
| 8 | Rakhine | Myanmar | 0.940 | 1 | Critical |
| 9 | Punjab | India | 0.937 | 1 | Critical |
| 10 | Kabul | Afghanistan | 0.911 | 1 | Critical |

## Safest regions (bottom 10)

| rank | geo unit | country | probability | class | category |
|---|---|---|---|---|---|
| 1 | Faryab | Afghanistan | 0.025 | 0 | Low |
| 2 | Sar-e Pol | Afghanistan | 0.025 | 0 | Low |
| 3 | Zabul | Afghanistan | 0.026 | 0 | Low |
| 4 | Lakshadweep | India | 0.032 | 0 | Low |
| 5 | Sikkim | India | 0.034 | 0 | Low |
| 6 | Laghman | Afghanistan | 0.035 | 0 | Low |
| 7 | Abyei | Sudan | 0.037 | 0 | Low |
| 8 | Samangan | Afghanistan | 0.038 | 0 | Low |
| 9 | Dadra and Nagar Haveli and Daman and Diu | India | 0.040 | 0 | Low |
| 10 | Wardak | Afghanistan | 0.042 | 0 | Low |

## Average risk by country

| country | avg risk | positive rate | mean events (7d) | mean fatalities (7d) |
|---|---|---|---|---|
| Pakistan | 0.730 | 0.857 | 40.00 | 24.29 |
| Myanmar | 0.717 | 1.000 | 12.67 | 6.89 |
| India | 0.515 | 0.833 | 25.00 | 0.14 |
| South Sudan | 0.481 | 0.800 | 1.90 | 3.10 |
| Sudan | 0.401 | 0.632 | 3.68 | 4.53 |
| Afghanistan | 0.177 | 0.235 | 1.00 | 0.24 |

## Top SHAP drivers

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | events_w30d | 0.5771 |
| 2 | fatalities_w30d | 0.3234 |
| 3 | velocity_events_w30d | 0.2679 |
| 4 | events_w7d | 0.2263 |
| 5 | spillover_w14d | 0.1846 |
| 6 | events_w14d | 0.1668 |
| 7 | admin1_code | 0.1534 |
| 8 | velocity_fatalities_w30d | 0.1420 |
| 9 | fatalities_w14d | 0.1043 |
| 10 | fat_std_w14d | 0.0946 |

## Interpretation

Each geo unit is scored at its most recent available feature date; the predicted probability is the model's assessment that escalation occurs within the next 14 days. The risk category comes from `config.RISK_LEVEL_BOUNDARIES`; the predicted class uses the operating threshold from `model_comparison.json`.

## Important observations

- Highest-risk geo unit over the next 14 days: **Balochistan** (Pakistan) with probability 0.998.
- 51.6% of geo units are High or Critical risk ({'Critical': 39, 'Low': 35, 'Medium': 25, 'High': 25}).
- Highest average risk country: **Pakistan** (avg 0.730).
- Strongest overall risk driver: **events_w30d** (mean |SHAP| 0.5771).
- This is a live forecast on each unit's most recent state — the model was never retrained for forecasting.
