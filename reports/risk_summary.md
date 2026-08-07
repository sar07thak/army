# Risk Summary — 14-Day Conflict Escalation Forecast

- Generated: 2026-08-08T00:13:37
- Model: escalation_best.pkl · operating threshold: 0.20
- Scope: 82 geo units · snapshot dates: 2026-05-23 .. 2026-07-11

## Highest-risk regions (top 10)

| rank | geo unit | country | probability | class | category |
|---|---|---|---|---|---|
| 1 | Sagaing | Myanmar | 0.999 | 1 | Critical |
| 2 | Magway | Myanmar | 0.994 | 1 | Critical |
| 3 | North Kordofan | Sudan | 0.993 | 1 | Critical |
| 4 | Rakhine | Myanmar | 0.990 | 1 | Critical |
| 5 | Mandalay | Myanmar | 0.984 | 1 | Critical |
| 6 | Tanintharyi | Myanmar | 0.926 | 1 | Critical |
| 7 | Assam | India | 0.887 | 1 | Critical |
| 8 | Jonglei | South Sudan | 0.834 | 1 | Critical |
| 9 | West Bengal | India | 0.831 | 1 | Critical |
| 10 | Kayin | Myanmar | 0.828 | 1 | Critical |

## Safest regions (bottom 10)

| rank | geo unit | country | probability | class | category |
|---|---|---|---|---|---|
| 1 | Lakshadweep | India | 0.030 | 0 | Low |
| 2 | Abyei | Sudan | 0.043 | 0 | Low |
| 3 | Red Sea | Sudan | 0.098 | 0 | Low |
| 4 | Sennar | Sudan | 0.106 | 0 | Low |
| 5 | Andaman and Nicobar Islands | India | 0.128 | 0 | Low |
| 6 | Ladakh | India | 0.139 | 0 | Low |
| 7 | Sikkim | India | 0.139 | 0 | Low |
| 8 | Nay Pyi Taw | Myanmar | 0.167 | 0 | Low |
| 9 | River Nile | Sudan | 0.175 | 0 | Low |
| 10 | Puducherry | India | 0.194 | 0 | Low |

## Average risk by country

| country | avg risk | positive rate | mean events (7d) | mean fatalities (7d) |
|---|---|---|---|---|
| Myanmar | 0.706 | 0.889 | 14.94 | 8.28 |
| South Sudan | 0.556 | 1.000 | 3.00 | 17.80 |
| India | 0.545 | 0.857 | 14.20 | 0.46 |
| Sudan | 0.447 | 0.789 | 3.89 | 9.79 |

## Top SHAP drivers

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | fatalities_w30d | 0.5526 |
| 2 | velocity_events_w30d | 0.3224 |
| 3 | events_w30d | 0.2779 |
| 4 | events_w7d | 0.2076 |
| 5 | admin1_code | 0.1945 |
| 6 | velocity_fatalities_w30d | 0.1792 |
| 7 | events_w14d | 0.1596 |
| 8 | month | 0.1427 |
| 9 | spillover_w14d | 0.1397 |
| 10 | fatalities_w14d | 0.1000 |

## Interpretation

The risk categories are derived from `config.RISK_LEVEL_BOUNDARIES`; the predicted class uses the operating threshold from `model_comparison.json`. Each geo unit is shown at its latest date in the test window (the model's most recent assessment). Full SHAP interpretations live in `reports/shap_summary.md`.

## Important observations

- Highest-risk geo unit: **Sagaing** (Myanmar) with probability 0.999.
- Safest geo unit: **Lakshadweep** (India) with probability 0.030.
- 73.2% of geo units are High or Critical risk ({'Critical': 39, 'High': 21, 'Low': 11, 'Medium': 11}).
- Highest average risk country: **Myanmar** (avg 0.706).
- Strongest overall risk driver: **fatalities_w30d** (mean |SHAP| 0.5526).
- Observations are computed on the out-of-sample test window; the model was never retrained for visualization.
