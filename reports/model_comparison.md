# Model Comparison — LightGBM vs XGBoost

- Generated: 2026-08-07T23:31:04
- Seed: 42 · Imbalance: scale_pos_weight
- Split: train 30790 rows (2016-12-31 → 2023-09-02) · val 6522 rows (2023-09-09 → 2025-02-01)

## Winner: **xgboost**

validation F1 0.8423 vs lightgbm 0.8400 (higher validation F1 decided)

- Operating threshold: **0.25** (max_f1)

**Why this threshold:** the sweep over 0.10–0.90 (step 0.05) maximized validation F1 instead of assuming 0.5. The label is majority-positive, so the point of maximum F1 sits below 0.5 (the model under-weights the majority class via scale_pos_weight < 1); the best-F1 threshold is the operating point with the best precision/recall trade-off on the held-out validation window.

## Metrics at threshold 0.5 (validation)

| metric | lightgbm | xgboost |
|---|---|---|
| precision | 0.8433 | 0.8555 |
| recall | 0.7480 | 0.7172 |
| f1 | 0.7928 | 0.7803 |
| auc_pr | 0.9004 | 0.9031 |
| roc_auc | 0.8112 | 0.8175 |
| brier | 0.1743 | 0.1755 |
| log_loss | 0.5145 | 0.5171 |
| confusion (tn/fp/fn/tp) | [[1491, 614], [1113, 3304]] | [[1570, 535], [1249, 3168]] |

## Threshold analysis (best points, validation)

### lightgbm

| criterion | threshold | precision | recall | f1 |
|---|---|---|---|---|
| best F1 | 0.20 | 0.7530 | 0.9497 | 0.8400 |
| best precision | 0.90 | 0.9540 | 0.3102 | 0.4681 |
| best recall | 0.10 | 0.7121 | 0.9866 | 0.8272 |

### xgboost

| criterion | threshold | precision | recall | f1 |
|---|---|---|---|---|
| best F1 | 0.25 | 0.7711 | 0.9280 | 0.8423 |
| best precision | 0.90 | 0.9742 | 0.2649 | 0.4165 |
| best recall | 0.10 | 0.7114 | 0.9857 | 0.8264 |


## Baselines (validation)

| baseline | precision | recall | f1 | auc_pr | brier |
|---|---|---|---|---|---|
| majority | 0.6772 | 1.0000 | 0.8076 | 0.6772 | 0.3228 |
| always_positive | 0.6772 | 1.0000 | 0.8076 | 0.6772 | 0.3228 |
| persistence | 0.7645 | 0.8983 | 0.8261 | 0.7557 | 0.2562 |
| event_count_heuristic | 0.7965 | 0.8125 | 0.8044 | 0.7741 | 0.2676 |

## Artifacts

| Role | Path |
|---|---|
| lightgbm | `C:\Users\sarth\OneDrive\Desktop\army\models\escalation_lgbm.pkl` |
| xgboost | `C:\Users\sarth\OneDrive\Desktop\army\models\escalation_xgb.pkl` |
| best | `C:\Users\sarth\OneDrive\Desktop\army\models\escalation_best.pkl` |
