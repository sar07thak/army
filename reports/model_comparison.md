# Model Comparison — LightGBM vs XGBoost

- Generated: 2026-08-08T00:12:24
- Seed: 42 · Imbalance: scale_pos_weight
- Split: train 19801 rows (2017-01-07 → 2023-09-02) · val 5099 rows (2023-09-09 → 2025-02-01)

## Winner: **lightgbm**

effectively equal; simpler family 'lightgbm' chosen (MODEL_SIMPLICITY_ORDER)

- Operating threshold: **0.20** (max_f1)

**Why this threshold:** the sweep over 0.10–0.90 (step 0.05) maximized validation F1 instead of assuming 0.5. The label is majority-positive, so the point of maximum F1 sits below 0.5 (the model under-weights the majority class via scale_pos_weight < 1); the best-F1 threshold is the operating point with the best precision/recall trade-off on the held-out validation window.

## Metrics at threshold 0.5 (validation)

| metric | lightgbm | xgboost |
|---|---|---|
| precision | 0.8357 | 0.8357 |
| recall | 0.7938 | 0.7938 |
| f1 | 0.8142 | 0.8142 |
| auc_pr | 0.9038 | 0.9038 |
| roc_auc | 0.7593 | 0.7593 |
| brier | 0.1766 | 0.1766 |
| log_loss | 0.5192 | 0.5192 |
| confusion (tn/fp/fn/tp) | [[714, 592], [782, 3011]] | [[714, 592], [782, 3011]] |

## Threshold analysis (best points, validation)

### lightgbm

| criterion | threshold | precision | recall | f1 |
|---|---|---|---|---|
| best F1 | 0.20 | 0.7663 | 0.9776 | 0.8591 |
| best precision | 0.90 | 0.9643 | 0.2702 | 0.4222 |
| best recall | 0.10 | 0.7543 | 0.9955 | 0.8583 |

### xgboost

| criterion | threshold | precision | recall | f1 |
|---|---|---|---|---|
| best F1 | 0.20 | 0.7663 | 0.9776 | 0.8591 |
| best precision | 0.90 | 0.9643 | 0.2702 | 0.4222 |
| best recall | 0.10 | 0.7543 | 0.9955 | 0.8583 |


## Baselines (validation)

| baseline | precision | recall | f1 | auc_pr | brier |
|---|---|---|---|---|---|
| majority | 0.7439 | 1.0000 | 0.8531 | 0.7439 | 0.2561 |
| always_positive | 0.7439 | 1.0000 | 0.8531 | 0.7439 | 0.2561 |
| persistence | 0.7770 | 0.9233 | 0.8439 | 0.7745 | 0.2542 |
| event_count_heuristic | 0.7935 | 0.8437 | 0.8178 | 0.7857 | 0.2797 |

## Artifacts

| Role | Path |
|---|---|
| lightgbm | `C:\Users\LENOVO\.gemini\antigravity-ide\scratch\army\models\escalation_lgbm.pkl` |
| xgboost | `C:\Users\LENOVO\.gemini\antigravity-ide\scratch\army\models\escalation_xgb.pkl` |
| best | `C:\Users\LENOVO\.gemini\antigravity-ide\scratch\army\models\escalation_best.pkl` |
