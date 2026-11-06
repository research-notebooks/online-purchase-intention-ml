# JEIT research reproducibility code

This package contains the analysis script for the manuscript:
**Feature Selection and Explainable Machine Learning for Online Purchase Intention Prediction**

## Dataset
Download the UCI Online Shoppers Purchasing Intention dataset (ID 468) from:
https://doi.org/10.24432/C5F88Q

Place `online_shoppers_intention.csv` in the same folder as the script. The dataset is not included in this ZIP.

## Install
```bash
python -m pip install -r requirements.txt
```

## Full run
```bash
python jeit_purchase_intention.py --data online_shoppers_intention.csv --steps main,shap,extended
```

## Core paper tables only
```bash
python jeit_purchase_intention.py --data online_shoppers_intention.csv --steps main,shap
```

## Quick smoke test
```bash
python jeit_purchase_intention.py --data online_shoppers_intention.csv --quick
```
Quick mode reduces repeated-CV repeats and bootstrap resamples. It is only for checking that the script runs; **do not report quick-mode inferential results in the manuscript**.

## Outputs
All outputs are written to `results/`, including:
- Main comparison tables for the four classifiers and both feature representations
- Test-set metrics, confusion matrices and precision-recall curves
- SHAP summaries on the 80-session sample and the full test set
- SHAP stability across repeated 80-session samples
- Permutation importance and full 74-feature SHAP importance
- Repeated cross-validation, corrected paired tests and bootstrap confidence intervals
- Feature-count sweep and ANOVA / mutual-information / random-feature comparisons
- PageValues and related-feature ablation
- Logistic Regression, Random Forest and optional XGBoost baselines
- Duplicate-row sensitivity analysis
- `run_info.json` with package versions, model parameters and dataset SHA-256

## Important methodological notes
1. The main comparison uses the predefined model parameters in `get_model()`; it does **not** execute GridSearchCV. The manuscript must not claim that GridSearchCV was performed unless a separate tuning experiment is run and reported.
2. SHAP is required for the `shap` step. Install it using `pip install shap`. The script now stops with a clear error if SHAP is requested but unavailable; it will not silently omit SHAP.
3. The default full run is computationally heavier than quick mode. Runtime depends on hardware and package versions.
4. The PageValues ablation is a sensitivity analysis. A performance drop after removing PageValues does not, by itself, prove data leakage. Discuss feature availability at prediction time.
5. All reported values must be taken from the output files produced by the final full run. Do not mix outputs from different package versions or quick runs.
6. The script uses a fixed stratified 80/20 split (`random_state=42`). The held-out test set is not used for model selection.
7. Review the manuscript's authorship, CRediT contributions, funding, conflicts of interest, AI-use disclosure, and journal anonymization requirements before submission.

## Reproducibility
`results/run_info.json` records the environment and dataset hash for each run. Keep it with the output tables and figures.
