# JEIT Revision Reproducibility Package

**Manuscript:** *Feature Selection and Explainable Machine Learning for Online Purchase Intention Prediction*

**Dataset:** UCI Online Shoppers Purchasing Intention (Dataset 468)  
**DOI:** https://doi.org/10.24432/C5F88Q

This repository accompanies the revised manuscript and its supplementary materials. It provides the source dataset, analysis scripts, environment requirements, and the generated tables and figures used in the revised study.

## Repository Contents

- `jeit_purchase_intention.py` — Core matched-model comparison and paired TreeSHAP analysis.
- `reviewer_supplementary.py` — Supplementary analyses addressing reviewer requests, including repeated stratified cross-validation, nested cross-validation, feature-selector comparisons, feature-selection sensitivity, paired bootstrap analysis, PageValues ablation, duplicate-aware evaluation, and baseline comparison.
- `JEIT_Colab_Run_All.ipynb` — Jupyter Notebook for running the analysis in Google Colab.
- `online_shoppers_intention.csv` — Source dataset used in the reported analyses.
- `requirements.txt` — Python package requirements for the revision analysis.
- `results/` — SHAP results, bootstrap summaries, feature-importance outputs, and figures.
- `reviewer_results/` — Matched-model metrics and supplementary analysis outputs.
- `DATASET_ATTRIBUTION.txt` — Dataset source and attribution information.
- `RUN_INSTRUCTIONS_AR.txt` — Instructions for running the analysis in Arabic.
- `PACKAGE_MANIFEST.json` — Package contents and metadata.
- `REVIEWER_REQUEST_COVERAGE.md` — Summary of how the reviewer requests are addressed.

## Environment

The recorded analysis environment was:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- scikit-learn 1.8.0
- matplotlib 3.10.8
- SHAP 0.50.0
- threadpoolctl 3.6.0

Additional environment and dataset information is available in:

- `results/run_info.json`
- `reviewer_results/metadata.json`

## Installation

Install the required Python packages from the repository directory:

```bash
python -m pip install -r requirements.txt
```

## Run the Core Analysis

Run the matched-model comparison and SHAP analysis using:

```bash
python jeit_purchase_intention.py --data online_shoppers_intention.csv --steps main,shap
```

The `main` step evaluates four classifiers using matched settings across the 74-feature and 10-feature representations.

The `shap` step uses matched 200-tree Extra Trees models and the same reproducible sample of 1,200 held-out sessions (`random_state=42`), followed by 1,000 paired row-bootstrap resamples.

## Run the Reviewer-Requested Supplementary Analyses

Run the supplementary analysis script using:

```bash
python reviewer_supplementary.py --data online_shoppers_intention.csv --out reviewer_results
```

This script generates the supplementary analyses, including:

- Matched-model performance metrics.
- Repeated stratified 5-fold cross-validation with three repeats.
- Nested 3×3 cross-validation.
- ANOVA, Mutual Information, and random-selector comparisons.
- Feature-selection sensitivity analysis for k = 5, 10, 15, and 20.
- Paired test-set bootstrap intervals for F1 and Average Precision (AP).
- PageValues and session-rate ablation analyses.
- Duplicate auditing and group-aware evaluation.
- Prior-probability baseline comparison.

## Important Interpretation Notes

1. The original feature-specific hyperparameter presets were selected after examining purchase-class F1 under cross-validation on the training partition. The original candidate grid and fold-level selection record were not retained, and the original submission script did not use automated `GridSearchCV` for that preliminary selection. This limitation is disclosed in the manuscript. The revised controlled comparison uses matched settings, while the separate nested search applies identical candidate grids to both feature representations.

2. The held-out test set was not used for cross-validation or nested model selection. Test-set bootstrap intervals are conditional on the fitted models and should not be interpreted as repeated-dataset uncertainty estimates.

3. The timing of `PageValues`, `BounceRates`, and `ExitRates` within an active session is not established by the dataset documentation. Their ablation results do not prove data leakage. Therefore, the study frames the task as retrospective session classification rather than validated early-session prediction.

4. The 1,200-session SHAP sample is paired between the two Extra Trees models. The grouped categorical-field summary is a descriptive sum of encoded-column mean absolute SHAP values; it is not a newly calculated group Shapley value.

5. Results generated using `--quick` mode should not be reported as manuscript results. This mode is intended only for a smoke test of the optional extended-analysis pathway.

6. All reported manuscript results should be taken from the full-run CSV files included in `results/` and `reviewer_results/`. Outputs from different versions or random seeds should not be mixed.

## Reproducibility

The scripts, dataset, environment requirements, notebook, and generated result files are provided to support transparency and reproducibility of the revised analyses.

For dataset attribution and source information, refer to `DATASET_ATTRIBUTION.txt`.

For Arabic execution instructions, refer to `RUN_INSTRUCTIONS_AR.txt`.
