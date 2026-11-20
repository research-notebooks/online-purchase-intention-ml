# JEIT Revision Reproducibility Package

**Manuscript:** *Feature Selection and Explainable Machine Learning for Online Purchase Intention Prediction*  
**Dataset:** UCI Online Shoppers Purchasing Intention (dataset 468; DOI: https://doi.org/10.24432/C5F88Q)

This package accompanies the revised manuscript and its supplementary tables. It contains the source dataset, analysis scripts, environment pins, and the generated tables/figures used in the revision.

## Contents

- `jeit_purchase_intention.py` — core matched-model comparison and paired TreeSHAP analysis.
- `reviewer_supplementary.py` — reviewer-requested analyses: repeated stratified CV, nested CV, selector/k sensitivity, paired bootstrap, PageValues ablation, duplicate-aware evaluation, and baseline.
- `online_shoppers_intention.csv` — the source dataset used for the reported analyses.
- `requirements.txt` — package versions used for the revision run.
- `results/` — SHAP tables, bootstrap summaries, importance outputs, and figures.
- `reviewer_results/` — matched metrics and supplementary analysis outputs.
- `DATASET_ATTRIBUTION.txt` — source and attribution information.
- `RUN_INSTRUCTIONS_AR.txt` — Arabic run instructions.

## Environment

The recorded environment was Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8, SHAP 0.50.0, and threadpoolctl 3.6.0. See `results/run_info.json` and `reviewer_results/metadata.json` for the environment and dataset hash.

## Run the core analysis

From this directory:

```bash
python -m pip install -r requirements.txt
python jeit_purchase_intention.py --data online_shoppers_intention.csv --steps main,shap
```

The `main` step evaluates the four classifiers with matched settings across 74 and 10 encoded columns. The `shap` step uses matched 200-tree Extra Trees models and the same reproducible sample of 1,200 held-out sessions (`random_state=42`), followed by 1,000 paired row-bootstrap resamples.

## Run the reviewer-requested supplementary analyses

```bash
python reviewer_supplementary.py --data online_shoppers_intention.csv --out reviewer_results
```

This command regenerates the matched metrics, repeated 5-fold CV (three repeats), nested 3×3 CV, ANOVA/Mutual Information/random-selector comparisons, k=5/10/15/20 sensitivity, paired test-set bootstrap intervals for F1 and AP, PageValues/session-rate ablations, duplicate audit and group-aware tests, and the prior-probability baseline.

## Important interpretation notes

1. The original feature-specific presets were selected after examining purchase-class F1 under cross-validation on the training partition. The original candidate grid and fold-level trace were not retained, and the original submission script did not use automated `GridSearchCV` for that preliminary step. This is disclosed in the manuscript. The revised controlled comparison uses matched settings, and the separate nested search uses identical candidate grids for both representations.
2. The held-out test set is not used for CV/nested model selection. Test-set bootstrap intervals are conditional on the fitted models and are not repeated-dataset uncertainty estimates.
3. The timing of `PageValues`, `BounceRates`, and `ExitRates` within an active session is not established by the dataset documentation. Their ablation results do not prove leakage. The paper therefore frames the task as retrospective session classification rather than validated early-session prediction.
4. The 1,200-session SHAP sample is paired between the two Extra Trees models. The grouped categorical-field summary is a descriptive sum of encoded-column mean absolute SHAP values; it is not a newly calculated group Shapley value.
5. Do not report results from `--quick` mode. It is only for a smoke test of the optional extended-analysis pathway.
6. All displayed manuscript results should be taken from the full-run CSV files included in `results/` and `reviewer_results/`; do not mix outputs from different versions or seeds.

## GitHub synchronization

The ZIP contains the authoritative revision snapshot. Before resubmission, replace the older repository files with this package's current scripts, README, run instructions, requirements, and manifest; add the dataset attribution, dataset, supplementary script, notebook, and result folders. Keep the existing anonymous MIT LICENSE and .gitignore files unchanged. In particular, do not leave the older README/script text that refers to an 80-session SHAP sample. This archive does not itself push commits to GitHub.
