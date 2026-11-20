# Reviewer-request coverage map

This map links the reviewer comments to the revisions and supporting outputs.

## Reviewer #53

| Comment | Revision | Evidence |
|---|---|---|
| Original feature-specific hyperparameters and selection process unclear | Manuscript Methods §3.4 and Appendix A disclose that the original presets were chosen after examining purchase-class F1 under CV on the training partition; the original candidate grid/fold trace was not retained. A matched-parameter comparison and a separate nested search were added. | Manuscript §§3.4, Appendix A; Supplementary Tables S2–S3; `reviewer_results/matched_hyperparameters.csv`, `nested_cv_tuning.csv` |
| SHAP sample of 80 too small | Paired SHAP increased to 1,200 held-out sessions with 1,000 paired bootstrap resamples. The abstract, results, discussion and figure caption report rho 0.9636, CI [0.9027, 0.9879], top-five overlap 4/5. | Manuscript §3.6 and §4.2; Supplementary Table S8; `results/shap_paired_bootstrap_summary.csv`; Figure 2 |
| Contribution/novelty is incremental | Contribution reframed as empirical replication/methodological best practice; two additional references on XAI feature selection and ranking stability included. | Manuscript Introduction/Related Work; References [11]–[12] |
| Justify k=10 and examine other k values | k=10 framed as a compact comparison, not an optimum; k=5, 10, 15 and 20 evaluated. | Manuscript §3.3/§4.3; Supplementary Table S4; `reviewer_results/k_sensitivity_5_10_15_20.csv` |
| Quantify duplicate rows and overlap across partitions | Predictor-equality audit, conflicting-label check, split-crossing group counts, and group-aware sensitivity analyses added. | Manuscript §§3.2/§4.3; Supplementary Table S6; `reviewer_results/duplicate_audit.json`, `group_aware_evaluation.csv` |
| CV SD and SHAP uncertainty needed | Main Table 1 includes CV mean ± SD; Supplementary Table S8 includes SHAP intervals. | Manuscript Table 1; Supplementary Table S8 |
| Improve figures | Figures regenerated at 300 dpi with labelled axes and sample-specific captions. | Manuscript Figures 1–2; `reviewer_results/figure1_matched_f1.png`, `results/figure2_shap_1200_sessions.png` |
| Add SHAP stability literature | References [11] and [12] added; local explanation agreement is not claimed. | Manuscript Related Work and References |
| Repository reproducibility | Public URL resolves; revision archive supplies exact scripts, data, requirements and outputs. The public repository must be synchronized with this archive before submission. | Manuscript Code and Data Availability; `README.md`, `PACKAGE_MANIFEST.json` |

## Reviewer #54

| Comment | Revision | Evidence |
|---|---|---|
| Matched parameters and nested tuning | Same settings applied to both feature representations; nested 3×3 CV uses identical two-candidate budgets. | Manuscript §3.4; Supplementary Tables S2–S3 |
| PageValues prediction-time leakage | Timing uncertainty disclosed; PageValues and PageValues/BounceRates/ExitRates ablations added; task reframed as retrospective session classification. | Manuscript §§3.6, 4.4, 5.1; Supplementary Table S5 |
| Duplicate dependence | Group audit and group-isolated test results for all models; grouped CV for tree ensembles. | Supplementary Table S6 |
| Stronger validation and feature selection | Repeated 5-fold CV × 3 repeats, nested CV, k sensitivity, MI and random-feature controls, fold-wise feature frequencies, and DummyClassifier baseline. | Supplementary Tables S3–S4, S9–S11 |
| SHAP reproducibility/grouped interpretation | SHAP 0.50.0, TreeExplainer settings, class index, shared-column ranking definition, full-74 top-five, bootstrap intervals, descriptive encoded-column aggregation; no local agreement claim. | Manuscript §3.6/§4.2; Supplementary Tables S8 and S12 |
| All metrics and decision rules for 8 configurations | Accuracy, balanced accuracy, precision, recall, F1, AP, ROC-AUC and confusion-matrix counts for all eight configurations; LinearSVC threshold at decision_function=0. | Supplementary Table S1 and its note; Manuscript §3.5 |
| Paired intervals | Paired test-set bootstrap intervals for F1 and AP; conditional uncertainty is described. | Manuscript §4.1; Supplementary Table S7 |
| Runtime/model-size claims | No operational efficiency claim is made because runtime, model size and feature acquisition costs were not measured. | Manuscript §5.1 |
| Generalization limits | Single historical retailer benchmark and need for external validation are explicit. | Manuscript §5.1 |
