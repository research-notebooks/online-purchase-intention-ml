#!/usr/bin/env python3
"""
Feature Selection and Explainable Machine Learning for Online Purchase Intention Prediction
============================================================================================
Reproduction + extension code for the JEIT manuscript.

Dataset : UCI "Online Shoppers Purchasing Intention" (ID 468, DOI 10.24432/C5F88Q)
Run     : python jeit_purchase_intention.py --data online_shoppers_intention.csv
          python jeit_purchase_intention.py --data online_shoppers_intention.csv --quick   (fast smoke test)

Steps (all on by default, choose with --steps):
  main      Table 1 (CV + test F1), full test metrics for 8 conditions, Figure 1, PR curves
  shap      Table 2 + Figure 2 (80-session sample as in the paper), full-test-set SHAP,
            stability over repeated 80-session samples, permutation importance
  extended  repeated CV + paired tests, bootstrap CIs, k-sweep and selector comparison,
            PageValues ablation, extra baselines, duplicate-row sensitivity

Everything is written to ./results (CSV tables, PNG figures at 300 DPI, run_info.json).
Important: the main four-model comparison uses the predefined parameters in get_model()
(Appendix A); it does NOT run GridSearchCV. The extended analyses are supplementary.
Unspecified hyper-parameters are scikit-learn defaults (except max_iter for LinearSVC/LogReg,
raised only to guarantee convergence). Quick mode is only a smoke test, not for reporting.
"""
import argparse, json, platform, sys, time, warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("results")

NUMERIC = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
           "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
           "PageValues", "SpecialDay"]
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
BINARY = ["Weekend"]
MODELS = ["LinearSVC", "KNN", "ExtraTrees", "HistGradientBoosting"]   # the four paper models
EXTRA_MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]      # extra baselines (extended)


# ----------------------------------------------------------------------------- data
def load_data(path):
    p = Path(path) if path else None
    if p is None or not p.exists():
        try:                                   # Google Colab fallback
            from google.colab import files     # type: ignore
            up = files.upload()
            p = Path(next(n for n in up if n.lower().endswith(".csv")))
        except Exception:
            sys.exit("Dataset not found. Pass --data path/to/online_shoppers_intention.csv "
                     "(download: https://doi.org/10.24432/C5F88Q).")
    df = pd.read_csv(p)
    need = set(NUMERIC + CATEGORICAL + BINARY + ["Revenue"])
    if need - set(df.columns):
        sys.exit(f"Missing columns: {sorted(need - set(df.columns))}")
    X = df.drop(columns="Revenue").copy()
    X["Weekend"] = X["Weekend"].astype(int)
    y = df["Revenue"].astype(int)
    return df, X, y


# ----------------------------------------------------------------------------- models
def get_model(name, rep):
    """rep = '74' (all encoded columns) or '10' (SelectKBest k=10). Values = paper Appendix A."""
    r = str(rep)
    if name == "LinearSVC":
        return LinearSVC(C=0.1 if r == "74" else 1, class_weight="balanced",
                         random_state=SEED, max_iter=20000)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=11, weights="distance", p=2)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=200 if r == "74" else 400, max_features=0.5,
                                    min_samples_leaf=2, class_weight="balanced",
                                    random_state=SEED, n_jobs=1)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15,
                                              max_iter=100 if r == "74" else 200,
                                              l2_regularization=0.0 if r == "74" else 1.0,
                                              random_state=SEED)
    if name == "LogisticRegression":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced_subsample", random_state=SEED, n_jobs=1)
    if name == "XGBoost":
        from xgboost import XGBClassifier  # optional dependency
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=5.5, eval_metric="logloss",
                             random_state=SEED, n_jobs=1, verbosity=0)
    raise ValueError(name)


class RandomKSelector(BaseEstimator, TransformerMixin):
    """Random-k baseline: keeps k randomly chosen encoded columns (control for SelectKBest)."""
    def __init__(self, k=10, random_state=0):
        self.k, self.random_state = k, random_state
    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        self.idx_ = np.sort(rng.choice(X.shape[1], self.k, replace=False))
        return self
    def transform(self, X):
        return X[:, self.idx_]


def make_preprocessor(columns):
    num = [c for c in NUMERIC if c in columns]
    cat = [c for c in CATEGORICAL if c in columns]
    bin_ = [c for c in BINARY if c in columns]
    return ColumnTransformer([
        ("numeric", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("categorical", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                  ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("weekend", "passthrough", bin_)], sparse_threshold=0)   # always dense (SHAP / HGB need it)


def make_pipeline(clf, columns, k=None, selector="anova", seed=0):
    steps = [("pre", make_preprocessor(columns))]
    if k:
        if selector == "anova":
            steps.append(("select", SelectKBest(f_classif, k=k)))
        elif selector == "mutual_info":
            steps.append(("select", SelectKBest(partial(mutual_info_classif, random_state=SEED), k=k)))
        elif selector == "random":
            steps.append(("select", RandomKSelector(k=k, random_state=seed)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def scores(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def metrics(y, pred, sc):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                f1=f1_score(y, pred), average_precision=average_precision_score(y, sc),
                roc_auc=roc_auc_score(y, sc), TN=tn, FP=fp, FN=fn, TP=tp)


def f1_fast(y, p):
    tp = (y & p).sum(-1); fp = ((1 - y) & p).sum(-1); fn = (y & (1 - p)).sum(-1)
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def save(df, name, index=True):
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / name, index=index)
    print(f"  saved results/{name}")


def savefig(name):
    OUT.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(OUT / name, dpi=300); plt.close()
    print(f"  saved results/{name}")


# ----------------------------------------------------------------------------- MAIN
def run_main(ctx):
    Xtr, Xte, ytr, yte = ctx["Xtr"], ctx["Xte"], ctx["ytr"], ctx["yte"]
    cols = Xtr.columns
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, cv_rows, preds, scs, fitted = [], [], {}, {}, {}
    for name in MODELS:
        for rep in ("74", "10"):
            pipe = make_pipeline(get_model(name, rep), cols, k=None if rep == "74" else 10)
            cv = cross_validate(pipe, Xtr, ytr, cv=skf, scoring="f1", n_jobs=-1)["test_score"]
            pipe.fit(Xtr, ytr)
            pr, sc = pipe.predict(Xte), scores(pipe, Xte)
            m = metrics(yte, pr, sc)
            rows.append(dict(model=name, features=rep, cv_f1_mean=cv.mean(), cv_f1_sd=cv.std(ddof=1), **m))
            preds[(name, rep)], scs[(name, rep)], fitted[(name, rep)] = pr, sc, pipe
            print(f"  {name:22s} {rep:>2s}  CV F1 {cv.mean():.4f} ± {cv.std(ddof=1):.4f} | test F1 {m['f1']:.4f}")
    res = pd.DataFrame(rows)
    save(res.round(4), "table_all_metrics.csv", index=False)
    t1 = res.pivot(index="model", columns="features", values=["cv_f1_mean", "cv_f1_sd", "f1"])
    table1 = pd.DataFrame({
        "CV74": t1[("cv_f1_mean", "74")].map("{:.3f}".format) + " ± " + t1[("cv_f1_sd", "74")].map("{:.3f}".format),
        "CV10": t1[("cv_f1_mean", "10")].map("{:.3f}".format) + " ± " + t1[("cv_f1_sd", "10")].map("{:.3f}".format),
        "Test74": t1[("f1", "74")].round(3), "Test10": t1[("f1", "10")].round(3)}).loc[MODELS]
    save(table1, "table1_cv_and_test_f1.csv")
    # Figure 1
    fig, ax = plt.subplots(figsize=(7, 4)); w = 0.38; xs = np.arange(len(MODELS))
    for j, rep in enumerate(("74", "10")):
        v = [res[(res.model == m) & (res.features == rep)].f1.iloc[0] for m in MODELS]
        b = ax.bar(xs + (j - .5) * w, v, w, label=f"{rep} features")
        ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(MODELS, rotation=15); ax.set_ylabel("Held-out test F1 (purchase class)")
    ax.set_ylim(0, .8); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    savefig("figure1_test_f1.png")
    # PR curves
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for key, lab in [(("ExtraTrees", "74"), "Extra Trees (74)"), (("ExtraTrees", "10"), "Extra Trees (10)"),
                     (("HistGradientBoosting", "74"), "HistGB (74)"), (("HistGradientBoosting", "10"), "HistGB (10)")]:
        p, r, _ = precision_recall_curve(yte, scs[key])
        ax.plot(r, p, label=f"{lab}, AP={average_precision_score(yte, scs[key]):.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    savefig("figure3_pr_curves.png")
    ctx.update(preds=preds, scs=scs, fitted=fitted, main=res)


# ----------------------------------------------------------------------------- SHAP
def shap_values_pos(explainer, X):
    sv = explainer.shap_values(X, check_additivity=False)
    if isinstance(sv, list):          # older SHAP: list [class0, class1]
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:  # newer SHAP: (n, features, classes)
        sv = sv[:, :, 1]
    return np.asarray(sv)


def run_shap(ctx, n_sample=80, repeats=200):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP analysis was requested but the 'shap' package is not installed. "
            "Install it with: pip install shap, then rerun with --steps main,shap,extended."
        ) from exc
    Xtr, Xte, ytr, yte = ctx["Xtr"], ctx["Xte"], ctx["ytr"], ctx["yte"]
    cols = Xtr.columns
    # Full (74) and reduced (10) Extra Trees, fitted on the training set only
    pre = make_preprocessor(cols).fit(Xtr)
    names74 = pre.get_feature_names_out()
    names74 = np.array([n.split("__", 1)[1] for n in names74])
    A_tr, A_te = pre.transform(Xtr), pre.transform(Xte)
    sel = SelectKBest(f_classif, k=10).fit(A_tr, ytr)
    keep = sel.get_support(); names10 = names74[keep]
    print("  Selected features:", list(names10))
    save(pd.DataFrame({"feature": names74, "anova_F": sel.scores_, "selected": keep})
         .sort_values("anova_F", ascending=False), "anova_scores_all_74.csv", index=False)
    et74 = get_model("ExtraTrees", "74").fit(A_tr, ytr)
    et10 = get_model("ExtraTrees", "10").fit(A_tr[:, keep], ytr)
    print("  computing SHAP on the full test set (can take a few minutes)...")
    sv74 = shap_values_pos(shap.TreeExplainer(et74), A_te)[:, keep]        # restrict to the 10 shared features
    sv10 = shap_values_pos(shap.TreeExplainer(et10), A_te[:, keep])
    # (a) paper protocol: 80 reproducibly sampled test sessions
    pos_pd = Xte.reset_index(drop=True).sample(n_sample, random_state=SEED).index.values  # pandas sampler
    def summarize(idx, tag):
        m74, m10 = np.abs(sv74[idx]).mean(0), np.abs(sv10[idx]).mean(0)
        rho = stats.spearmanr(m74, m10)[0]
        t5 = len(set(names10[np.argsort(-m74)[:5]]) & set(names10[np.argsort(-m10)[:5]]))
        tab = pd.DataFrame({"feature": names10, "ET_74": m74, "ET_10": m10}).sort_values("ET_74", ascending=False)
        save(tab.round(4), f"table2_mean_abs_shap_{tag}.csv", index=False)
        print(f"  [{tag}] Spearman rho = {rho:.3f}; top-5 overlap = {t5}/5; top feature: {tab.feature.iloc[0]}")
        return tab, rho, t5
    tab80, rho80, t5_80 = summarize(pos_pd, "80_sessions")
    tabAll, rhoAll, t5All = summarize(np.arange(len(Xte)), "full_test_set")
    # Figure 2 (80-session sample) and full-test figure
    for tab, fn in [(tab80, "figure2_shap_80_sessions.png"), (tabAll, "figure2b_shap_full_test.png")]:
        fig, ax = plt.subplots(figsize=(7, 4.2)); t = tab.iloc[::-1]; y = np.arange(len(t)); h = .38
        ax.barh(y + h / 2, t.ET_74, h, label="Extra Trees (74 features)")
        ax.barh(y - h / 2, t.ET_10, h, label="Extra Trees (10 features)")
        ax.set_yticks(y); ax.set_yticklabels(t.feature); ax.set_xlabel("Mean |SHAP| (purchase class)")
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        savefig(fn)
    # (b) stability: repeated random 80-session samples
    rng = np.random.default_rng(SEED); rhos, t5s = [], []
    for _ in range(repeats):
        idx = rng.choice(len(Xte), n_sample, replace=False)
        m74, m10 = np.abs(sv74[idx]).mean(0), np.abs(sv10[idx]).mean(0)
        rhos.append(stats.spearmanr(m74, m10)[0])
        t5s.append(len(set(np.argsort(-m74)[:5]) & set(np.argsort(-m10)[:5])))
    stab = dict(repeats=repeats, rho_mean=float(np.mean(rhos)), rho_p2_5=float(np.percentile(rhos, 2.5)),
                rho_p97_5=float(np.percentile(rhos, 97.5)), top5_overlap_mean=float(np.mean(t5s)),
                top5_overlap_min=int(np.min(t5s)))
    print("  stability over repeated 80-session samples:", {k: round(v, 3) for k, v in stab.items()})
    save(pd.DataFrame([stab]).round(4), "shap_stability_80_session_resamples.csv", index=False)
    # (c) permutation importance of the 74-feature model (cross-check on SHAP ranking)
    pi = permutation_importance(et74, A_te, yte, scoring="average_precision", n_repeats=10,
                                random_state=SEED, n_jobs=-1)
    pit = pd.DataFrame({"feature": names74, "perm_importance_AP": pi.importances_mean,
                        "sd": pi.importances_std}).sort_values("perm_importance_AP", ascending=False)
    save(pit.round(4), "permutation_importance_ET74.csv", index=False)
    # (d) SHAP of ALL 74 features of the full model (not only the ten retained)
    sv_all = shap_values_pos(shap.TreeExplainer(et74), A_te)
    top = pd.DataFrame({"feature": names74, "mean_abs_shap": np.abs(sv_all).mean(0)}) \
            .sort_values("mean_abs_shap", ascending=False)
    save(top.round(4), "shap_all_74_features_ET74.csv", index=False)
    ctx["shap_summary"] = dict(rho_80=rho80, top5_80=t5_80, rho_full=rhoAll, top5_full=t5All, **stab)


# ----------------------------------------------------------------------------- EXTENDED
def nb_corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV differences."""
    d = np.asarray(d); k = len(d)
    var = (1 / k + n_test / n_train) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.t.cdf(abs(t), k - 1)); h = stats.t.ppf(.975, k - 1) * np.sqrt(var)
    return d.mean(), d.mean() - h, d.mean() + h, p


def run_extended(ctx, quick=False):
    Xtr, Xte, ytr, yte = ctx["Xtr"], ctx["Xte"], ctx["ytr"], ctx["yte"]
    cols = Xtr.columns
    R = 2 if quick else 5
    B = 300 if quick else 5000
    # --- 1. repeated stratified CV + paired corrected t-test (74 vs 10)
    print("[ext 1] repeated 5x%d CV, paired 74 vs 10" % R)
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=SEED)
    ntr, nte = int(len(Xtr) * .8), int(len(Xtr) * .2)
    rows = []
    for name in MODELS:
        f = {}
        for rep in ("74", "10"):
            pipe = make_pipeline(get_model(name, rep), cols, k=None if rep == "74" else 10)
            f[rep] = cross_validate(pipe, Xtr, ytr, cv=rcv, scoring="f1", n_jobs=-1)["test_score"]
        m, lo, hi, p = nb_corrected_ttest(f["10"] - f["74"], ntr, nte)
        rows.append(dict(model=name, f1_74=f["74"].mean(), f1_10=f["10"].mean(), diff_10_minus_74=m,
                         ci95_low=lo, ci95_high=hi, p_corrected_t=p))
        print(f"  {name:22s} diff(10-74) = {m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f}")
    save(pd.DataFrame(rows).round(4), "ext1_repeated_cv_paired_tests.csv", index=False)

    # --- 2. bootstrap CIs on the held-out test set
    print("[ext 2] bootstrap (%d resamples) on test set" % B)
    P, y = ctx["preds"], yte.values.astype(int)
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(y), (B, len(y)))
    Y = y[idx]; rows = []; boots = {}
    for key, pr in P.items():
        boots[key] = f1_fast(Y, pr.astype(int)[idx])
        lo, hi = np.percentile(boots[key], [2.5, 97.5])
        rows.append(dict(model=key[0], features=key[1], test_f1=f1_fast(y, pr.astype(int)), ci95_low=lo, ci95_high=hi))
    save(pd.DataFrame(rows).round(4), "ext2_bootstrap_test_f1_ci.csv", index=False)
    rows = []
    for name in MODELS:
        d = boots[(name, "10")] - boots[(name, "74")]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(model=name, diff_10_minus_74=d.mean(), ci95_low=lo, ci95_high=hi,
                         prop_bootstrap_diff_gt0=(d > 0).mean()))
    save(pd.DataFrame(rows).round(4), "ext2_bootstrap_paired_difference.csv", index=False)

    # --- 3. k-sweep and selector comparison (5-fold CV F1 on training set)
    print("[ext 3] k-sweep / selectors")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ks = [5, 10, 20] if quick else [5, 10, 15, 20, 30, 40, 50, 74]
    rows = []
    for name in MODELS:
        for k in ks:
            pipe = make_pipeline(get_model(name, "74"), cols, k=None if k == 74 else k)
            s = cross_validate(pipe, Xtr, ytr, cv=skf, scoring="f1", n_jobs=-1)["test_score"]
            rows.append(dict(model=name, k=k, cv_f1=s.mean(), cv_sd=s.std(ddof=1)))
    sweep = pd.DataFrame(rows); save(sweep.round(4), "ext3_k_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in MODELS:
        t = sweep[sweep.model == name]; ax.errorbar(t.k, t.cv_f1, t.cv_sd, marker="o", capsize=2, label=name)
    ax.set_xlabel("Number of selected encoded features (k)"); ax.set_ylabel("5-fold CV F1")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    savefig("figure4_k_sweep.png")
    rows = []
    for name in ["ExtraTrees", "HistGradientBoosting"]:
        for sel in ("anova", "mutual_info"):
            pipe = make_pipeline(get_model(name, "10"), cols, k=10, selector=sel)
            s = cross_validate(pipe, Xtr, ytr, cv=skf, scoring="f1", n_jobs=-1)["test_score"]
            rows.append(dict(model=name, selector=sel, cv_f1=s.mean(), cv_sd=s.std(ddof=1)))
        rs = []
        for seed in range(3 if quick else 10):
            pipe = make_pipeline(get_model(name, "10"), cols, k=10, selector="random", seed=seed)
            rs.append(cross_validate(pipe, Xtr, ytr, cv=skf, scoring="f1", n_jobs=-1)["test_score"].mean())
        rows.append(dict(model=name, selector="random_10 (mean of draws)", cv_f1=np.mean(rs), cv_sd=np.std(rs, ddof=1)))
    save(pd.DataFrame(rows).round(4), "ext3_selector_comparison_k10.csv", index=False)

    # --- 4. PageValues / session-level metrics ablation (leakage-aware, early-session scenario)
    print("[ext 4] ablation of PageValues (and exit/bounce metrics)")
    scen = {"all features": [], "no PageValues": ["PageValues"],
            "no PageValues/BounceRates/ExitRates": ["PageValues", "BounceRates", "ExitRates"]}
    rows = []
    for sname, drop in scen.items():
        Xa_tr, Xa_te = Xtr.drop(columns=drop), Xte.drop(columns=drop)
        for name in ["ExtraTrees", "HistGradientBoosting", "LogisticRegression"]:
            for rep in ("74", "10"):
                pipe = make_pipeline(get_model(name, "74" if name == "LogisticRegression" else rep),
                                     Xa_tr.columns, k=None if rep == "74" else 10)
                cv = cross_validate(pipe, Xa_tr, ytr, cv=skf, scoring="f1", n_jobs=-1)["test_score"]
                pipe.fit(Xa_tr, ytr); m = metrics(yte, pipe.predict(Xa_te), scores(pipe, Xa_te))
                rows.append(dict(scenario=sname, model=name, features="all" if rep == "74" else "top10",
                                 cv_f1=cv.mean(), cv_sd=cv.std(ddof=1), test_f1=m["f1"],
                                 test_ap=m["average_precision"], test_auc=m["roc_auc"]))
    ab = pd.DataFrame(rows); save(ab.round(4), "ext4_pagevalues_ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4)); t = ab[(ab.features == "all")]
    sc_names = list(scen); w = .26
    for j, name in enumerate(["ExtraTrees", "HistGradientBoosting", "LogisticRegression"]):
        v = [t[(t.scenario == s) & (t.model == name)].test_ap.iloc[0] for s in sc_names]
        ax.bar(np.arange(3) + (j - 1) * w, v, w, label=name)
    ax.set_xticks(range(3)); ax.set_xticklabels([s.replace("/", "/\n") for s in sc_names], fontsize=8)
    ax.set_ylabel("Test average precision"); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    savefig("figure5_ablation_ap.png")

    # --- 5. extra baselines (default hyper-parameters, NOT tuned)
    print("[ext 5] extra baselines")
    rows = []
    for name in EXTRA_MODELS:
        try:
            get_model(name, "74")
        except ImportError:
            print(f"  {name} skipped (not installed)"); continue
        for rep in ("74", "10"):
            pipe = make_pipeline(get_model(name, rep), cols, k=None if rep == "74" else 10)
            cv = cross_validate(pipe, Xtr, ytr, cv=skf, scoring="f1", n_jobs=-1)["test_score"]
            pipe.fit(Xtr, ytr); m = metrics(yte, pipe.predict(Xte), scores(pipe, Xte))
            rows.append(dict(model=name, features=rep, cv_f1=cv.mean(), cv_sd=cv.std(ddof=1), **m))
    save(pd.DataFrame(rows).round(4), "ext5_extra_baselines.csv", index=False)

    # --- 6. duplicate-row sensitivity
    print("[ext 6] duplicates removed (sensitivity)")
    df = ctx["df"].drop_duplicates(); X2 = df.drop(columns="Revenue").copy(); X2["Weekend"] = X2["Weekend"].astype(int)
    y2 = df["Revenue"].astype(int)
    a, b, c, d = train_test_split(X2, y2, test_size=.2, random_state=SEED, stratify=y2)
    rows = []
    for name in ["ExtraTrees", "HistGradientBoosting"]:
        for rep in ("74", "10"):
            pipe = make_pipeline(get_model(name, rep), cols, k=None if rep == "74" else 10).fit(a, c)
            m = metrics(d, pipe.predict(b), scores(pipe, b))
            rows.append(dict(model=name, features=rep, n_train=len(a), n_test=len(b), test_f1=m["f1"],
                             test_ap=m["average_precision"], test_auc=m["roc_auc"]))
    save(pd.DataFrame(rows).round(4), "ext6_duplicates_removed.csv", index=False)


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="online_shoppers_intention.csv")
    ap.add_argument("--steps", default="main,shap,extended")
    ap.add_argument("--quick", action="store_true", help="fewer repeats/resamples for a fast test")
    a = ap.parse_args()
    steps = set(a.steps.split(",")); t0 = time.time()
    OUT.mkdir(exist_ok=True)
    df, X, y = load_data(a.data)
    if a.quick:
        print("WARNING: QUICK MODE is a smoke test only. Do not report quick-mode inferential results in the manuscript.")
    print(f"Data: {df.shape}, missing={int(df.isna().sum().sum())}, duplicate rows={int(df.duplicated().sum())}")
    print("Revenue counts:", y.value_counts().to_dict())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    print(f"Train {Xtr.shape}, Test {Xte.shape} (test purchases = {int(yte.sum())})")
    ctx = dict(df=df, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
    if steps & {"main", "extended"}:
        print("\n=== MAIN ==="); run_main(ctx)
    if "shap" in steps:
        print("\n=== SHAP ==="); run_shap(ctx)
    if "extended" in steps:
        print("\n=== EXTENDED ==="); run_extended(ctx, quick=a.quick)
    info = dict(python=sys.version, platform=platform.platform(), sklearn=sklearn.__version__,
                numpy=np.__version__, pandas=pd.__version__, scipy=stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__,
                matplotlib=matplotlib.__version__, seed=SEED, quick=a.quick,
                steps=sorted(steps), minutes=round((time.time() - t0) / 60, 1),
                model_parameters={name: get_model(name, "74").get_params(deep=False) for name in MODELS})
    try:
        import shap; info["shap"] = shap.__version__
    except ImportError:
        info["shap"] = "not installed"
    try:
        import xgboost; info["xgboost"] = xgboost.__version__
    except ImportError:
        info["xgboost"] = "not installed"
    data_path = Path(a.data)
    if data_path.exists():
        import hashlib
        info["dataset_file"] = data_path.name
        info["dataset_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nDone in {info['minutes']} min. Outputs in ./results")


if __name__ == "__main__":
    main()
