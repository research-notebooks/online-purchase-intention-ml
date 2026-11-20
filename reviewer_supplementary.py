#!/usr/bin/env python3
"""Supplementary analyses requested by JEIT reviewers.
Uses the same source dataset, fixed seed (42), and the manuscript's preprocessing.
All preprocessing/feature selection remain inside each CV pipeline.
"""
import argparse, hashlib, json, platform, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    StratifiedGroupKFold, GridSearchCV)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
import joblib
import jeit_purchase_intention as core

SEED=42
OUT=Path('reviewer_results'); OUT.mkdir(exist_ok=True)

def metric_dict(y, pred, score):
    return core.metrics(np.asarray(y), np.asarray(pred), np.asarray(score))

def dump(df, name):
    df.to_csv(OUT/name, index=False)
    print('saved',name,flush=True)

def matched_params(Xtr,Xte,ytr,yte,cols):
    print('\n[MATCHED PARAMETERS]',flush=True); rows=[]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for model in core.MODELS:
        # Identical classifier settings for 74-column and ANOVA-10 conditions.
        estimator=core.get_model(model,'74')
        for rep in ['74','10']:
            pipe=core.make_pipeline(clone(estimator),cols,k=None if rep=='74' else 10)
            with threadpool_limits(limits=1):
                vals=[]
                for a,b in cv.split(Xtr,ytr):
                    pipe.fit(Xtr.iloc[a],ytr.iloc[a]); vals.append(f1_score(ytr.iloc[b],pipe.predict(Xtr.iloc[b])))
                pipe.fit(Xtr,ytr)
                pred=pipe.predict(Xte); sc=core.scores(pipe,Xte); met=metric_dict(yte,pred,sc)
            rows.append(dict(model=model,representation=rep,cv_f1_mean=np.mean(vals),cv_f1_sd=np.std(vals,ddof=1),**met))
            print(model,rep,round(np.mean(vals),4),round(met['f1'],4),flush=True)
    dump(pd.DataFrame(rows).round(5),'matched_hyperparameters.csv')


def repeated_stratified_cv(Xtr, ytr, cols, repeats=3):
    """Repeated 5-fold stratified evaluation on training data for robustness."""
    print('\n[REPEATED STRATIFIED CV]', flush=True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    splits = list(cv.split(Xtr, ytr))
    rows, paired = [], []
    for name in core.MODELS:
        fold_scores = {}
        for rep in ['74', '10']:
            pipe = core.make_pipeline(core.get_model(name, '74'), cols, k=None if rep == '74' else 10)
            vals = []
            with threadpool_limits(limits=1):
                for a, b in splits:
                    pipe.fit(Xtr.iloc[a], ytr.iloc[a])
                    vals.append(f1_score(ytr.iloc[b], pipe.predict(Xtr.iloc[b])))
            fold_scores[rep] = np.asarray(vals)
            rows.append(dict(model=name, representation=rep, repeats=repeats, folds_per_repeat=5,
                             n_fold_scores=len(vals), cv_f1_mean=fold_scores[rep].mean(),
                             cv_f1_sd=fold_scores[rep].std(ddof=1)))
            print(name, rep, 'mean', round(fold_scores[rep].mean(), 4), 'SD', round(fold_scores[rep].std(ddof=1), 4), flush=True)
        d = fold_scores['10'] - fold_scores['74']
        paired.append(dict(model=name, repeats=repeats, paired_fold_scores=len(d),
                           mean_difference_10_minus_74=d.mean(), sd_difference=d.std(ddof=1),
                           note='Descriptive paired fold differences; folds are dependent across repeats.'))
    dump(pd.DataFrame(rows).round(6), 'repeated_stratified_cv.csv')
    dump(pd.DataFrame(paired).round(6), 'repeated_stratified_cv_paired_differences.csv')

def nested_cv(Xtr,Xte,ytr,yte,cols):
    print('\n[NESTED CV]',flush=True)
    # Small, explicit and equally sized candidate grids for each representation.
    grids={
      'LinearSVC': {'clf__C':[0.1,1.0]},
      'KNN': {'clf__n_neighbors':[7,11]},
      'ExtraTrees': [{'clf__max_features':[0.5], 'clf__min_samples_leaf':[1]}, {'clf__max_features':[0.5], 'clf__min_samples_leaf':[2]}],
      'HistGradientBoosting': [{'clf__max_iter':[100], 'clf__l2_regularization':[0.0]}, {'clf__max_iter':[200], 'clf__l2_regularization':[1.0]}],
    }
    outer=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED+1)
    rows=[]
    for name in core.MODELS:
      for rep in ['74','10']:
        pipe=core.make_pipeline(core.get_model(name,'74'),cols,k=None if rep=='74' else 10)
        fold_scores=[]; fold_params=[]
        for fold,(a,b) in enumerate(outer.split(Xtr,ytr),1):
          gs=GridSearchCV(pipe,grids[name],scoring='f1',cv=inner,n_jobs=1,refit=True,error_score='raise')
          with threadpool_limits(limits=1): gs.fit(Xtr.iloc[a],ytr.iloc[a])
          p=gs.predict(Xtr.iloc[b]); s=f1_score(ytr.iloc[b],p)
          fold_scores.append(s); fold_params.append(json.dumps(gs.best_params_,sort_keys=True))
          print(name,rep,'outer',fold,'F1',round(s,4),'params',gs.best_params_,flush=True)
        # Final inner-CV search on the original training partition; test set used once for evaluation.
        final=GridSearchCV(pipe,grids[name],scoring='f1',cv=inner,n_jobs=1,refit=True,error_score='raise')
        with threadpool_limits(limits=1): final.fit(Xtr,ytr)
        pr=final.predict(Xte); sc=core.scores(final,Xte); met=metric_dict(yte,pr,sc)
        rows.append(dict(model=name,representation=rep,outer_cv_f1_mean=np.mean(fold_scores),
          outer_cv_f1_sd=np.std(fold_scores,ddof=1),outer_fold_f1=json.dumps([round(x,6) for x in fold_scores]),
          outer_fold_best_params=' || '.join(fold_params),final_train_cv_best_params=json.dumps(final.best_params_,sort_keys=True),
          final_inner_cv_best_f1=final.best_score_,test_f1=met['f1'],test_precision=met['precision'],
          test_recall=met['recall'],test_ap=met['average_precision'],test_roc_auc=met['roc_auc'],
          test_balanced_accuracy=met['balanced_accuracy'],TN=met['TN'],FP=met['FP'],FN=met['FN'],TP=met['TP']))
        print(name,rep,'nested mean',round(np.mean(fold_scores),4),'test',round(met['f1'],4),flush=True)
    dump(pd.DataFrame(rows).round(5),'nested_cv_tuning.csv')

def feature_freq(Xtr,ytr,cols):
    print('\n[FEATURE SELECTION FREQUENCY]',flush=True)
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED); counts={}; all_names=None
    for k,(a,b) in enumerate(cv.split(Xtr,ytr),1):
      pre=core.make_preprocessor(cols).fit(Xtr.iloc[a],ytr.iloc[a]); names=np.asarray(pre.get_feature_names_out()); Z=pre.transform(Xtr.iloc[a])
      sel=SelectKBest(f_classif,k=10).fit(Z,ytr.iloc[a])
      chosen=names[sel.get_support()]
      for f in chosen: counts[f]=counts.get(f,0)+1
      print('fold',k, 'selected',len(chosen),flush=True)
    out=pd.DataFrame({'feature':list(counts),'selected_folds':list(counts.values())}).sort_values(['selected_folds','feature'],ascending=[False,True])
    dump(out,'anova_feature_selection_frequency.csv')

def group_duplicate_analysis(X,y,df):
    print('\n[DUPLICATE / GROUP ANALYSIS]',flush=True)
    hashes=pd.util.hash_pandas_object(X,index=False).astype('uint64')
    gdf=pd.DataFrame({'group':hashes.values,'y':y.values})
    agg=gdf.groupby('group').agg(n=('y','size'),label_count=('y','nunique'),positive=('y','sum'))
    dup=agg[agg.n>1]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    part=pd.Series(np.where(X.index.isin(Xte.index),'test','train'),index=X.index)
    cross=pd.DataFrame({'group':hashes.values,'part':part.values}).groupby('group').part.nunique()
    audit={'n_rows':len(df),'exact_duplicate_rows_beyond_first':int(df.duplicated().sum()),
      'predictor_duplicate_groups':int(len(dup)),'rows_in_predictor_duplicate_groups':int(dup.n.sum()),
      'additional_rows_in_predictor_duplicate_groups':int((dup.n-1).sum()),
      'predictor_duplicate_groups_with_conflicting_labels':int((dup.label_count>1).sum()),
      'predictor_duplicate_rows_with_conflicting_labels':int(dup.loc[dup.label_count>1,'n'].sum()) if (dup.label_count>1).any() else 0,
      'predictor_duplicate_groups_crossing_original_split':int((cross>1).sum()),
      'rows_in_predictor_groups_crossing_original_split':int(gdf.groupby('group').size()[cross>1].sum())}
    (OUT/'duplicate_audit.json').write_text(json.dumps(audit,indent=2))
    print(audit,flush=True)
    # Stratified group split: exact predictor-value groups are never split across partitions.
    sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    tr_idx,te_idx=next(sg.split(X,y,groups=hashes))
    Xgtr,Xgte=X.iloc[tr_idx],X.iloc[te_idx]; ygtr,ygte=y.iloc[tr_idx],y.iloc[te_idx]; gtr=hashes.iloc[tr_idx]
    assert len(set(gtr).intersection(set(hashes.iloc[te_idx])))==0
    groupcv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1)
    rows=[]
    for name in core.MODELS:
      for rep in ['74','10']:
        pipe=core.make_pipeline(core.get_model(name,'74'),Xgtr.columns,k=None if rep=='74' else 10)
        fs=[]
        # Grouped CV is focused on the two tree ensembles; all four classifiers
        # receive a group-isolated held-out test evaluation.
        if name in ['ExtraTrees','HistGradientBoosting']:
          for a,b in groupcv.split(Xgtr,ygtr,groups=gtr):
            pipe.fit(Xgtr.iloc[a],ygtr.iloc[a]);fs.append(f1_score(ygtr.iloc[b],pipe.predict(Xgtr.iloc[b])))
        pipe.fit(Xgtr,ygtr);pr=pipe.predict(Xgte);sc=core.scores(pipe,Xgte);met=metric_dict(ygte,pr,sc)
        rows.append(dict(model=name,representation=rep,train_n=len(tr_idx),test_n=len(te_idx),test_purchase_prevalence=ygte.mean(),
          grouped_cv_f1_mean=np.mean(fs) if fs else np.nan,grouped_cv_f1_sd=np.std(fs,ddof=1) if len(fs)>1 else np.nan,**met))
        cv_text=round(np.mean(fs),4) if fs else 'not calculated'
        print(name,rep,'grouped CV',cv_text,'group test',round(met['f1'],4),flush=True)
    dump(pd.DataFrame(rows).round(5),'group_aware_evaluation.csv')


def selector_comparison(Xtr,ytr,cols):
    print('\n[ANOVA / MUTUAL INFORMATION / RANDOM-10 COMPARISON]',flush=True)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED); rows=[]
    for name in core.MODELS:
      for selector in ['anova','mutual_info','random']:
        pipe=core.make_pipeline(core.get_model(name,'74'),cols,k=10,selector=selector,seed=SEED)
        fold=[]
        with threadpool_limits(limits=1):
          for a,b in cv.split(Xtr,ytr):
            pipe.fit(Xtr.iloc[a],ytr.iloc[a]); fold.append(f1_score(ytr.iloc[b],pipe.predict(Xtr.iloc[b])))
        rows.append(dict(model=name,selector={'anova':'ANOVA top-10','mutual_info':'Mutual information top-10','random':'Random 10 columns (seed 42)'}[selector],
                         cv_f1_mean=np.mean(fold),cv_f1_sd=np.std(fold,ddof=1)))
        print(name,selector,round(np.mean(fold),4),flush=True)
    dump(pd.DataFrame(rows).round(5),'selector_comparison_all_models.csv')


def paired_test_bootstrap(Xtr,Xte,ytr,yte,cols,B=1000):
    print('\n[PAIRED TEST BOOTSTRAP: F1 AND AP]',flush=True)
    rng=np.random.default_rng(SEED); y=np.asarray(yte,dtype=int); n=len(y)
    predictions={}; scores={}
    for name in core.MODELS:
      for rep in ['74','10']:
        pipe=core.make_pipeline(core.get_model(name,'74'),cols,k=None if rep=='74' else 10)
        with threadpool_limits(limits=1): pipe.fit(Xtr,ytr)
        predictions[(name,rep)]=pipe.predict(Xte); scores[(name,rep)]=core.scores(pipe,Xte)
    rows=[]
    for name in core.MODELS:
      d_f1=[]; d_ap=[]
      for _ in range(B):
        idx=rng.integers(0,n,n); yy=y[idx]
        f10=f1_score(yy,predictions[(name,'10')][idx],zero_division=0)
        f74=f1_score(yy,predictions[(name,'74')][idx],zero_division=0)
        ap10=average_precision_score(yy,scores[(name,'10')][idx])
        ap74=average_precision_score(yy,scores[(name,'74')][idx])
        d_f1.append(f10-f74); d_ap.append(ap10-ap74)
      rows.append(dict(model=name,metric='F1',difference_10_minus_74=float(np.mean(d_f1)),ci95_low=float(np.percentile(d_f1,2.5)),ci95_high=float(np.percentile(d_f1,97.5)),probability_positive=float(np.mean(np.asarray(d_f1)>0)),bootstrap_resamples=B))
      rows.append(dict(model=name,metric='Average Precision',difference_10_minus_74=float(np.mean(d_ap)),ci95_low=float(np.percentile(d_ap,2.5)),ci95_high=float(np.percentile(d_ap,97.5)),probability_positive=float(np.mean(np.asarray(d_ap)>0)),bootstrap_resamples=B))
      print(name,'F1 diff CI',np.round(np.percentile(d_f1,[2.5,97.5]),4),'AP diff CI',np.round(np.percentile(d_ap,[2.5,97.5]),4),flush=True)
    dump(pd.DataFrame(rows).round(5),'paired_bootstrap_test_differences.csv')

def k_sensitivity(Xtr,ytr,cols):
    print('\n[K SENSITIVITY]',flush=True); cv=StratifiedKFold(5,shuffle=True,random_state=SEED); rows=[]
    for name in core.MODELS:
      for k in [5,10,15,20]:
        pipe=core.make_pipeline(core.get_model(name,'74'),cols,k=k)
        vals=[]
        with threadpool_limits(limits=1):
          for a,b in cv.split(Xtr,ytr):
            pipe.fit(Xtr.iloc[a],ytr.iloc[a]); vals.append(f1_score(ytr.iloc[b],pipe.predict(Xtr.iloc[b])))
        rows.append(dict(model=name,k=k,cv_f1_mean=np.mean(vals),cv_f1_sd=np.std(vals,ddof=1)))
        print(name,'k',k,round(np.mean(vals),4),flush=True)
    dump(pd.DataFrame(rows).round(5),'k_sensitivity_5_10_15_20.csv')

def pagevalues_ablation(Xtr,Xte,ytr,yte):
    print('\n[PAGEVALUES ABLATION]',flush=True); cv=StratifiedKFold(5,shuffle=True,random_state=SEED); rows=[]
    scenarios={'All predictors':[],'Without PageValues':['PageValues'],'Without PageValues, BounceRates and ExitRates':['PageValues','BounceRates','ExitRates']}
    for scen,drop in scenarios.items():
      A,B=Xtr.drop(columns=drop),Xte.drop(columns=drop)
      for name in ['ExtraTrees','HistGradientBoosting']:
        for rep in ['74','10']:
          pipe=core.make_pipeline(core.get_model(name,'74'),A.columns,k=None if rep=='74' else 10)
          vals=[]
          with threadpool_limits(limits=1):
            for a,b in cv.split(A,ytr):
              pipe.fit(A.iloc[a],ytr.iloc[a]); vals.append(f1_score(ytr.iloc[b],pipe.predict(A.iloc[b])))
            pipe.fit(A,ytr); pr=pipe.predict(B); sc=core.scores(pipe,B); met=metric_dict(yte,pr,sc)
          rows.append(dict(scenario=scen,model=name,representation=rep,cv_f1_mean=np.mean(vals),cv_f1_sd=np.std(vals,ddof=1),test_f1=met['f1'],test_ap=met['average_precision'],test_roc_auc=met['roc_auc'],test_precision=met['precision'],test_recall=met['recall']))
          print(scen,name,rep,'test F1',round(met['f1'],4),'AP',round(met['average_precision'],4),flush=True)
    dump(pd.DataFrame(rows).round(5),'pagevalues_and_session_rate_ablation.csv')

def dummy_baseline(Xtr,Xte,ytr,yte):
    d=DummyClassifier(strategy='prior').fit(np.zeros((len(ytr),1)),ytr)
    pr=d.predict(np.zeros((len(yte),1))); sc=d.predict_proba(np.zeros((len(yte),1)))[:,1]
    met=metric_dict(yte,pr,sc); dump(pd.DataFrame([met]),'dummy_baseline.csv')

def main():
    global OUT
    parser=argparse.ArgumentParser(description='Run supplementary JEIT reviewer analyses.')
    parser.add_argument('--data', default='online_shoppers_intention.csv', help='Path to the UCI CSV dataset')
    parser.add_argument('--out', default='reviewer_results', help='Output directory')
    args=parser.parse_args()
    OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=True)
    df,X,y=core.load_data(args.data)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    cols=Xtr.columns
    matched_params(Xtr,Xte,ytr,yte,cols)
    repeated_stratified_cv(Xtr,ytr,cols,repeats=3)
    feature_freq(Xtr,ytr,cols)
    selector_comparison(Xtr,ytr,cols)
    k_sensitivity(Xtr,ytr,cols)
    paired_test_bootstrap(Xtr,Xte,ytr,yte,cols,B=1000)
    pagevalues_ablation(Xtr,Xte,ytr,yte)
    group_duplicate_analysis(X,y,df)
    dummy_baseline(Xtr,Xte,ytr,yte)
    nested_cv(Xtr,Xte,ytr,yte,cols)
    dataset_hash=hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    from importlib.metadata import version, PackageNotFoundError
    def package_version(name):
      try: return version(name)
      except PackageNotFoundError: return 'not-installed'
    metadata={'seed':SEED,'dataset_file':Path(args.data).name,'dataset_sha256':dataset_hash,
      'python':sys.version,'platform':platform.platform(),
      'packages':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':stats.__version__ if hasattr(stats,'__version__') else package_version('scipy'),
        'scikit_learn':core.sklearn.__version__,'matplotlib':package_version('matplotlib'),
        'shap':package_version('shap'),'threadpoolctl':package_version('threadpoolctl')},
      'nested_outer_folds':3,'nested_inner_folds':3,
      'inner_scoring':'purchase-class F1','candidate_grids':{'LinearSVC':2,'KNN':2,'ExtraTrees':2,'HistGradientBoosting':2},
      'preprocessing':'all fitted within pipelines and inner training folds','feature_selection':'SelectKBest ANOVA k=10 inside each pipeline',
      'test_set':'original held-out split used only for final evaluation, not nested model selection',
      'execution_note':'Supplementary metrics and sensitivity analyses are generated by this script; SHAP outputs are generated by jeit_purchase_intention.py using the documented matched Extra Trees models.'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
