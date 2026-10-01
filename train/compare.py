"""Honest 10-algorithm comparison + stacking ensemble.

Methodology:
- Exact duplicate feature profiles are dropped BEFORE any split (this file
  has 520 rows but only 251 unique profiles, with no conflicting labels).
  Without this, a random split puts identical copies on both sides and every
  metric is memorization, not generalization.
- Every model is a Pipeline(StandardScaler + classifier), so scaling is fit
  INSIDE each fold. No pre-split selection anywhere.
- Repeated stratified 5-fold x 4 repeats -> pooled out-of-fold probabilities
  (each unique profile scored 4x by models that never saw it).
- Metrics: accuracy, ROC-AUC, PR-AUC, F1, sensitivity, specificity, Brier.
- 95% bootstrap CIs (2000 resamples) over the SAME vectors the point
  estimates used: hard labels for accuracy, probabilities for AUC.
  An interval that misses its point estimate is a bug, and the script exits.
- The stacking ensemble uses StackingClassifier(cv=5) with a 100-tree forest,
  the same forest the shipped model uses.
Writes train/comparison.json (consumed by summary page at build time).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, StackingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, recall_score, brier_score_loss,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RNG = 7
N_SPLITS, N_REPEATS = 5, 4
N_BOOT = 2000
RF_TREES = 100


def load():
    df = pd.read_csv("diabetes-repo/data.csv")
    feats = [c for c in df.columns if c != "class"]
    n_raw = len(df)
    dup = df.duplicated(subset=feats, keep=False)
    if dup.any():
        clash = df.loc[dup].groupby(feats, dropna=False)["class"].nunique()
        if (clash > 1).any():
            raise SystemExit("duplicate profiles with conflicting labels — refusing to drop silently")
    df = df.drop_duplicates(subset=feats).reset_index(drop=True)
    X = df[feats].copy()
    X["Gender"] = (X["Gender"] == "Male").astype(int)
    for c in feats[2:]:
        X[c] = (X[c] == "Yes").astype(int)
    X["Age"] = X["Age"].astype(float)
    y = (df["class"] == "Positive").astype(int).to_numpy()
    load.n_raw = n_raw
    load.n_unique = len(df)
    return X[feats].to_numpy(float), y, feats


def models():
    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG, n_jobs=-1)
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "Gaussian NB": make_pipeline(StandardScaler(), GaussianNB()),
        "Decision Tree": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=10, random_state=RNG)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG, n_jobs=-1)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=21, weights="distance")),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=RNG)),
        "XGBoost": make_pipeline(StandardScaler(), XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                                                  subsample=0.9, colsample_bytree=0.9, random_state=RNG,
                                                                  n_jobs=-1, eval_metric="logloss")),
        "LightGBM": make_pipeline(StandardScaler(), LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.05,
                                                                   random_state=RNG, n_jobs=-1, verbose=-1)),
        "Hist Gradient Boost": make_pipeline(StandardScaler(), HistGradientBoostingClassifier(max_iter=200, random_state=RNG)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=1500, random_state=RNG)),
        "Stacking Ensemble": make_pipeline(StandardScaler(), StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=1.0, max_iter=2000)),
                ("nb", GaussianNB()),
                ("knn", KNeighborsClassifier(n_neighbors=21, weights="distance")),
                ("gb", GradientBoostingClassifier(random_state=RNG)),
                ("rf", rf),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000),
            cv=5, n_jobs=-1,
        )),
    }


def oof_probs(make, X, y):
    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RNG)
    acc = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for tr, te in rkf.split(X, y):
        clf = make()
        clf.fit(X[tr], y[tr])
        acc[te] += clf.predict_proba(X[te])[:, 1]
        cnt[te] += 1
    return acc / cnt


def boot_ci(y, pred, fn, seed):
    """Bootstrap 95% CI for fn(y_boot, pred_boot). pred must be the exact
    vector the point estimate used (hard labels for accuracy, probabilities
    for AUC) — never a different statistic."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(y[idx], pred[idx]))
        except ValueError:
            continue  # a resample with one class: AUC undefined, skip it
    if len(vals) < int(N_BOOT * 0.9):
        raise SystemExit(f"bootstrap undefined too often ({len(vals)}/{N_BOOT})")
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    X, y, feats = load()
    print(f"Dataset: {load.n_raw} rows -> {load.n_unique} unique profiles x {len(feats)} features | positives={y.sum()} ({y.mean()*100:.1f}%)")
    t0 = time.time()
    rows = []
    for name, proto in models().items():
        # fresh clone per fold to avoid any state carryover
        from sklearn.base import clone
        p = oof_probs(lambda p=proto: clone(p), X, y)
        hard = (p >= 0.5).astype(int)
        tn = int(((hard == 0) & (y == 0)).sum())
        fp = int(((hard == 1) & (y == 0)).sum())
        spec = tn / (tn + fp)
        acc = float(accuracy_score(y, hard))
        auc = float(roc_auc_score(y, p))
        acc_ci = boot_ci(y, hard, accuracy_score, seed=11)
        auc_ci = boot_ci(y, p, roc_auc_score, seed=12)
        if not (acc_ci[0] <= acc <= acc_ci[1]):
            raise SystemExit(f"{name}: accuracy CI {acc_ci} misses point {acc}")
        if not (auc_ci[0] <= auc <= auc_ci[1]):
            raise SystemExit(f"{name}: AUC CI {auc_ci} misses point {auc}")
        row = {
            "name": name,
            "deployed": name == "Stacking Ensemble",
            "accuracy": acc,
            "roc_auc": auc,
            "pr_auc": float(average_precision_score(y, p)),
            "f1": float(f1_score(y, hard)),
            "sensitivity": float(recall_score(y, hard)),
            "specificity": float(spec),
            "brier": float(brier_score_loss(y, p)),
            "accuracy_ci95": acc_ci,
            "auc_ci95": auc_ci,
            "n_rows": load.n_unique, "n_positive": int(y.sum()),
            "folds": N_SPLITS, "n_repeats": N_REPEATS,
        }
        rows.append(row)
        print(f"{name:20s} acc={row['accuracy']:.4f} auc={row['roc_auc']:.4f} brier={row['brier']:.4f}", flush=True)
    rows.sort(key=lambda r: -r["roc_auc"])
    out = {
        "models": rows,
        "folds": N_SPLITS, "n_repeats": N_REPEATS,
        "n_rows": load.n_unique, "n_unique": load.n_unique, "n_raw": load.n_raw,
        "n_positive": int(y.sum()), "bootstrap": N_BOOT, "seed": RNG,
        "deduped": True, "rf_trees": RF_TREES,
        "notes": "Exact duplicate profiles dropped before splitting. "
                 "Metrics are repeated stratified out-of-fold on unique profiles; "
                 "scaling is fit inside each fold. The shipped stack uses the same "
                 "100-tree forest; its stacker is fit on out-of-fold expert "
                 "probabilities, so these CV numbers describe the recipe, not a "
                 "test of the exported weights.",
    }
    with open("train/comparison.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote train/comparison.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
