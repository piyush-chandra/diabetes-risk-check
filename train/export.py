"""Fit the shipped 5-expert stack on unique profiles and export model.json.

Duplicates: this file has 520 rows but 251 unique feature profiles, with no
conflicting labels. Copies are dropped before any fit — otherwise a split
puts the same person on both sides, and KNN memorizes copied rows.

Stacker: logistic regression fit on OUT-OF-FOLD expert probabilities
(5-fold), not on in-sample probabilities. The exported experts are then
refit on all unique rows so the browser has a single model. The CV number
in comparison.json describes the recipe; it is not a test of these weights.

The forest is 100 trees, the same forest the comparison measures — never a
larger forest with trees sliced off.

model.js must reproduce predict_proba of THIS artifact. train/parity.js
checks against train/parity_fixture.json, written here from these exact
fitted objects — never against a fresh retrain.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

RNG = 7
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


def export_tree(est, regress=False):
    t = est.tree_
    v = t.value
    d = {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feat": t.feature.tolist(),
        "thr": t.threshold.tolist(),
    }
    if regress:
        # gradient-boosting residual leaf: single raw value
        d["val"] = v[:, 0, 0].tolist()
    else:
        # classifier leaf: [neg, pos] weighted counts
        d["neg"] = v[:, 0, 0].tolist()
        d["pos"] = v[:, 0, 1].tolist()
    return d


def fit_experts(Xs, y):
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y)
    nb = GaussianNB().fit(Xs, y)
    knn = KNeighborsClassifier(n_neighbors=21, weights="distance").fit(Xs, y)
    gb = GradientBoostingClassifier(random_state=RNG).fit(Xs, y)
    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG, n_jobs=-1).fit(Xs, y)
    return lr, nb, knn, gb, rf


def expert_probs(models, Xs):
    lr, nb, knn, gb, rf = models
    return np.column_stack([
        lr.predict_proba(Xs)[:, 1], nb.predict_proba(Xs)[:, 1],
        knn.predict_proba(Xs)[:, 1], gb.predict_proba(Xs)[:, 1],
        rf.predict_proba(Xs)[:, 1],
    ])


def main():
    X, y, feats = load()
    print(f"rows: {load.n_raw} -> {load.n_unique} unique | positives={y.sum()} ({y.mean()*100:.1f}%)")

    # Honest quality check of THIS recipe: same structure, 5-fold OOF.
    stack = make_pipeline(StandardScaler(), StackingClassifier(
        estimators=[("lr", LogisticRegression(C=1.0, max_iter=2000)), ("nb", GaussianNB()),
                    ("knn", KNeighborsClassifier(n_neighbors=21, weights="distance")),
                    ("gb", GradientBoostingClassifier(random_state=RNG)),
                    ("rf", RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG, n_jobs=-1))],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000), cv=5, n_jobs=-1))
    oof = cross_val_predict(stack, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RNG),
                            method="predict_proba")[:, 1]
    from sklearn.metrics import accuracy_score, roc_auc_score
    print(f"recipe 5-fold OOF: acc={accuracy_score(y, oof >= 0.5):.4f} auc={roc_auc_score(y, oof):.4f}")

    # OOF expert probabilities -> the stacker that ships.
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    P_oof = np.zeros((len(y), 5))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        P_oof[te] = expert_probs(fit_experts(sc.transform(X[tr]), y[tr]), sc.transform(X[te]))
    st = LogisticRegression(C=1.0, max_iter=2000).fit(P_oof, y)
    p_oof = st.predict_proba(P_oof)[:, 1]
    print(f"shipped stacker on OOF experts: acc={accuracy_score(y, p_oof >= 0.5):.4f} auc={roc_auc_score(y, p_oof):.4f}")

    # Refit experts once on all unique rows for the browser artifact.
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    lr, nb, knn, gb, rf = fit_experts(Xs, y)

    # Bands from the OOF recipe scores: low = below the 90th percentile of
    # true-negative scores; high = at/above the 10th percentile of
    # true-positive scores. Falls back to 0.25/0.60 if those cross.
    neg, pos = oof[y == 0], oof[y == 1]
    low_max, high_min = float(np.quantile(neg, 0.90)), float(np.quantile(pos, 0.10))
    if not (0.02 < low_max < high_min < 0.98):
        low_max, high_min = 0.25, 0.60
    print(f"bands: low<{low_max:.3f} moderate high>={high_min:.3f}")

    # KNN: exact Age + 15-bit mask (bit i-1 = feature i: Gender + 14 symptoms).
    masks, ages = [], []
    for row in X:
        m = 0
        for i in range(1, 16):
            if row[i] >= 0.5:
                m |= (1 << (i - 1))
        masks.append(m)
        ages.append(float(row[0]))

    # Binary GB initial raw score = log prior odds (binomial deviance init).
    p1 = float(y.mean())
    gb_init = float(np.log(p1 / (1 - p1)))

    model = {
        "features": feats,
        "scaler": {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
        "experts": {
            "lr": {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])},
            "nb": {"theta": nb.theta_.tolist(), "sigma": np.maximum(nb.var_, 1e-300).tolist(),
                   "priors": nb.class_prior_.tolist(), "classes": nb.classes_.tolist()},
            "knn": {"k": 21, "masks": masks, "ages": ages, "y": y.tolist()},
            "gb": {"trees": [export_tree(e[0], regress=True) for e in gb.estimators_],
                   "lr": float(gb.learning_rate), "init": gb_init},
            "rf": {"n_trees": RF_TREES,
                   "trees": [export_tree(e) for e in rf.estimators_]},
        },
        "stacker": {"coef": st.coef_[0].tolist(), "intercept": float(st.intercept_[0]),
                    "names": ["lr", "nb", "knn", "gb", "rf"]},
        "meta": {
            "n_rows_raw": load.n_raw, "n_unique": load.n_unique,
            "n_positive": int(y.sum()), "rf_trees": RF_TREES,
            "stacker": "logistic regression on 5-fold out-of-fold expert probabilities",
            "bands": {"low_max": low_max, "high_min": high_min},
            "source": "UCI Early Stage Diabetes Risk Prediction (Sylhet Diabetes Hospital, Bangladesh; Islam et al. 2019; DOI 10.24432/C5VG8H)",
        },
    }
    with open("train/model.json", "w") as f:
        json.dump(model, f)
    import os
    raw = os.path.getsize("train/model.json")
    print(f"wrote train/model.json ({raw/1024:.0f} KB raw)")

    # Fixture for parity.js: THIS artifact's predictions on unique rows plus
    # off-table edge cases (age extremes, both sexes with no symptoms, all-yes).
    edge = np.array([
        [16, 0, *([0] * 14)], [90, 1, *([1] * 14)],
        [45, 0, *([0] * 14)], [45, 1, *([0] * 14)],
        [35, 0, *([0] * 14)], [65, 1, *([1] * 14)],
    ], dtype=float)
    rows = np.vstack([X, edge])
    P = expert_probs((lr, nb, knn, gb, rf), sc.transform(rows))
    stack = st.predict_proba(P)[:, 1]
    with open("train/parity_fixture.json", "w") as f:
        json.dump({
            "rows": rows.tolist(),
            "parts": {n: P[:, i].tolist() for i, n in enumerate(["lr", "nb", "knn", "gb", "rf"])},
            "stack": stack.tolist(),
            "n_train": len(X),
        }, f)
    print(f"wrote train/parity_fixture.json ({len(rows)} rows: {len(X)} unique + 6 edge)")


if __name__ == "__main__":
    main()
