"""Does GAN augmentation help train the diabetes model?

Leakage-safe experiment on unique questionnaire profiles (520 rows -> 251
unique; exact copies are dropped before splitting, otherwise every arm just
memorizes):

For each of 5 outer folds:
  - split the fold's train slice into fit (CTGAN training) and select
    (filter reference) parts — the held-out test fold is NEVER touched
    until final scoring, not even for adversarial filtering;
  - fit CTGAN ONLY on the fit part (labels included, the standard setup);
  - generate synthetic rows with their generated labels, adversarially
    filter: train a RF discriminator to tell synth rows from the select
    rows and keep the most select-like ones (no ground-truth labels used);
  - train the stacked-ensemble recipe on (full original train + synth) and
    score on the untouched test fold.

Arms:
  baseline        — recipe on original train only
  gan_augmented   — recipe on train + CTGAN synth (generated labels)
  gan_only        — recipe on synth only (stress test)
  oversampled     — duplicate the smaller class until balanced (here that is
                    the negative class) — the classic cheap baseline
Big question, answered honestly: does any of this beat baseline?

Metric: out-of-fold accuracy / ROC-AUC / Brier + per-fold spread.
Writes train/gan_results.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from ctgan import CTGAN

from export import load

RNG = 7
N_FOLDS = 5
SYN_PER_FOLD = 400
RF_TREES = 100


def recipe():
    return make_pipeline(StandardScaler(), StackingClassifier(
        estimators=[("lr", LogisticRegression(C=1.0, max_iter=2000)), ("nb", GaussianNB()),
                    ("knn", KNeighborsClassifier(n_neighbors=21, weights="distance")),
                    ("gb", GradientBoostingClassifier(random_state=RNG)),
                    ("rf", RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG, n_jobs=-1))],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000), cv=5, n_jobs=-1))


def adversarial_filter(synth, ref, keep=SYN_PER_FOLD // 2):
    """RF discriminator: synth = 0, real reference rows = 1. Keep synth rows
    the discriminator finds MOST reference-like (highest P(ref)).
    The reference is a slice of TRAIN — never the held-out test fold.
    Ground-truth labels never enter."""
    disc = RandomForestClassifier(n_estimators=200, random_state=RNG, n_jobs=-1)
    X = np.vstack([synth, ref])
    y = np.array([0] * len(synth) + [1] * len(ref))
    disc.fit(X, y)
    p = disc.predict_proba(synth)[:, 1]
    order = np.argsort(-p)[:keep]
    return order


def main():
    X, y, feats = load()
    print(f"unique profiles: {load.n_unique} of {load.n_raw} | positives={y.sum()}")
    arms = ["baseline", "gan_augmented", "gan_only", "oversampled"]
    fold_metrics = {a: {"acc": [], "auc": [], "brier": []} for a in arms}
    gan_stats = {"kept_per_fold": []}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

        # baseline
        clf = recipe().fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        fold_metrics["baseline"]["acc"].append(accuracy_score(yte, p >= 0.5))
        fold_metrics["baseline"]["auc"].append(roc_auc_score(yte, p))
        fold_metrics["baseline"]["brier"].append(brier_score_loss(yte, p))

        # CTGAN fit on a fit-slice of TRAIN ONLY; filter against a select
        # slice of train. The test fold stays untouched until scoring.
        fit_idx, sel_idx = train_test_split(
            np.arange(len(Xtr)), test_size=0.25, stratify=ytr, random_state=RNG + fi)
        tr_df = pd.DataFrame(Xtr[fit_idx], columns=feats)
        tr_df["target"] = ytr[fit_idx]
        ctgan = CTGAN(epochs=300, verbose=False)
        ctgan.set_random_state(RNG + fi)
        ctgan.fit(tr_df)
        synth = ctgan.sample(SYN_PER_FOLD)
        synth_X = synth[feats].to_numpy(float)
        synth_y = np.rint(synth["target"].to_numpy(float)).astype(int)
        synth_y = np.clip(synth_y, 0, 1)

        keep_order = adversarial_filter(synth_X, Xtr[sel_idx])
        kept_X, kept_y = synth_X[keep_order], synth_y[keep_order]
        gan_stats["kept_per_fold"].append(len(kept_X))

        # gan_augmented — generated labels, same source as gan_only
        Xa = np.vstack([Xtr, kept_X])
        ya = np.concatenate([ytr, kept_y])
        clf = recipe().fit(Xa, ya)
        p = clf.predict_proba(Xte)[:, 1]
        fold_metrics["gan_augmented"]["acc"].append(accuracy_score(yte, p >= 0.5))
        fold_metrics["gan_augmented"]["auc"].append(roc_auc_score(yte, p))
        fold_metrics["gan_augmented"]["brier"].append(brier_score_loss(yte, p))

        # gan_only
        clf = recipe().fit(synth_X, synth_y)
        p = clf.predict_proba(Xte)[:, 1]
        fold_metrics["gan_only"]["acc"].append(accuracy_score(yte, p >= 0.5))
        fold_metrics["gan_only"]["auc"].append(roc_auc_score(yte, p))
        fold_metrics["gan_only"]["brier"].append(brier_score_loss(yte, p))

        # oversampled: duplicate the SMALLER class until balanced
        # (on this data that is the negative class)
        rng = np.random.default_rng(RNG + fi)
        pos_idx = np.where(ytr == 1)[0]
        neg_idx = np.where(ytr == 0)[0]
        small = pos_idx if len(pos_idx) < len(neg_idx) else neg_idx
        extra = rng.choice(small, abs(len(pos_idx) - len(neg_idx)), replace=True)
        Xo = np.vstack([Xtr, Xtr[extra]])
        yo = np.concatenate([ytr, ytr[extra]])
        clf = recipe().fit(Xo, yo)
        p = clf.predict_proba(Xte)[:, 1]
        fold_metrics["oversampled"]["acc"].append(accuracy_score(yte, p >= 0.5))
        fold_metrics["oversampled"]["auc"].append(roc_auc_score(yte, p))
        fold_metrics["oversampled"]["brier"].append(brier_score_loss(yte, p))

        print(f"fold {fi}: done", flush=True)

    results = {"n_folds": N_FOLDS, "syn_per_fold": SYN_PER_FOLD, "epochs": 300,
               "deduped": True, "n_unique": int(load.n_unique), "n_raw": int(load.n_raw),
               "filter_reference": "train select-slice (test fold never touched before scoring)",
               "synth_labels": "CTGAN-generated target column in both gan arms",
               "arms": {}}
    for a in arms:
        m = fold_metrics[a]
        results["arms"][a] = {
            "acc_mean": float(np.mean(m["acc"])), "acc_std": float(np.std(m["acc"])),
            "auc_mean": float(np.mean(m["auc"])), "auc_std": float(np.std(m["auc"])),
            "brier_mean": float(np.mean(m["brier"])), "brier_std": float(np.std(m["brier"])),
            "folds": {k: [float(x) for x in v] for k, v in m.items()},
        }
    results["gan_stats"] = gan_stats
    with open("train/gan_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({a: round(v["auc_mean"], 4) for a, v in results["arms"].items()}, indent=1))
    print("wrote train/gan_results.json")


if __name__ == "__main__":
    main()
