#!/usr/bin/env python3
"""
Hierarchical Breast Cancer Classifier 
====================================================================

Two-stage hierarchical classification pipeline motivated by:
1. SHAP analysis showing chr17 ERBB2 signal dominates HER2+ predictions
2. Confusion matrix showing HR+ vs Triple Neg as the main error source

Stage 1: HER2+ vs Rest using only chr17 features
Stage 2: HR+ vs Triple Neg using non-chr17 features

Works directly on the 2834 original regions.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats

from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors        import NearestCentroid
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import balanced_accuracy_score, confusion_matrix
from xgboost                  import XGBClassifier
from skopt                    import BayesSearchCV
from skopt.space              import Real, Integer, Categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_DIR   = "data"
OUTPUT_DIR = "output"
MODEL_DIR  = "model"


# ─────────────────────────────────────────────────────────────────────────────
# KRUSKAL-WALLIS FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
class KruskalFilter(BaseEstimator, TransformerMixin):
    """
    Selects top k features by Kruskal-Wallis H-statistic.
    Fit on training data only — no data leakage.
    """
    def __init__(self, k=50):
        self.k = k

    def fit(self, X, y):
        subtypes = np.unique(y)
        h_scores = []
        for col in range(X.shape[1]):
            groups = [X[y == s, col] for s in subtypes]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    h, _ = stats.kruskal(*groups)
            except ValueError:
                h = 0.0
            h_scores.append(h)
        self.selected_ = np.argsort(h_scores)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.selected_]


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PIPELINES AND SEARCH SPACES
# ─────────────────────────────────────────────────────────────────────────────
def build_pipelines_and_spaces(k_min, k_max):
    pipelines = {
        "NearestCentroid": Pipeline([
            ("kruskal", KruskalFilter()),
            ("clf",     NearestCentroid()),
        ]),
        "RandomForest": Pipeline([
            ("kruskal", KruskalFilter()),
            ("clf",     RandomForestClassifier(random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("kruskal", KruskalFilter()),
            ("clf",     XGBClassifier(
                            eval_metric="mlogloss",
                            random_state=42,
                            verbosity=0,
                        )),
        ]),
    }

    search_spaces = {
        "NearestCentroid": {
            "kruskal__k":            Integer(k_min, k_max),
            "clf__shrink_threshold": Categorical([None, 0.1, 0.25, 0.5, 1.0, 1.5]),
        },
        "RandomForest": {
            "kruskal__k":            Integer(k_min, k_max),
            "clf__n_estimators":     Integer(100, 500),
            "clf__max_depth":        Categorical([None, 5, 10, 20]),
            "clf__min_samples_leaf": Integer(1, 5),
            "clf__max_features":     Categorical(["sqrt", "log2"]),
        },
        "XGBoost": {
            "kruskal__k":            Integer(k_min, k_max),
            "clf__n_estimators":     Integer(100, 500),
            "clf__max_depth":        Integer(3, 8),
            "clf__learning_rate":    Real(1e-3, 3e-1, prior="log-uniform"),
            "clf__subsample":        Real(0.6, 1.0),
            "clf__colsample_bytree": Real(0.6, 1.0),
        },
    }

    return pipelines, search_spaces


# ─────────────────────────────────────────────────────────────────────────────
# CV LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_cv_stage(X, y, pipelines, search_spaces, outer_cv, inner_cv, stage_name):
    print(f"\n{'='*60}")
    print(f"{stage_name}")
    print(f"{'='*60}")

    all_scores = {}
    for name, pipeline in pipelines.items():
        print(f"\n  {name}:")
        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(outer_cv.split(X, y)):
            X_tr, X_cv = X[tr_idx], X[val_idx]
            y_tr, y_cv = y[tr_idx], y[val_idx]
            search = BayesSearchCV(
                estimator     = pipeline,
                search_spaces = search_spaces[name],
                n_iter        = 20,
                cv            = inner_cv,
                scoring       = "balanced_accuracy",
                n_jobs        = -1,
                random_state  = fold_idx,
                verbose       = 0,
            )
            search.fit(X_tr, y_tr)
            val_score = balanced_accuracy_score(y_cv, search.predict(X_cv))
            fold_scores.append(val_score)

        all_scores[name] = fold_scores
        print(f"    BA = {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    best_name  = max(all_scores, key=lambda k: np.mean(all_scores[k]))
    best_score = np.mean(all_scores[best_name])
    print(f"\n  Best: {best_name} (BA = {best_score:.3f})")
    return best_name, best_score, all_scores


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)

    # Load data
    print("Loading data...")
    meta_cols  = ["Start", "End", "Nclone"]
    train_call = pd.read_csv(f"{DATA_DIR}/Train_call.tsv", sep="\t", index_col=0)
    region_names = [f"chr{c}_{int(s)}_{int(e)}" for c, s, e in zip(
        train_call.index, train_call["Start"], train_call["End"])]
    train_call.index = region_names
    train_call = train_call.drop(columns=meta_cols)
    X_raw = train_call.T

    clinical = pd.read_csv(f"{DATA_DIR}/Train_clinical.tsv", sep="\t", index_col=0)
    y_raw    = clinical.loc[X_raw.index, "Subgroup"]

    val_call = pd.read_csv(f"{DATA_DIR}/Validation_call.tsv", sep="\t", index_col=0)
    val_call.index = region_names
    val_call = val_call.drop(columns=meta_cols)
    X_val_raw = val_call.T

    print(f"Training:   {X_raw.shape}")
    print(f"Validation: {X_val_raw.shape}")
    print(f"Classes:\n{y_raw.value_counts()}")

    # Split features by chromosome
    # chr17 features for Stage 1 (HER2+ separation)
    # non-chr17 features for Stage 2 (HR+ vs Triple Neg)
    chr17_cols     = [c for c in X_raw.columns if c.startswith("chr17_")]
    non_chr17_cols = [c for c in X_raw.columns if not c.startswith("chr17_")]
    print(f"\nChr17 features:     {len(chr17_cols)}")
    print(f"Non-chr17 features: {len(non_chr17_cols)}")

    X_stage1     = X_raw[chr17_cols].values
    X_stage2_all = X_raw[non_chr17_cols].values
    X_val_s1     = X_val_raw[chr17_cols].values
    X_val_s2     = X_val_raw[non_chr17_cols].values

    # Labels as integers (required for XGBoost)
    y_stage1      = np.where(y_raw.values == "HER2+", 1, 0)
    non_her2_mask = y_raw.values != "HER2+"
    X_stage2      = X_stage2_all[non_her2_mask]
    y_stage2_str  = y_raw.values[non_her2_mask]
    y_stage2      = np.where(y_stage2_str == "HR+", 0, 1)

    print(f"\nStage 1: {len(y_stage1)} samples (HER2+: {(y_stage1==1).sum()}, rest: {(y_stage1==0).sum()})")
    print(f"Stage 2: {len(y_stage2)} samples (HR+: {(y_stage2==0).sum()}, TN: {(y_stage2==1).sum()})")

    # CV setup — Wessels protocol
    outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=30, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Stage 1
    pipes_s1, ss_s1 = build_pipelines_and_spaces(1, min(50, len(chr17_cols)))
    best_name_s1, best_score_s1, results_s1 = run_cv_stage(
        X_stage1, y_stage1, pipes_s1, ss_s1,
        outer_cv, inner_cv, "STAGE 1: HER2+ vs Rest (chr17 features)")

    # Stage 2
    pipes_s2, ss_s2 = build_pipelines_and_spaces(10, 200)
    best_name_s2, best_score_s2, results_s2 = run_cv_stage(
        X_stage2, y_stage2, pipes_s2, ss_s2,
        outer_cv, inner_cv, "STAGE 2: HR+ vs Triple Neg (non-chr17 features)")

    # Estimate accuracy
    p_her2 = 32/100; p_hr = 36/100; p_tn = 32/100
    est_acc     = p_her2 * best_score_s1 + p_hr * best_score_s2 + p_tn * best_score_s2
    est_correct = round(est_acc * 57)
    print(f"\nEstimated accuracy: {est_acc:.3f}")
    print(f"Estimated correct:  {est_correct}/57")

    # Train final models on all data
    print(f"\nTraining final Stage 1 ({best_name_s1})...")
    pipes_f1, ss_f1 = build_pipelines_and_spaces(1, min(50, len(chr17_cols)))
    final_s1 = BayesSearchCV(
        estimator=pipes_f1[best_name_s1], search_spaces=ss_f1[best_name_s1],
        n_iter=20, cv=inner_cv, scoring="balanced_accuracy",
        n_jobs=-1, random_state=42, verbose=0)
    final_s1.fit(X_stage1, y_stage1)
    model_s1 = final_s1.best_estimator_

    print(f"Training final Stage 2 ({best_name_s2})...")
    pipes_f2, ss_f2 = build_pipelines_and_spaces(10, 200)
    final_s2 = BayesSearchCV(
        estimator=pipes_f2[best_name_s2], search_spaces=ss_f2[best_name_s2],
        n_iter=20, cv=inner_cv, scoring="balanced_accuracy",
        n_jobs=-1, random_state=42, verbose=0)
    final_s2.fit(X_stage2, y_stage2)
    model_s2 = final_s2.best_estimator_

    # Generate confusion matrix on training data (for report figure)
    s1_train_preds    = model_s1.predict(X_stage1)
    non_her2_train    = s1_train_preds == 0
    s2_train_preds_n  = model_s2.predict(X_stage2_all[non_her2_train])
    s2_train_preds    = np.where(s2_train_preds_n == 0, "HR+", "Triple Neg")
    all_train_preds   = np.where(s1_train_preds == 1, "HER2+", "")
    all_train_preds[non_her2_train] = s2_train_preds

    plot_confusion_matrix(
        y_raw.values, all_train_preds,
        labels=["HER2+", "HR+", "Triple Neg"],
        title="Hierarchical classifier — training set confusion matrix",
        filename="confusion_matrix_hierarchical.png"
    )

    # Predict validation set
    print("\nPredicting validation set...")
    s1_preds     = model_s1.predict(X_val_s1)
    non_her2_val = s1_preds == 0
    s2_preds_num = model_s2.predict(X_val_s2[non_her2_val])
    s2_preds     = np.where(s2_preds_num == 0, "HR+", "Triple Neg")

    final_preds = np.where(s1_preds == 1, "HER2+", "")
    final_preds[non_her2_val] = s2_preds

    print("Prediction distribution:")
    unique, counts = np.unique(final_preds, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt}")

    # Save everything
    joblib.dump({"stage1": model_s1, "stage2": model_s2,
                 "chr17_cols": chr17_cols,
                 "non_chr17_cols": non_chr17_cols},
                f"{MODEL_DIR}/hierarchical_v2_model.pkl")

    with open(f"{OUTPUT_DIR}/prediction_hierarchical_v2.txt", "w") as f:
        f.write('"Sample"\t"Subgroup"\n')
        for sample, pred in zip(X_val_raw.index, final_preds):
            f.write(f'"{sample}"\t"{pred}"\n')

    with open(f"{OUTPUT_DIR}/estimate_hierarchical_v2.txt", "w") as f:
        f.write(str(est_correct))

    print(f"\nSaved: model, predictions, estimate ({est_correct}/57)")
    print(f"\nFull results summary:")
    print(f"\nStage 1:")
    for name, scores in results_s1.items():
        print(f"  {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"\nStage 2:")
    for name, scores in results_s2.items():
        print(f"  {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


if __name__ == "__main__":
    main()
