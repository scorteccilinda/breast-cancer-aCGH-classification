import os
import warnings
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from scipy import stats
 
from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.model_selection  import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors        import NearestCentroid
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import accuracy_score
from xgboost                  import XGBClassifier
from skopt                    import BayesSearchCV
from skopt.space              import Real, Integer, Categorical
 
# suppress skopt duplicate-point warning
warnings.filterwarnings("ignore", message=".*objective.*evaluated.*point.*before.*", category=UserWarning)
# suppress NearestCentroid zero-std warning (expected with discrete data)
warnings.filterwarnings("ignore", message=".*within_class_std_dev_.*", category=UserWarning)
 
DATA_DIR   = "data"
OUTPUT_DIR = "output"
MODEL_DIR  = "model"
 
 
class KruskalFilter(BaseEstimator, TransformerMixin):
    """
    Selects the top k genomic regions by Kruskal-Wallis H-statistic.
 
    Fit is called only on the training fold, never on the test fold.
    This ensures no data leakage when used inside a cross-validation loop,
    as indicated by the Wessels et al. (2005) protocol.
 
    Parameters
    ----------
    k : int
        Number of top features to keep.
    """
    def __init__(self, k=100):
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
 
 
def build_pipelines():
    """
    Returns a dictionary of pipelines, one per classifier.
    Each pipeline:
        1. KruskalFilter  -- feature selection on training data only (no leakage)
        2. StandardScaler -- only for SVM pipelines (scale-sensitive)
        3. Classifier
    """
    return {
        "NearestCentroid": Pipeline([
            ("kruskal", KruskalFilter()),
            ("clf",     NearestCentroid()),
        ]),
 
        "SVM_linear": Pipeline([
            ("kruskal", KruskalFilter()),
            ("scaler",  StandardScaler()),
            ("clf",     SVC(kernel="linear", probability=True)),
        ]),
 
        "SVM_RBF": Pipeline([
            ("kruskal", KruskalFilter()),
            ("scaler",  StandardScaler()),
            ("clf",     SVC(kernel="rbf", probability=True)),
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
 
 
def build_search_spaces():
    """
    BayesSearchCV search spaces for each classifier.
    kruskal__k is included in every space so feature selection is optimised
    jointly with classifier hyperparameters inside the inner CV loop.
    """
    return {
        "NearestCentroid": {
            "kruskal__k":             Integer(83, 607),
            "clf__shrink_threshold":  Categorical([None, 0.1, 0.25, 0.5,
                                                   0.75, 1.0, 1.5, 2.0]),
        },
 
        "SVM_linear": {
            "kruskal__k": Integer(83, 607),
            "clf__C":     Real(1e-3, 1e2, prior="log-uniform"),
        },
 
        "SVM_RBF": {
            "kruskal__k": Integer(83, 607),
            "clf__C":     Real(1e-2, 1e2, prior="log-uniform"),
            "clf__gamma": Categorical(["scale", "auto", 1e-3, 1e-2, 1e-1]),
        },
 
        "RandomForest": {
            "kruskal__k":              Integer(83, 607),
            "clf__n_estimators":       Integer(100, 500),
            "clf__max_depth":          Categorical([None, 5, 10, 20]),
            "clf__min_samples_split":  Integer(2, 10),
            "clf__min_samples_leaf":   Integer(1, 4),
            "clf__max_features":       Categorical(["sqrt", "log2", None]),
        },
 
        "XGBoost": {
            "kruskal__k":              Integer(83, 607),
            "clf__n_estimators":       Integer(100, 500),
            "clf__max_depth":          Integer(3, 10),
            "clf__learning_rate":      Real(1e-3, 3e-1, prior="log-uniform"),
            "clf__subsample":          Real(0.6, 1.0),
            "clf__colsample_bytree":   Real(0.6, 1.0),
            "clf__reg_alpha":          Real(0, 1.0),
            "clf__reg_lambda":         Real(1.0, 2.0),
        },
    }
 
 
def train_and_evaluate():
    """
    Training and evaluation following the Wessels et al. (2005) protocol.
 
    No hold-out test set — all 100 samples are used in the outer CV.
    The mean CV accuracy across folds IS the accuracy estimate (V).
    Training accuracy per fold (T) is also recorded to check for overfitting.
 
    Note on ROC AUC: not computed here because there is no held-out test set
    in the Wessels protocol. The CV accuracy averaged across 90 folds is the
    primary performance metric.
 
    OUTER LOOP: RepeatedStratifiedKFold (3-fold x 30 repeats = 90 folds)
      Each fold: ~67 train, ~33 test
      INNER LOOP: BayesSearchCV (5-fold CV, 30 iterations)
        - tries 30 combinations of (k, hyperparams)
        - KruskalFilter and StandardScaler fit on training fold only
        - selects best combination by mean 5-fold CV accuracy
        - retrains on full ~67 training samples with best params
 
    After CV: retrain best classifier on all 100 samples for predictions.
 
    Saves
    -----
    output/cv_results_wessels.csv     per-fold val + train accuracy, best_k, best_params
    output/cv_summary_wessels.csv     mean +/- std val + train accuracy per classifier
    output/best_k_summary_wessels.csv distribution of selected k per classifier
    model/model.pkl              best classifier retrained on all 100 samples
    model/label_encoder.pkl           label encoder for decoding predictions
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)
 
    # load data
    X = pd.read_csv(f"{DATA_DIR}/x_train.csv", index_col=0)
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv", index_col=0).squeeze()
 
    # encode labels (required for XGBoost, consistent for all)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Total samples:  {X.shape[0]}")
    print(f"Total regions:  {X.shape[1]}")
 
    X_all = X.values
    y_all = y_encoded
 
    pipelines     = build_pipelines()
    search_spaces = build_search_spaces()
 
    # outer CV: 3-fold x 30 repeats = 90 folds (Wessels protocol)
    # 3-fold gives ~67 train / ~33 test per fold
    outer_cv = RepeatedStratifiedKFold(
        n_splits=3, n_repeats=30, random_state=42)
 
    # inner CV used by BayesSearchCV (5-fold, cheaper than 10-fold because
    # we tune many parameters across 30 Bayesian iterations per fold)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
    n_outer_folds = 3 * 30   # 90
    all_results   = []
    best_score    = 0
    best_name     = None
    best_model    = None
 
    for name, pipeline in pipelines.items():
        print(f"\n{'='*55}")
        print(f"Training: {name}")
        print(f"{'='*55}")
 
        fold_scores       = []
        fold_train_scores = []
        fold_k            = []
        fold_params       = []
 
        for fold_idx, (train_idx, val_idx) in enumerate(
                outer_cv.split(X_all, y_all)):
 
            X_fold_train = X_all[train_idx]
            y_fold_train = y_all[train_idx]
            X_fold_val   = X_all[val_idx]
            y_fold_val   = y_all[val_idx]
 
            # inner loop: 30-iteration Bayesian search over (k, hyperparams)
            # evaluated by 5-fold CV on X_fold_train only
            search = BayesSearchCV(
                estimator     = pipeline,
                search_spaces = search_spaces[name],
                n_iter        = 30,
                cv            = inner_cv,
                scoring       = "accuracy",
                n_jobs        = -1,
                random_state  = fold_idx,
                verbose       = 0,
            )
            search.fit(X_fold_train, y_fold_train)
 
            # validation accuracy v_j — on unseen outer fold
            val_score   = accuracy_score(y_fold_val,   search.predict(X_fold_val))
            # training accuracy t_j* — on the samples used to train
            train_score = accuracy_score(y_fold_train, search.predict(X_fold_train))
 
            fold_scores.append(val_score)
            fold_train_scores.append(train_score)
 
            best_k = search.best_params_.get("kruskal__k", np.nan)
            fold_k.append(best_k)
            fold_params.append(str(search.best_params_))
 
            if (fold_idx + 1) % 15 == 0:
                print(f"  Fold {fold_idx+1:2d}/{n_outer_folds} — "
                      f"val: {val_score:.3f}  train: {train_score:.3f}  "
                      f"k: {best_k}")
 
        mean_val   = np.mean(fold_scores)
        std_val    = np.std(fold_scores)
        mean_train = np.mean(fold_train_scores)
        std_train  = np.std(fold_train_scores)
        mean_k     = np.nanmean(fold_k)
 
        print(f"\n  CV val accuracy   (V): {mean_val:.3f} ± {std_val:.3f}")
        print(f"  CV train accuracy (T): {mean_train:.3f} ± {std_train:.3f}")
        print(f"  Best k               : {mean_k:.1f} ± {np.nanstd(fold_k):.1f}"
              f"  (range {int(np.nanmin(fold_k))}–{int(np.nanmax(fold_k))})")
 
        # print most frequent best-params combination for quick inspection
        most_common = Counter(fold_params).most_common(1)[0][0]
        print(f"  Most frequent best params:\n    {most_common}")
 
        for i, (val, train, k, params) in enumerate(
                zip(fold_scores, fold_train_scores, fold_k, fold_params)):
            all_results.append({
                "classifier":     name,
                "fold":           i,
                "val_accuracy":   val,
                "train_accuracy": train,
                "best_k":         k,
                "best_params":    params,
            })
 
        if mean_val > best_score:
            best_score = mean_val
            best_name  = name
            best_model = search
 
    # save CV results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{OUTPUT_DIR}/cv_results_wessels.csv", index=False)
 
    summary = (results_df.groupby("classifier")[["val_accuracy", "train_accuracy"]]
               .agg(["mean", "std"])
               .sort_values(("val_accuracy", "mean"), ascending=False))
    summary.to_csv(f"{OUTPUT_DIR}/cv_summary_wessels.csv")
 
    k_summary = (results_df.groupby("classifier")["best_k"]
                 .agg(["mean", "std", "min", "max"])
                 .round(1))
    k_summary.to_csv(f"{OUTPUT_DIR}/best_k_summary_wessels.csv")
 
    print(f"\n{'='*55}")
    print("CV SUMMARY:")
    print(summary.to_string())
    print(f"\nBest k distribution by classifier:")
    print(k_summary.to_string())
    print(f"\nBest classifier: {best_name} (mean val acc = {best_score:.3f})")
 
    # retrain best classifier on ALL 100 samples
    print(f"\nRetraining {best_name} on all {X_all.shape[0]} samples...")
    final_model = best_model.best_estimator_
    final_model.fit(X_all, y_all)
 
    joblib.dump(final_model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(le,          f"{MODEL_DIR}/label_encoder.pkl")
    print(f"Model saved to {MODEL_DIR}/model.pkl")
    print(f"\nAccuracy estimate for 57 validation samples: "
          f"{int(round(best_score * 57))} / 57")
 
 
def main():
    train_and_evaluate()
 
 
if __name__ == "__main__":
    main()