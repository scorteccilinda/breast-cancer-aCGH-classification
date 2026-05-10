import os
import numpy as np
import pandas as pd
import joblib
from scipy import stats

from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.model_selection  import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.neighbors        import NearestCentroid
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import accuracy_score, classification_report, confusion_matrix
from xgboost                  import XGBClassifier
from skopt                    import BayesSearchCV
from skopt.space              import Real, Integer, Categorical

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
    k : int
        Number of top features to keep.
    """
    def __init__(self, k=100):
        self.k = k

    def fit(self, X, y):
        subtypes  = np.unique(y)
        h_scores  = []
        for col in range(X.shape[1]):
            groups = [X[y == s, col] for s in subtypes]
            h, _   = stats.kruskal(*groups)
            h_scores.append(h)
        # keep indices of top k features by H-score (higher = more discriminative)
        self.selected_ = np.argsort(h_scores)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.selected_]


def build_pipelines():
    """
    Returns a dictionary of pipelines, one per classifier.
    Each pipeline contains:
        1. KruskalFilter  -- feature selection on training data only
        2. StandardScaler -- only for SVM pipelines (scale-sensitive)
        3. Classifier

    The k parameter in KruskalFilter is treated as a hyperparameter
    and will be tuned by BayesSearchCV alongside classifier parameters.
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
    k is included in every space so feature selection is optimised jointly
    with classifier hyperparameters inside the inner cross-validation loop.
    """
    k_space = Integer(50, 600, name="kruskal__k")

    return {
        "NearestCentroid": {
            "kruskal__k":             Integer(50, 600),
            "clf__shrink_threshold":  Categorical([None, 0.1, 0.25, 0.5,
                                                   0.75, 1.0, 1.5, 2.0]),
        },

        "SVM_linear": {
            "kruskal__k": Integer(50, 600),
            "clf__C":     Real(1e-3, 1e2, prior="log-uniform"),
        },

        "SVM_RBF": {
            "kruskal__k": Integer(50, 600),
            "clf__C":     Real(1e-2, 1e2, prior="log-uniform"),
            "clf__gamma": Categorical(["scale", "auto", 1e-3, 1e-2, 1e-1]),
        },

        "RandomForest": {
            "kruskal__k":              Integer(50, 600),
            "clf__n_estimators":       Integer(100, 500),
            "clf__max_depth":          Categorical([None, 5, 10, 20]),
            "clf__min_samples_split":  Integer(2, 10),
            "clf__min_samples_leaf":   Integer(1, 4),
            "clf__max_features":       Categorical(["sqrt", "log2", None]),
        },

        "XGBoost": {
            "kruskal__k":              Integer(50, 600),
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
    Full training and evaluation pipeline following the Wessels et al. (2005)
    double cross-validation protocol:

    1. Hold out 20% of data as final test set (never used during training)

    2. On remaining 80%:

       OUTER LOOP: RepeatedStratifiedKFold (5-fold x 10 repeats = 50 evaluations)
         INNER LOOP: BayesSearchCV (5-fold CV) for hyperparameter + k optimisation
           - KruskalFilter fit on training fold only (no leakage)
           - StandardScaler fit on training fold only (SVM only)

    3. Best hyperparameters selected by inner CV accuracy

    4. Outer fold test performance averaged across all 50 
    
    5. Final model trained on all 80 training samples
    
    6. Final evaluation on held-out 20%

    Saves:
        output/cv_results.csv       -- per-fold accuracy for all classifiers
        output/cv_summary.csv       -- mean ± std accuracy per classifier
        output/confusion_matrix.csv -- confusion matrix on held-out test set
        model/best_model.pkl        -- best classifier saved with joblib
        model/label_encoder.pkl     -- label encoder for decoding predictions
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

    # 80/20 stratified hold-out split 
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded      # preserves class proportions in both sets
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")

    pipelines     = build_pipelines()
    search_spaces = build_search_spaces()

    # outer CV: 5-fold x 10 repeats (Wessels protocol)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=42)

    # inner CV for BayesSearchCV
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results  = []
    best_score   = 0
    best_name    = None
    best_model   = None
    best_params  = None

    for name, pipeline in pipelines.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(
                outer_cv.split(X_train, y_train)):

            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val   = X_train[val_idx]
            y_fold_val   = y_train[val_idx]

            # inner loop: BayesSearchCV optimises hyperparameters + k
            # fit only on X_fold_train — X_fold_val never seen here
            search = BayesSearchCV(
                estimator   = pipeline,
                search_spaces = search_spaces[name],
                n_iter      = 30,       # number of Bayesian iterations
                cv          = inner_cv,
                scoring     = "accuracy",
                n_jobs      = -1,
                random_state= fold_idx,
                verbose     = 0,
            )
            search.fit(X_fold_train, y_fold_train)

            # evaluate best model from inner loop on outer validation fold
            val_score = accuracy_score(
                y_fold_val, search.predict(X_fold_val))
            fold_scores.append(val_score)

            if (fold_idx + 1) % 10 == 0:
                print(f"  Fold {fold_idx+1}/50 — val accuracy: {val_score:.3f}")

        mean_score = np.mean(fold_scores)
        std_score  = np.std(fold_scores)
        print(f"\n  CV accuracy: {mean_score:.3f} ± {std_score:.3f}")

        # store per-fold results
        for i, score in enumerate(fold_scores):
            all_results.append({
                "classifier": name,
                "fold":       i,
                "accuracy":   score,
            })

        # track best classifier by mean CV score
        if mean_score > best_score:
            best_score  = mean_score
            best_name   = name
            best_model  = search     # BayesSearchCV object with best estimator

    # ave CV results 
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{OUTPUT_DIR}/cv_results.csv", index=False)

    summary = (results_df.groupby("classifier")["accuracy"]
               .agg(["mean", "std"])
               .sort_values("mean", ascending=False))
    summary.to_csv(f"{OUTPUT_DIR}/cv_summary.csv")

    print(f"\n{'='*50}")
    print(f"CV SUMMARY:")
    print(summary.to_string())
    print(f"\nBest classifier: {best_name} ({best_score:.3f})")

    # retrain best classifier on full training set 
    # use best hyperparameters found across all outer folds
    print(f"\nRetraining {best_name} on full training set...")
    final_model = best_model.best_estimator_
    final_model.fit(X_train, y_train)

    # save selected features to a file
    selected_features_idx = final_model.named_steps["kruskal"].selected_
    feature_names = X.columns
    selected_features = feature_names[selected_features_idx]
    pd.Series(selected_features, name="feature").to_csv(
        f"{OUTPUT_DIR}/selected_features.csv", index=False
    )

    # evaluate on held-out test set (done only once) 
    y_pred       = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"FINAL TEST SET ACCURACY: {test_accuracy:.3f}")
    print(f"\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=le.classes_,
                         columns=le.classes_)
    print(f"\nConfusion matrix:")
    print(cm_df)
    cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")

    # save model and label encoder 
    joblib.dump(final_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(le,          f"{MODEL_DIR}/label_encoder.pkl")
    print(f"\nModel saved to {MODEL_DIR}/best_model.pkl")


def main():
    train_and_evaluate()


if __name__ == "__main__":
    main()