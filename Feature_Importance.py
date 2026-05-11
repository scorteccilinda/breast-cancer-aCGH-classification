import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

# KruskalFilter must be defined for joblib to unpickle the model 
class KruskalFilter(BaseEstimator, TransformerMixin):
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


# load model and data
model   = joblib.load("model/best_model.pkl")
X_train = pd.read_csv("data/x_train.csv", index_col=0)
all_region_names = np.array(X_train.columns)

kruskal = model.named_steps["kruskal"]
xgb     = model.named_steps["clf"]

# Map filtered indices back to original region names 
# KruskalFilter stores indices into the original 2834-region matrix in `selected_`.
# After filtering, XGBoost sees columns 0..k-1, so XGBoost's 'fN' corresponds
# to the original region at all_region_names[selected_[N]].
selected_idx = kruskal.selected_
selected_region_names = all_region_names[selected_idx]
print(f"KruskalFilter kept {len(selected_idx)} / {len(all_region_names)} regions")

# Extract XGBoost feature importance by gain 
# 'gain' = mean improvement in loss when this feature is used in a split.
# More informative than 'weight' (split count) for biological interpretation.
booster   = xgb.get_booster()
gain_dict = booster.get_score(importance_type="gain")

rows = []
for fname, gain in gain_dict.items():
    idx = int(fname.replace("f", ""))
    rows.append({
        "region":         selected_region_names[idx],
        "original_index": int(selected_idx[idx]),
        "gain":           gain,
    })

importance_df = (pd.DataFrame(rows)
                 .sort_values("gain", ascending=False)
                 .reset_index(drop=True))

print(f"XGBoost actually split on {len(importance_df)} of {len(selected_idx)} regions\n")
print("Top 20 regions by gain:")
print(importance_df[["region", "gain"]].head(20).to_string())

importance_df.to_csv("output/feature_importance_xgboost.csv", index=False)

#  annotate with chromosome info from metadata
try:
    meta = pd.read_csv("data/x_train_metadata.csv")
    # adjust column names if yours differ
    importance_df = importance_df.merge(
        meta[["region", "chromosome", "start", "end"]],
        on="region", how="left")
    importance_df.to_csv("output/feature_importance_xgboost_annotated.csv", index=False)
    print("\nAnnotated with chromosome metadata.")
except Exception as e:
    print(f"\n(skipping metadata join: {e})")

# plot top 30
top_n = 30
top   = importance_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(top["region"][::-1], top["gain"][::-1],
        color="steelblue", edgecolor="white")
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title(f"XGBoost — Top {top_n} Genomic Regions by Gain")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig("output/feature_importance_xgboost.png", dpi=200)
plt.show()
print("\nSaved CSV and PNG to output/")