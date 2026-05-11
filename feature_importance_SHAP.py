import shap
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# KruskalFilter definition is required for joblib to load the model
class KruskalFilter(BaseEstimator, TransformerMixin):
    def __init__(self, k=100): self.k = k
    def fit(self, X, y):
        subtypes = np.unique(y); h_scores = []
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
    def transform(self, X): return X[:, self.selected_]


# load the trained model and training data
model   = joblib.load("model/best_model.pkl")
le      = joblib.load("model/label_encoder.pkl")
X_train = pd.read_csv("data/x_train.csv", index_col=0)

# apply the KruskalFilter so we have the same matrix XGBoost saw
kruskal = model.named_steps["kruskal"]
xgb     = model.named_steps["clf"]
X_filtered   = kruskal.transform(X_train.values)
region_names = X_train.columns[kruskal.selected_].to_numpy()

# compute SHAP values (TreeExplainer is fast and exact for XGBoost)
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_filtered)

# standardise output shape to (n_samples, n_features, n_classes)
shap_arr = np.array(shap_values)
if shap_arr.shape[0] == len(le.classes_):
    shap_arr = np.transpose(shap_arr, (1, 2, 0))

# Global importance: mean(|SHAP|) over samples and classes
mean_abs = np.abs(shap_arr).mean(axis=(0, 2))
importance_df = (pd.DataFrame({"region": region_names, "mean_abs_shap": mean_abs})
                 .sort_values("mean_abs_shap", ascending=False)
                 .reset_index(drop=True))

importance_df.to_csv("output/feature_importance_shap.csv", index=False)
print("Top 20 regions by SHAP:")
print(importance_df.head(20).to_string())

cn_cmap = ListedColormap(["#2166AC", "#D9D9D9", "#F4A582", "#B2182B"])

legend_elements = [
    mpatches.Patch(color="#2166AC", label="−1"),
    mpatches.Patch(color="#D9D9D9", label="0"),
    mpatches.Patch(color="#F4A582", label="1"),
    mpatches.Patch(color="#B2182B", label="2"),
]

for c, cls in enumerate(le.classes_):
    shap.summary_plot(
        shap_arr[:, :, c],
        X_filtered,
        feature_names=region_names,
        max_display=15,
        cmap=cn_cmap,
        color_bar=False,        # turn off SHAP's broken continuous colorbar
        show=False,
    )
    plt.legend(
        handles=legend_elements,
        title="Call value",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    plt.title(f"SHAP — {cls}", fontsize=14)
    plt.tight_layout()
    safe = cls.replace("+", "pos").replace(" ", "_")
    plt.savefig(f"output/shap_{safe}.png", dpi=200, bbox_inches="tight")
    plt.close()