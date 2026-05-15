#!/usr/bin/env python3
"""Reproduce your result by your saved model.

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


# This class needs to be defined here so joblib can load the saved model.
# It is the same KruskalFilter used during training.
class KruskalFilter(BaseEstimator, TransformerMixin):

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


def main():
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True, dest='output_file',
                        metavar='output.txt', type=str,
                        help='Path of the output file')

    args = parser.parse_args()

    # load the saved hierarchical model
    saved = joblib.load(args.model_file)
    model_s1 = saved["stage1"]
    model_s2 = saved["stage2"]
    chr17_cols = saved["chr17_cols"]
    non_chr17_cols = saved["non_chr17_cols"]

    # load and prepare the input data
    meta_cols = ["Start", "End", "Nclone"]
    call_data = pd.read_csv(args.input_file, sep="\t", index_col=0)

    region_names = [
        f"chr{c}_{int(s)}_{int(e)}"
        for c, s, e in zip(call_data.index, call_data["Start"], call_data["End"])
    ]
    call_data.index = region_names
    call_data = call_data.drop(columns=[c for c in meta_cols if c in call_data.columns])
    X = call_data.T

    # split features by chromosome for the two stages
    X_s1 = X[chr17_cols].values
    X_s2 = X[non_chr17_cols].values

    # stage 1: separate HER2+ from the rest using chr17 features
    s1_preds = model_s1.predict(X_s1)
    non_her2 = s1_preds == 0

    # stage 2: classify HR+ vs Triple Neg for non-HER2+ samples
    s2_preds_num = model_s2.predict(X_s2[non_her2])
    s2_preds = np.where(s2_preds_num == 0, "HR+", "Triple Neg")

    # combine predictions from both stages
    final_preds = np.where(s1_preds == 1, "HER2+", "")
    final_preds[non_her2] = s2_preds

    # write output file
    with open(args.output_file, "w") as f:
        f.write('"Sample"\t"Subgroup"\n')
        for sample, pred in zip(X.index, final_preds):
            f.write(f'"{sample}"\t"{pred}"\n')

    print(f"Predictions written to {args.output_file}")


if __name__ == '__main__':
    main()
