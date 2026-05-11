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
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats


# KruskalFilter must be defined here so joblib.load can resolve the class eference inside the pickled pipeline. Do not remove.
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


# LabelEncoder is alphabetical on these labels, so this mapping is fixed.
INT_TO_LABEL = {0: "HER2+", 1: "HR+", 2: "Triple Neg"}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')
    if args.model_file is None:
        sys.exit('Model file is missing!')
    if args.output_file is None:
        sys.exit('Output is not designated!')

    # loads the trained pipeline (KruskalFilter + XGBoost)
    model = joblib.load(args.model_file)

    # read the input file.
    # Train_call.txt format: tab-separated, regions as rows, samples as columns.
    # First 4 columns are region metadata (Chromosome, Start, End, Nclone).
    df = pd.read_csv(args.input_file, sep="\t")
    sample_columns = df.columns[4:]
    X = df[sample_columns].T.values   # shape: (n_samples, n_regions)

    # predict and decode numeric labels to subtype names
    preds = model.predict(X)
    labels = [INT_TO_LABEL[int(p)] for p in preds]

    # writes output in the required format. Header must be exactly: "Sample"\t"Subgroup"
    with open(args.output_file, 'w') as f:
        f.write('"Sample"\t"Subgroup"\n')
        for sample, label in zip(sample_columns, labels):
            f.write(f'"{sample}"\t"{label}"\n')


if __name__ == '__main__':
    main()