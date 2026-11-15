"""Train the Romania mixture of experts and export sealed-record verdicts.

python main.py --data-dir /path/to/dataset --output outputs/predictions.csv
The evaluator and vault are deliberately not imported by the model.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

CHANNELS = [f"c{i:02d}" for i in range(1, 49)]


def weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return values[np.searchsorted(np.cumsum(weights), weights.sum() / 2)]


class VranceaEnsemble:
    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
        if x.ndim != 2 or x.shape[1] != 48 or not np.isfinite(x).all():
            raise ValueError("Expected 48 finite channel values per training record")
        if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
            raise ValueError("Expected one binary label per record")
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.gate = GaussianMixture(
            2, covariance_type="diag", n_init=5, random_state=42
        ).fit(z)
        groups = self.gate.predict(z)
        variances = self.gate.covariances_
        self.small = int(np.argmin(variances.mean(axis=1)))
        self.stable = np.flatnonzero(variances.max(axis=0) / variances.min(axis=0) <= 2)
        if not 0 < len(self.stable) < 48:
            raise ValueError("The expected variance structure was not found")
        low = groups == self.small
        high = ~low
        self.quadratic = make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False),
            StandardScaler(), LogisticRegression(C=1, max_iter=2000),
        ).fit(x[low][:, self.stable], y[low])

        forest = RandomForestClassifier(
            n_estimators=40, max_depth=8, min_samples_leaf=40,
            max_features=1., n_jobs=1, random_state=42,
        ).fit(x[high], y[high])
        self.context_columns = np.argsort(forest.feature_importances_)[-4:]
        thresholds = []
        for column in self.context_columns:
            values, gains = [], []
            for estimator in forest.estimators_:
                tree = estimator.tree_
                for node in np.flatnonzero(tree.feature == column):
                    left, right = tree.children_left[node], tree.children_right[node]
                    count, impurity = tree.weighted_n_node_samples, tree.impurity
                    gain = (count[node] * impurity[node] - count[left] * impurity[left]
                            - count[right] * impurity[right])
                    values.append(tree.threshold[node])
                    gains.append(gain)
            thresholds.append(weighted_median(values, gains))
        self.context_thresholds = np.asarray(thresholds)
        contexts = self.context(x[high])
        self.context_encoder = OneHotEncoder(handle_unknown="ignore").fit(contexts[:, None])
        design = self.linear_design(x[high], z[high])
        # Shared linear slopes plus an intercept for each of the 16 contexts.
        self.linear = LogisticRegression(C=10, max_iter=3000).fit(design, y[high])
        self.group_sizes = [int(low.sum()), int(high.sum())]
        return self

    def context(self, x):
        bits = x[:, self.context_columns] > self.context_thresholds
        return bits.astype(int) @ 2 ** np.arange(4)

    def linear_design(self, x, z):
        one_hot = self.context_encoder.transform(self.context(x)[:, None])
        return sparse.hstack([sparse.csr_matrix(z), one_hot], format="csr")

    def component_probabilities(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 48 or not np.isfinite(x).all():
            raise ValueError("Expected 48 finite channel values per query record")
        z = self.scaler.transform(x)
        weight = self.gate.predict_proba(z)[:, self.small]
        quadratic = self.quadratic.predict_proba(x[:, self.stable])[:, 1]
        linear = self.linear.predict_proba(self.linear_design(x, z))[:, 1]
        return {"small_weight": weight, "quadratic": quadratic,
                "linear": linear}

    def predict_proba(self, x):
        p = self.component_probabilities(x)
        return (p["small_weight"] * p["quadratic"]
                + (1 - p["small_weight"]) * p["linear"])

    def predict(self, x):
        return (self.predict_proba(x) >= .5).astype(int)


def load_training(data_dir):
    frame = pd.read_csv(Path(data_dir) / "train.csv")
    if frame.record_id.duplicated().any():
        raise ValueError("Duplicated training record_id")
    return frame[CHANNELS].to_numpy(), frame.vrancea.to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()
    x, y = load_training(args.data_dir)
    model = VranceaEnsemble().fit(x, y)
    sealed = pd.read_csv(args.data_dir / "sealed.csv")
    if sealed.record_id.duplicated().any():
        raise ValueError("Duplicated sealed record_id")
    predictions = model.predict(sealed[CHANNELS].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"record_id": sealed.record_id, "vrancea": predictions}).to_csv(
        args.output, index=False
    )
    print(f"Group sizes (low/high variance): {model.group_sizes}")
    print("Stable channels:", [CHANNELS[j] for j in model.stable])
    print("Context thresholds:", dict(zip(
        [CHANNELS[j] for j in model.context_columns], model.context_thresholds.tolist()
    )))
    print(f"Wrote {len(predictions)} verdicts to {args.output}")


if __name__ == "__main__":
    main()
