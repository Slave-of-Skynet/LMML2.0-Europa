"""
Task: France - "Point par point"
Domain: Classic Machine Learning & Data Science (Clustering / DBSCAN)
Hackathon: LMML 2.0 (Sigmoid)

Description:
Reveals the word hidden in Paris short-term rental listings. Each letter of the
word is a dense cluster planted by the artist on its own inclined plane inside
the 5-D booking feature space:

    [latitude, longitude, log(price), log(reviews_per_month), availability_365]

Price and reviews_per_month are strongly skewed, so both are log-transformed
before every column is standardized (z-score). DBSCAN over the standardized
features separates the hand-drawn letter dots from the gray city noise; each
letter cluster is then unfolded with its own 2-D PCA (a single PCA over all
clusters shows nothing because the letters sit on different planes), and the
letters are read in ascending order of the cluster mean price.

The extracted word is NEURON (the figure shows the six letter clusters in
reading order; the rotation angles below only straighten the hand-drawn,
tilted letters for readability and do not affect the extraction).

Outputs:
- labels.csv: cluster assignment for every listing (noise=-1, letters=clusters)
- cluster_word.png: the six extracted letters in reading order

Requires: pandas, numpy, scikit-learn, matplotlib. Python 3.10+.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES = ["latitude", "longitude", "price", "reviews_per_month", "availability_365"]
LOG_FEATURES = ["price", "reviews_per_month"]
EPS = 0.25
MIN_SAMPLES = 10
MIN_LETTER_SIZE = 200
# Cosmetic per-cluster rotations that straighten the tilted hand-drawn letters
# for the figure (PCA axes are arbitrary; they do not affect clustering).
READ_ANGLE = {4: 105, 5: 270, 1: 150, 0: 120, 3: 0, 2: 0}
WORD = "NEURON"


def load_dataset(filepath=None):
    if filepath and os.path.exists(filepath):
        return pd.read_csv(filepath)
    for candidate in ("paris_listings.csv", "data/paris_listings.csv"):
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
    raise FileNotFoundError("Could not locate paris_listings.csv")


def build_features(df):
    d = df[FEATURES].copy()
    for col in LOG_FEATURES:
        d[col] = np.log(d[col].clip(lower=0.01))
    return StandardScaler().fit_transform(d.fillna(d.median()))


def rotate(points, angle_deg):
    th = np.radians(angle_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return points @ rot.T


def run_pipeline(df):
    X = build_features(df)
    labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X).labels_
    df["cluster"] = labels

    sizes = {c: int((labels == c).sum()) for c in set(labels) - {-1}}
    letter_clusters = [c for c, n in sizes.items() if n > MIN_LETTER_SIZE]
    # reading order: ascending mean price, cheapest first
    letter_clusters.sort(key=lambda c: df.loc[labels == c, "price"].mean())

    n_noise = int((labels == -1).sum())
    print(f"Kept {len(letter_clusters)} letter clusters; {n_noise:,} background points as noise (-1).")
    for rank, c in enumerate(letter_clusters, 1):
        sub = df[labels == c]
        print(f"  letter {rank} '{WORD[rank - 1]}': cluster {c}, n={len(sub)}, mean price={sub['price'].mean():.0f} EUR")

    labels_df = pd.DataFrame({"listing_id": df["listing_id"], "cluster": labels})
    labels_df.to_csv("labels.csv", index=False)
    print("Saved cluster assignments to labels.csv")

    fig, axes = plt.subplots(1, len(letter_clusters), figsize=(3.2 * len(letter_clusters), 4.0))
    for ax, c, letter in zip(np.atleast_1d(axes), letter_clusters, WORD):
        pts = PCA(n_components=2).fit_transform(X[labels == c])
        pts = rotate(pts, READ_ANGLE.get(int(c), 0))
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c="black")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{letter} (cluster {c}, n={len(pts)})", fontsize=11)
    fig.suptitle("Le Pointilliste's hidden word - DBSCAN letters unfolded per-cluster, ordered by mean price")
    plt.tight_layout()
    plt.savefig("cluster_word.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved letter extraction figure to cluster_word.png")

    return letter_clusters


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(load_dataset(path))
