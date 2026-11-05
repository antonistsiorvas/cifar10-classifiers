"""Interim assignment: Nearest Neighbour (k=1, k=3) vs Nearest Class Centroid on CIFAR-10.

Reads the training and test data, classifies every test image with each
classifier, and reports accuracy and running time.

All three classifiers are written from scratch with NumPy (no scikit-learn),
using Euclidean distance on the raw pixel values scaled to [0, 1].

Usage (from the repo root):
    python interim/knn_ncc.py                 # full dataset (50k train / 10k test)
    python interim/knn_ncc.py --subset 1000   # quick run on the first 1000 test images
    python interim/knn_ncc.py --pca 100       # reduce to 100 dimensions with PCA first
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.cifar10 import CLASS_NAMES, load_cifar10  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def squared_distances(a, b, b_sq_norms=None):
    """Squared Euclidean distance between every row of a and every row of b.

    Uses ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, so the heavy part is a single
    matrix multiplication instead of a Python loop. Returns shape (len(a), len(b)).
    Squared distances give the same ordering as real distances, so we skip the sqrt.
    """
    a_sq = np.sum(a * a, axis=1, keepdims=True)          # (n, 1)
    if b_sq_norms is None:
        b_sq_norms = np.sum(b * b, axis=1)
    d = a_sq - 2.0 * (a @ b.T) + b_sq_norms[None, :]      # (n, m)
    np.maximum(d, 0, out=d)  # tiny negatives can appear from floating point error
    return d


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def knn_predict(x_train, y_train, x_test, ks=(1, 3), batch_size=500):
    """k-Nearest Neighbour prediction for several values of k in one pass.

    The distance matrix (10k x 50k) is too big to hold at once, so the test set
    is processed in batches. For each batch we keep the max(ks) nearest training
    samples and reuse them for every k.

    Returns a dict {k: predictions}.
    """
    k_max = max(ks)
    train_sq = np.sum(x_train * x_train, axis=1)
    preds = {k: np.empty(len(x_test), dtype=np.int64) for k in ks}

    for start in range(0, len(x_test), batch_size):
        batch = x_test[start:start + batch_size]
        d = squared_distances(batch, x_train, train_sq)

        # Indices of the k_max closest training samples, sorted nearest first.
        nearest = np.argpartition(d, k_max, axis=1)[:, :k_max]
        rows = np.arange(len(batch))[:, None]
        order = np.argsort(d[rows, nearest], axis=1)
        nearest = nearest[rows, order]
        neighbour_labels = y_train[nearest]               # (batch, k_max)

        for k in ks:
            preds[k][start:start + len(batch)] = [
                majority_vote(labels[:k]) for labels in neighbour_labels
            ]

        done = min(start + batch_size, len(x_test))
        print(f"\r  kNN: {done}/{len(x_test)} test samples", end="", flush=True)
    print()
    return preds


def majority_vote(labels):
    """Most common label among the neighbours (given nearest first).

    Ties (e.g. 3-NN with three different labels) are broken in favour of the
    tied class that has the closest neighbour.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    best = counts.max()
    for label in labels:          # nearest first
        if counts[label] == best:
            return label


def ncc_fit(x_train, y_train):
    """Nearest Class Centroid training: the centroid is the mean of each class."""
    return np.stack([x_train[y_train == c].mean(axis=0) for c in range(NUM_CLASSES)])


def ncc_predict(centroids, x_test):
    """Assign each test sample to the class with the closest centroid."""
    return np.argmin(squared_distances(x_test, centroids), axis=1)


# ---------------------------------------------------------------------------
# PCA (optional feature extraction)
# ---------------------------------------------------------------------------

def pca_fit(x_train, n_components):
    """Return (mean, components) computed from the training data only."""
    mean = x_train.mean(axis=0)
    # Rows of vt are the principal directions, ordered by explained variance.
    _, s, vt = np.linalg.svd(x_train - mean, full_matrices=False)
    explained = (s[:n_components] ** 2).sum() / (s ** 2).sum()
    print(f"  PCA: {n_components} components keep {explained:.1%} of the variance")
    return mean, vt[:n_components]


def pca_transform(x, mean, components):
    return (x - mean) @ components.T


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def per_class_accuracy(y_true, y_pred):
    return {
        CLASS_NAMES[c]: float(np.mean(y_pred[y_true == c] == c))
        for c in range(NUM_CLASSES)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=int, default=None,
                        help="use only the first N test images (for quick runs)")
    parser.add_argument("--pca", type=int, default=None,
                        help="project the data onto N principal components first")
    args = parser.parse_args()

    print("Loading CIFAR-10 ...")
    x_train, y_train, x_test, y_test = load_cifar10()
    if args.subset:
        x_test, y_test = x_test[:args.subset], y_test[:args.subset]

    # Scale pixels from 0..255 to 0..1. float32 halves memory vs float64.
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    if args.pca:
        t0 = time.perf_counter()
        mean, components = pca_fit(x_train, args.pca)
        x_train = pca_transform(x_train, mean, components).astype(np.float32)
        x_test = pca_transform(x_test, mean, components).astype(np.float32)
        print(f"  PCA took {time.perf_counter() - t0:.1f} s")

    results = {}

    print("\nNearest Class Centroid ...")
    t0 = time.perf_counter()
    centroids = ncc_fit(x_train, y_train)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_ncc = ncc_predict(centroids, x_test)
    t_pred = time.perf_counter() - t0
    results["NCC"] = {"accuracy": accuracy(y_test, pred_ncc),
                      "train_time_s": t_fit, "test_time_s": t_pred,
                      "per_class": per_class_accuracy(y_test, pred_ncc)}

    print("\nk-Nearest Neighbour (k = 1 and 3) ...")
    t0 = time.perf_counter()
    knn_preds = knn_predict(x_train, y_train, x_test, ks=(1, 3))
    t_pred = time.perf_counter() - t0
    for k, pred in knn_preds.items():
        # kNN has no training step: it just stores the training set.
        # Both k values share one distance computation, so they share the time.
        results[f"{k}-NN"] = {"accuracy": accuracy(y_test, pred),
                              "train_time_s": 0.0, "test_time_s": t_pred,
                              "per_class": per_class_accuracy(y_test, pred)}

    # ---- Report ----
    print("\n" + "=" * 52)
    print(f"{'Classifier':<12}{'Accuracy':>12}{'Train (s)':>13}{'Test (s)':>13}")
    print("-" * 52)
    for name in ("1-NN", "3-NN", "NCC"):
        r = results[name]
        print(f"{name:<12}{r['accuracy']:>11.2%}{r['train_time_s']:>13.2f}{r['test_time_s']:>13.2f}")
    print("=" * 52)
    print("(1-NN and 3-NN share one distance computation, so their test time is the same.)")

    print("\nPer-class accuracy:")
    print(f"{'class':<12}" + "".join(f"{n:>9}" for n in ("1-NN", "3-NN", "NCC")))
    for c in CLASS_NAMES:
        print(f"{c:<12}" + "".join(f"{results[n]['per_class'][c]:>9.1%}"
                                   for n in ("1-NN", "3-NN", "NCC")))

    # ---- Save ----
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "full" if not args.subset else f"subset{args.subset}"
    if args.pca:
        tag += f"_pca{args.pca}"
    out = RESULTS_DIR / f"results_{tag}.json"
    with open(out, "w") as f:
        json.dump({"settings": {"n_train": len(x_train), "n_test": len(x_test),
                                "pca": args.pca},
                   "results": results}, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
