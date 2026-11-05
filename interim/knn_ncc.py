"""Interim assignment: Nearest Neighbour (k=1, k=3) vs Nearest Class Centroid on CIFAR-10.

Reads the training and test data, classifies every test image with each
classifier, and reports accuracy and running time.

The classifiers come from scikit-learn and use Euclidean distance on the raw
pixel values scaled to [0, 1].

Usage (from the repo root):
    python interim/knn_ncc.py                 # full dataset (50k train / 10k test)
    python interim/knn_ncc.py --subset 1000   # quick run on the first 1000 test images
    python interim/knn_ncc.py --pca 100       # reduce to 100 dimensions with PCA first
"""

import argparse
import sys
import time
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.cifar10 import load_cifar10  # noqa: E402


def evaluate(name, classifier, x_train, y_train, x_test, y_test):
    """Train the classifier, test it, and print accuracy and timings."""
    start = time.perf_counter()
    classifier.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    accuracy = classifier.score(x_test, y_test)  # fraction of correct predictions
    test_time = time.perf_counter() - start

    print(f"{name:<6} accuracy: {accuracy:.2%}   "
          f"train: {train_time:.2f} s   test: {test_time:.2f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, help="use only the first N test images")
    parser.add_argument("--pca", type=int, help="reduce the data to N dimensions with PCA")
    args = parser.parse_args()

    # 1. Load the data and scale pixels from 0..255 to 0..1.
    x_train, y_train, x_test, y_test = load_cifar10()
    if args.subset:
        x_test, y_test = x_test[:args.subset], y_test[:args.subset]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print(f"Train: {x_train.shape}, test: {x_test.shape}")

    # 2. Optional: PCA, learned from the training data only.
    if args.pca:
        pca = PCA(n_components=args.pca)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        print(f"PCA keeps {pca.explained_variance_ratio_.sum():.1%} of the variance")

    # 3. Run the three classifiers.
    evaluate("1-NN", KNeighborsClassifier(n_neighbors=1), x_train, y_train, x_test, y_test)
    evaluate("3-NN", KNeighborsClassifier(n_neighbors=3), x_train, y_train, x_test, y_test)
    evaluate("NCC", NearestCentroid(), x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
