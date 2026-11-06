"""Interim assignment: Nearest Neighbour (k=1, k=3) vs Nearest Class Centroid on CIFAR-10.

Reads the training and test data, classifies every test image with each
classifier, and reports accuracy and running time.

The classifiers come from scikit-learn and use Euclidean distance on the raw
pixel values scaled to [0, 1].

Usage (from the repo root):
    python interim/knn_ncc.py                 # full dataset (50k train / 10k test)
    python interim/knn_ncc.py --subset 1000   # quick run on the first 1000 test images
    python interim/knn_ncc.py --pca 100       # reduce to 100 dimensions with PCA first
    python interim/knn_ncc.py --examples      # also save example predictions (all settings)
    python interim/knn_ncc.py --per-class     # also save per-class accuracy (all settings)
"""

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.cifar10 import DATA_DIR, load_cifar10, to_images  # noqa: E402


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


# --- All settings: example predictions (--examples), per-class accuracy (--per-class)

EXAMPLES_DIR = Path(__file__).resolve().parent / "results" / "examples"
PER_CLASS_CSV = Path(__file__).resolve().parent / "results" / "per_class_accuracy.csv"
PCA_SIZES = [10, 20, 50, 100, 200, 500]
# (title name, file name, classifier) for the three classifiers.
CLASSIFIERS = [
    ("1-NN", "1nn", KNeighborsClassifier(n_neighbors=1)),
    ("3-NN", "3nn", KNeighborsClassifier(n_neighbors=3)),
    ("NCC", "ncc", NearestCentroid()),
]


def load_class_names():
    """Read the class names from batches.meta (stored as bytes, e.g. b"cat")."""
    with open(DATA_DIR / "cifar-10-batches-py" / "batches.meta", "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [name.decode() for name in meta[b"label_names"]]


def save_examples(name, setting, filename, images, y_test, y_pred, class_names, rows):
    """Save 2 correct + 2 wrong test images in one figure and add them to rows."""
    rng = np.random.default_rng(42)  # same seed for every figure -> reproducible
    correct = np.flatnonzero(y_pred == y_test)
    wrong = np.flatnonzero(y_pred != y_test)
    picks = list(rng.choice(correct, 2, replace=False)) + list(rng.choice(wrong, 2, replace=False))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, i in zip(axes, picks):
        ok = y_pred[i] == y_test[i]
        ax.imshow(images[i], interpolation="nearest")  # nearest: sharp enlarged pixels
        ax.set_title(f"true: {class_names[y_test[i]]} / pred: {class_names[y_pred[i]]}",
                     color="green" if ok else "red", fontsize=10)
        ax.axis("off")
        rows.append([name, setting, i,
                     class_names[y_test[i]], class_names[y_pred[i]], "yes" if ok else "no"])
    fig.suptitle(f"{name}, {setting}", fontsize=14)  # e.g. "3-NN, PCA 50"
    fig.savefig(EXAMPLES_DIR / f"{filename}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_all_settings(x_train, y_train, x_test, y_test, examples, per_class):
    """Run every classifier on every setting (raw, PCA k) and save the requested
    outputs: example figures (--examples) and/or per-class accuracy (--per-class)."""
    class_names = load_class_names()
    images = to_images(x_test)  # (N, 3072) -> (N, 32, 32, 3), already in [0, 1]
    example_rows = []
    per_class_columns = {}  # "1-NN raw" -> list of 10 per-class accuracies
    if examples:
        EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for k in [None] + PCA_SIZES:
        # Transform the data once per setting; all three classifiers reuse it.
        if k is None:
            xtr, xte, setting, tag = x_train, x_test, "raw", "raw"
        else:
            pca = PCA(n_components=k, random_state=42)  # fit on training data only
            xtr, xte = pca.fit_transform(x_train), pca.transform(x_test)
            setting, tag = f"PCA {k}", f"pca{k}"

        for name, file_name, clf in CLASSIFIERS:
            y_pred = clf.fit(xtr, y_train).predict(xte)  # predict once, use for both

            if examples:
                save_examples(name, setting, f"{file_name}_{tag}",
                              images, y_test, y_pred, class_names, example_rows)
                print(f"saved {file_name}_{tag}.png")

            if per_class:
                # Accuracy of class c = fraction of test images of class c predicted as c.
                accs = [np.mean(y_pred[y_test == c] == c) for c in range(len(class_names))]
                per_class_columns[f"{name} {setting}"] = accs
                print(f"{name:<4} {setting:<7} per class: "
                      + " ".join(f"{a:.1%}" for a in accs))

    if examples:
        with open(EXAMPLES_DIR / "examples_index.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["classifier", "setting", "test_index",
                             "true_label", "predicted_label", "correct"])
            writer.writerows(example_rows)

    if per_class:
        # One row per class, one column per classifier + setting (values in %).
        with open(PER_CLASS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class"] + list(per_class_columns))
            for c, class_name in enumerate(class_names):
                writer.writerow([class_name] + [f"{100 * accs[c]:.2f}"
                                                for accs in per_class_columns.values()])
        print(f"saved {PER_CLASS_CSV.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, help="use only the first N test images")
    parser.add_argument("--pca", type=int, help="reduce the data to N dimensions with PCA")
    parser.add_argument("--examples", action="store_true",
                        help="save example predictions for every classifier and setting")
    parser.add_argument("--per-class", action="store_true",
                        help="save per-class accuracy for every classifier and setting")
    args = parser.parse_args()

    # 1. Load the data and scale pixels from 0..255 to 0..1.
    x_train, y_train, x_test, y_test = load_cifar10()
    if args.subset:
        x_test, y_test = x_test[:args.subset], y_test[:args.subset]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print(f"Train: {x_train.shape}, test: {x_test.shape}")
    x_train_raw, x_test_raw = x_train, x_test  # kept for --examples / --per-class (before PCA)

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

    # 4. Optional: example predictions and/or per-class accuracy for all settings.
    if args.examples or args.per_class:
        run_all_settings(x_train_raw, y_train, x_test_raw, y_test,
                         args.examples, args.per_class)


if __name__ == "__main__":
    main()
