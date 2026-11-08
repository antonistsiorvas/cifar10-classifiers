import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.cifar10 import CLASS_NAMES, load_cifar10, to_images


def evaluate(name, classifier, x_train, y_train, x_test, y_test):
    start = time.perf_counter()
    classifier.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    accuracy = classifier.score(x_test, y_test)
    test_time = time.perf_counter() - start

    print(f"{name:<6} accuracy: {accuracy:.2%}   "
          f"train: {train_time:.2f} s   test: {test_time:.2f} s")


EXAMPLES_DIR = Path(__file__).resolve().parent / "results" / "examples"
PER_CLASS_CSV = Path(__file__).resolve().parent / "results" / "per_class_accuracy.csv"
PCA_SIZES = [10, 20, 50, 100, 200, 500]
CLASSIFIERS = [
    ("1-NN", "1nn", KNeighborsClassifier(n_neighbors=1)),
    ("3-NN", "3nn", KNeighborsClassifier(n_neighbors=3)),
    ("NCC", "ncc", NearestCentroid()),
]


def save_examples(name, setting, filename, images, y_test, y_pred, class_names, rows):
    rng = np.random.default_rng(42)
    correct = np.flatnonzero(y_pred == y_test)
    wrong = np.flatnonzero(y_pred != y_test)
    picks = list(rng.choice(correct, 2, replace=False)) + list(rng.choice(wrong, 2, replace=False))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, i in zip(axes, picks):
        ok = y_pred[i] == y_test[i]
        ax.imshow(images[i], interpolation="nearest")
        ax.set_title(f"true: {class_names[y_test[i]]} / pred: {class_names[y_pred[i]]}",
                     color="green" if ok else "red", fontsize=10)
        ax.axis("off")
        rows.append([name, setting, i,
                     class_names[y_test[i]], class_names[y_pred[i]], "yes" if ok else "no"])
    fig.suptitle(f"{name}, {setting}", fontsize=14)
    fig.savefig(EXAMPLES_DIR / f"{filename}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_all_settings(x_train, y_train, x_test, y_test, examples, per_class):
    class_names = CLASS_NAMES
    images = to_images(x_test)
    example_rows = []
    per_class_columns = {}
    if examples:
        EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for k in [None] + PCA_SIZES:
        if k is None:
            xtr, xte, setting, tag = x_train, x_test, "raw", "raw"
        else:
            pca = PCA(n_components=k, random_state=42)
            xtr, xte = pca.fit_transform(x_train), pca.transform(x_test)
            setting, tag = f"PCA {k}", f"pca{k}"

        for name, file_name, clf in CLASSIFIERS:
            y_pred = clf.fit(xtr, y_train).predict(xte)

            if examples:
                save_examples(name, setting, f"{file_name}_{tag}",
                              images, y_test, y_pred, class_names, example_rows)
                print(f"saved {file_name}_{tag}.png")

            if per_class:
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

    x_train, y_train, x_test, y_test = load_cifar10()
    if args.subset:
        x_test, y_test = x_test[:args.subset], y_test[:args.subset]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print(f"Train: {x_train.shape}, test: {x_test.shape}")
    x_train_raw, x_test_raw = x_train, x_test

    if args.pca:
        pca = PCA(n_components=args.pca, random_state=42)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    evaluate("1-NN", KNeighborsClassifier(n_neighbors=1), x_train, y_train, x_test, y_test)
    evaluate("3-NN", KNeighborsClassifier(n_neighbors=3), x_train, y_train, x_test, y_test)
    evaluate("NCC", NearestCentroid(), x_train, y_train, x_test, y_test)

    if args.examples or args.per_class:
        run_all_settings(x_train_raw, y_train, x_test_raw, y_test,
                         args.examples, args.per_class)


if __name__ == "__main__":
    main()
