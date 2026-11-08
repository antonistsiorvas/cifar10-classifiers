import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _load_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], np.array(batch[b"labels"], dtype=np.int64)


def load_cifar10(data_dir=DATA_DIR):
    folder = data_dir / "cifar-10-batches-py"
    if not folder.exists():
        raise FileNotFoundError(
            f"CIFAR-10 not found in {folder}. Download the python version from "
            "https://www.cs.toronto.edu/~kriz/cifar.html, extract it and place "
            "the cifar-10-batches-py/ folder in data/."
        )

    xs, ys = [], []
    for i in range(1, 6):
        x, y = _load_batch(folder / f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    x_test, y_test = _load_batch(folder / "test_batch")
    return x_train, y_train, x_test, y_test


def to_images(x):
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
