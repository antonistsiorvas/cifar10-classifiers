"""Download and load the CIFAR-10 dataset.

Shared by every assignment in this repo, so they all read the data the same way.

CIFAR-10: 60,000 colour images of size 32x32 in 10 classes.
Official split: 50,000 training images and 10,000 test images.
Source: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "cifar-10-batches-py"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def download(data_dir=DATA_DIR):
    """Download and extract the dataset into data/ if it is not there yet."""
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "cifar-10-batches-py"
    if extracted.exists():
        return extracted

    archive = data_dir / "cifar-10-python.tar.gz"
    if not archive.exists():
        print(f"Downloading CIFAR-10 (~163 MB) from {URL} ...")
        urllib.request.urlretrieve(URL, archive)

    print("Extracting ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir, filter="data")
    return extracted


def _load_batch(path):
    # Each batch file is a pickled dict with b"data" (N x 3072 uint8) and b"labels".
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], np.array(batch[b"labels"], dtype=np.int64)


def load_cifar10(data_dir=DATA_DIR):
    """Return (x_train, y_train, x_test, y_test).

    x_* are uint8 arrays of shape (N, 3072): each row is one image, stored as
    1024 red values, then 1024 green, then 1024 blue (row-major 32x32 each).
    Use to_images() to get them as (N, 32, 32, 3).
    y_* are int64 arrays of shape (N,) with labels 0..9 (see CLASS_NAMES).
    """
    folder = download(data_dir)

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
    """Reshape flat (N, 3072) rows into (N, 32, 32, 3) images for plotting."""
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
