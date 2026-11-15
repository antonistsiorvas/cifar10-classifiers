import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _load_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], np.array(batch[b"labels"], dtype=np.int64)


def download_cifar10(data_dir=DATA_DIR):
    # Download and extract the python version of CIFAR-10, unless it is already there
    if (data_dir / "cifar-10-batches-py").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "cifar-10-python.tar.gz"
    if not archive.exists():
        print(f"Downloading {CIFAR10_URL} ...")
        urllib.request.urlretrieve(CIFAR10_URL, archive)
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir, filter="data")


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
