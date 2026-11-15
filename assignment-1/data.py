import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.cifar10 import download_cifar10, load_cifar10

NUM_CLASSES = 10
VAL_PER_CLASS = 500
SPLIT_SEED = 42  # fixed, so every run uses the same train/validation split


def train_val_split(y, val_per_class=VAL_PER_CLASS, seed=SPLIT_SEED):
    # Stratified split: pick val_per_class random images of every class for validation
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in range(NUM_CLASSES):
        class_idx = np.flatnonzero(y == c)
        val_idx.extend(rng.choice(class_idx, val_per_class, replace=False))
    val_idx = np.sort(np.array(val_idx))
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    return train_idx, val_idx


def normalize(x_train, x_val, x_test, norm):
    # x_* are float32 arrays of shape N x 3 x 32 x 32 with values 0-255
    if norm == "raw":
        return x_train, x_val, x_test
    if norm == "unit":
        return x_train / 255.0, x_val / 255.0, x_test / 255.0
    raise ValueError(f"unknown normalization: {norm}")


def load_data(norm="unit", device="cpu", train_subset=None):
    download_cifar10()
    x_train_full, y_train_full, x_test, y_test = load_cifar10()

    train_idx, val_idx = train_val_split(y_train_full)
    if train_subset:
        # Smoke tests: train on a random subset of the training split
        rng = np.random.default_rng(SPLIT_SEED)
        train_idx = np.sort(rng.choice(train_idx, train_subset, replace=False))

    # Rows of 3072 values (R, G, B planes) -> images N x 3 x 32 x 32
    def to_tensor_shape(x):
        return x.reshape(-1, 3, 32, 32).astype(np.float32)

    x_train = to_tensor_shape(x_train_full[train_idx])
    x_val = to_tensor_shape(x_train_full[val_idx])
    x_test = to_tensor_shape(x_test)
    x_train, x_val, x_test = normalize(x_train, x_val, x_test, norm)

    # Keep the whole dataset as tensors on the device
    def to_device(x):
        return torch.from_numpy(x).to(device)

    return {
        "x_train": to_device(x_train), "y_train": to_device(y_train_full[train_idx]),
        "x_val": to_device(x_val), "y_val": to_device(y_train_full[val_idx]),
        "x_test": to_device(x_test), "y_test": to_device(y_test),
    }


def batches(x, y, batch_size, shuffle):
    # Yield (x, y) mini-batches; with shuffle, a new random order every call (= every epoch)
    n = len(x)
    if shuffle:
        order = torch.randperm(n, device=x.device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            yield x[idx], y[idx]
    else:
        for start in range(0, n, batch_size):
            yield x[start:start + batch_size], y[start:start + batch_size]


def augment(x, padding=4):
    # Random horizontal flip + random 32x32 crop from the zero-padded image, per image
    n, c, h, w = x.shape

    # Flip half of the images left-right
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)

    # Pad with zeros to 40x40, then choose a random top-left corner for each image
    padded = F.pad(x, (padding, padding, padding, padding))
    top = torch.randint(0, 2 * padding + 1, (n,), device=x.device)
    left = torch.randint(0, 2 * padding + 1, (n,), device=x.device)

    # rows[i] = the 32 row indices of the crop of image i, same for cols
    rows = top[:, None] + torch.arange(h, device=x.device)
    cols = left[:, None] + torch.arange(w, device=x.device)
    # Index the padded batch with broadcasting: result[i, ch, r, q] = padded[i, ch, rows[i, r], cols[i, q]]
    image_idx = torch.arange(n, device=x.device)[:, None, None, None]
    channel_idx = torch.arange(c, device=x.device)[None, :, None, None]
    return padded[image_idx, channel_idx, rows[:, None, :, None], cols[:, None, None, :]]
