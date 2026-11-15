import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.cifar10 import download_cifar10, load_cifar10

SEED = 42  # fixed, so every run uses the same train/validation split


def train_val_split(y):
    """Pick 500 random images of every class for validation; the rest is training."""
    rng = np.random.default_rng(SEED)
    val_idx = []
    for c in range(10):
        class_idx = np.flatnonzero(y == c)
        val_idx.extend(rng.choice(class_idx, 500, replace=False))
    val_idx = np.sort(val_idx)
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    return train_idx, val_idx


def to_tensor(x, norm, device):
    """Rows of 3072 pixels -> float tensor N x 3 x 32 x 32 on the device, normalized."""
    x = x.reshape(-1, 3, 32, 32).astype(np.float32)
    if norm == "unit":
        x = x / 255.0
    return torch.from_numpy(x).to(device)


def load_data(norm, device, train_subset=None):
    """Load CIFAR-10 and return train, validation and test tensors."""
    download_cifar10()
    x_all, y_all, x_test, y_test = load_cifar10()

    train_idx, val_idx = train_val_split(y_all)
    if train_subset is not None:
        # Smoke test: keep only a small random part of the training split
        rng = np.random.default_rng(SEED)
        train_idx = np.sort(rng.choice(train_idx, train_subset, replace=False))

    x_train = to_tensor(x_all[train_idx], norm, device)
    x_val = to_tensor(x_all[val_idx], norm, device)
    x_test = to_tensor(x_test, norm, device)
    y_train = torch.from_numpy(y_all[train_idx]).to(device)
    y_val = torch.from_numpy(y_all[val_idx]).to(device)
    y_test = torch.from_numpy(y_test).to(device)
    return x_train, y_train, x_val, y_val, x_test, y_test


def augment(x):
    """Random 32x32 crop (after 4 pixels of zero padding) and random horizontal flip, per image."""
    padded = F.pad(x, (4, 4, 4, 4))  # N x 3 x 40 x 40
    out = torch.empty_like(x)
    for i in range(len(x)):
        top = random.randint(0, 8)
        left = random.randint(0, 8)
        image = padded[i, :, top:top + 32, left:left + 32]
        if random.random() < 0.5:
            image = image.flip(2)  # dimension 2 is the width
        out[i] = image
    return out
