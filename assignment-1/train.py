import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import augment, load_data
from models import CNN, MLP

DEFAULTS = {
    "model": "mlp",
    "hidden": [512, 512, 512],
    "channels": [32, 64, 128],
    "kernel_size": 3,
    "pool": "max",
    "batchnorm": False,
    "dropout": 0.0,
    "augment": False,
    "activation": "relu",
    "norm": "unit",
    "optimizer": "adam",
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 50,
    "patience": 8,
    "seed": 42,
    "train_subset": None,
}

MODEL_ONLY_SETTINGS = {
    "mlp": ["hidden"],
    "cnn": ["channels", "kernel_size", "pool", "batchnorm", "dropout"],
}


def train_one_epoch(model, optimizer, x, y, config):
    model.train()
    order = torch.randperm(len(x), device=x.device)
    total_loss = 0.0
    correct = 0
    for start in range(0, len(x), config["batch_size"]):
        idx = order[start:start + config["batch_size"]]
        xb = x[idx]
        yb = y[idx]
        if config["augment"]:
            xb = augment(xb)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
    return total_loss / len(x), correct / len(x)


def evaluate(model, x, y):
    model.eval()
    total_loss = 0.0
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(x), 1000):
            xb = x[start:start + 1000]
            yb = y[start:start + 1000]
            logits = model(xb)
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            all_probs.append(F.softmax(logits, dim=1))
    probs = torch.cat(all_probs)
    accuracy = (probs.argmax(dim=1) == y).float().mean().item()
    return total_loss / len(x), accuracy, probs


def run(config, run_dir, summary_path):
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(
        config["norm"], device, config["train_subset"])

    if config["model"] == "mlp":
        model = MLP(config["hidden"], config["activation"])
    else:
        model = CNN(config["channels"], config["kernel_size"], config["pool"],
                    config["batchnorm"], config["dropout"], config["activation"])
    model = model.to(device)

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    n_params = 0
    for p in model.parameters():
        n_params += p.numel()

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    train_start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, optimizer, x_train, y_train, config)
        val_loss, val_acc, _ = evaluate(model, x_val, y_val)
        epoch_time = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "epoch_time_s": epoch_time})
        print(f"  epoch {epoch:2d}  train loss {train_loss:.4f} acc {train_acc:.2%}  "
              f"val loss {val_loss:.4f} acc {val_acc:.2%}  {epoch_time:.1f} s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config["patience"]:
            print(f"  early stopping: no improvement for {config['patience']} epochs")
            break
    train_time = time.perf_counter() - train_start

    model.load_state_dict(best_weights)
    _, train_acc, _ = evaluate(model, x_train, y_train)
    _, val_acc, _ = evaluate(model, x_val, y_val)
    test_start = time.perf_counter()
    test_loss, test_acc, test_probs = evaluate(model, x_test, y_test)
    inference_time = time.perf_counter() - test_start

    results = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_time_s": train_time,
        "inference_time_s": inference_time,
        "n_params": n_params,
        "device": str(device),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    test_probs = test_probs.cpu().numpy()
    np.save(run_dir / "test_probabilities.npy", test_probs)
    np.save(run_dir / "test_predictions.npy", test_probs.argmax(axis=1))

    row = dict(config)
    for model_type, keys in MODEL_ONLY_SETTINGS.items():
        if model_type != config["model"]:
            row.update({key: "NA" for key in keys})
    row.update(results)
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)
    return results
