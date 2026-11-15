import argparse
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
from data import augment, batches, load_data
from models import CNN, MLP, count_parameters

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Default configuration; every run overrides only what it changes
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
    "name": None,
    "experiment": "",
    "out_dir": str(RESULTS_DIR),
    "save_model": False,
    "verbose": True,
}

SUMMARY_FIELDS = [
    "experiment", "run_name", "model", "hidden", "channels", "kernel_size", "pool",
    "batchnorm", "dropout", "augment", "activation", "norm", "optimizer", "lr",
    "batch_size", "train_acc", "val_acc", "test_acc", "best_epoch", "epochs_run",
    "train_time_s", "inference_time_s", "n_params", "device",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def build_model(config):
    if config["model"] == "mlp":
        return MLP(config["hidden"], config["activation"])
    return CNN(config["channels"], config["kernel_size"], config["pool"],
               config["batchnorm"], config["dropout"], config["activation"])


def build_optimizer(config, model):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    return torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)


def default_name(config):
    if config["model"] == "mlp":
        arch = "h" + "-".join(map(str, config["hidden"]))
    else:
        arch = "c" + "-".join(map(str, config["channels"])) + f"_k{config['kernel_size']}"
    return f"{config['model']}_{arch}_{config['activation']}_{config['norm']}_{config['optimizer']}_lr{config['lr']:g}"


def sync(device):
    # GPU work is asynchronous: wait for it to finish before reading the clock
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(model, optimizer, x, y, config):
    model.train()
    total_loss = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    for xb, yb in batches(x, y, config["batch_size"], shuffle=True):
        if config["augment"]:
            xb = augment(xb)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()   # back-propagation
        optimizer.step()  # gradient step

        # Running totals, kept on the device to avoid a GPU->CPU copy per batch
        total_loss += loss.detach() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum()
    return total_loss.item() / len(x), correct.item() / len(x)


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    total_loss, correct, predictions = 0.0, 0, []
    for xb, yb in batches(x, y, batch_size, shuffle=False):
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        predictions.append(pred)
    return total_loss / len(x), correct / len(x), torch.cat(predictions)


def save_summary_row(path, row):
    # Rewrite summary.csv with this run's row replacing an older row of the same run
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run_name"] != row["run_name"]]
    rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(config):
    config = {**DEFAULTS, **config}
    if not config["name"]:
        config["name"] = default_name(config)
    out_dir = Path(config["out_dir"])
    run_dir = out_dir / "runs" / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(config["norm"], device, config["train_subset"])
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    n_params = count_parameters(model)
    if config["verbose"]:
        print(f"{config['name']}: {n_params:,} parameters, {len(x_train):,} training images, device {device}")

    history = []
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    epochs_without_improvement = 0

    sync(device)
    train_start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, optimizer, x_train, y_train, config)
        val_loss, val_acc, _ = evaluate(model, x_val, y_val)
        sync(device)
        epoch_time = time.perf_counter() - epoch_start

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "epoch_time_s": epoch_time})
        if config["verbose"]:
            print(f"  epoch {epoch:3d}  train loss {train_loss:.4f} acc {train_acc:.2%}   "
                  f"val loss {val_loss:.4f} acc {val_acc:.2%}   {epoch_time:.1f} s")

        # Early stopping on validation loss: keep a copy of the best weights
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                if config["verbose"]:
                    print(f"  early stopping: no improvement for {config['patience']} epochs")
                break
    sync(device)
    train_time = time.perf_counter() - train_start

    # Final evaluation with the best weights
    model.load_state_dict(best_state)
    train_loss, train_acc, _ = evaluate(model, x_train, y_train)
    val_loss, val_acc, _ = evaluate(model, x_val, y_val)
    sync(device)
    test_start = time.perf_counter()
    test_loss, test_acc, test_pred = evaluate(model, x_test, y_test)
    sync(device)
    inference_time = time.perf_counter() - test_start

    metrics = {
        "config": config,
        "train_acc": train_acc, "train_loss": train_loss,
        "val_acc": val_acc, "val_loss": val_loss,
        "test_acc": test_acc, "test_loss": test_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_time_s": train_time,
        "time_per_epoch_s": train_time / len(history),
        "inference_time_s": inference_time,
        "n_params": n_params,
        "n_train": len(x_train),
        "device": str(device),
    }

    # Save the results of this run
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.save(run_dir / "test_predictions.npy", test_pred.cpu().numpy())
    if config["save_model"]:
        torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "metrics.json", "w") as f:  # written last: marks the run as finished
        json.dump(metrics, f, indent=2)

    row = {k: config.get(k) for k in SUMMARY_FIELDS}
    row["run_name"] = config["name"]
    row["hidden"] = "-".join(map(str, config["hidden"])) if config["model"] == "mlp" else ""
    row["channels"] = "-".join(map(str, config["channels"])) if config["model"] == "cnn" else ""
    for k in ["train_acc", "val_acc", "test_acc", "best_epoch", "epochs_run",
              "n_params", "device"]:
        row[k] = metrics[k]
    row["train_time_s"] = round(train_time, 2)
    row["inference_time_s"] = round(inference_time, 4)
    save_summary_row(out_dir / "summary.csv", row)

    if config["verbose"]:
        print(f"  best epoch {best_epoch}: train {train_acc:.2%}  val {val_acc:.2%}  "
              f"test {test_acc:.2%}   training {train_time:.1f} s, test inference {inference_time:.3f} s")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate one MLP or CNN on CIFAR-10")
    parser.add_argument("--model", choices=["mlp", "cnn"], default=DEFAULTS["model"])
    parser.add_argument("--hidden", type=int, nargs="+", default=DEFAULTS["hidden"],
                        help="MLP hidden layer sizes, e.g. --hidden 512 512")
    parser.add_argument("--channels", type=int, nargs="+", default=DEFAULTS["channels"],
                        help="CNN channels per conv block, e.g. --channels 32 64 128")
    parser.add_argument("--kernel-size", type=int, default=DEFAULTS["kernel_size"])
    parser.add_argument("--pool", choices=["max", "avg"], default=DEFAULTS["pool"])
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--augment", action="store_true", help="random crop + horizontal flip")
    parser.add_argument("--activation", choices=["sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu"],
                        default=DEFAULTS["activation"])
    parser.add_argument("--norm", choices=["raw", "unit"], default=DEFAULTS["norm"])
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=DEFAULTS["optimizer"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--train-subset", type=int, help="train on only N images (smoke tests)")
    parser.add_argument("--name", help="run name (default: built from the config)")
    parser.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    parser.add_argument("--save-model", action="store_true", help="also save the weights (model.pt)")
    args = parser.parse_args()

    run(vars(args))  # argparse names (kernel_size, batch_size, ...) match the config keys


if __name__ == "__main__":
    main()
