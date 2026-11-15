# Assignment 1: Feedforward neural network (MLP / CNN) on CIFAR-10

Due: 22 November 2026, 23:59

Train feedforward networks (an MLP and a CNN) with back-propagation and compare them
against the 1-NN, 3-NN and Nearest Class Centroid results from [the interim assignment](../interim/)
(best: 1-NN 39.08%, 3-NN 37.47% with PCA 50, NCC ~27.7%).

Framework: PyTorch. Runs on CPU or a CUDA GPU (selected automatically).

## Structure

| File | Contents |
|---|---|
| [`data.py`](data.py) | Loading, train/validation split, normalization, batching, data augmentation |
| [`models.py`](models.py) | `MLP`, `CNN`, activation functions, parameter count |
| [`train.py`](train.py) | Trains and evaluates one configuration (CLI, or `run(config)` from Python) |
| [`experiments.py`](experiments.py) | Definitions of experiments E1–E6 and a runner |
| `results/` | Output of the experiments (see [Outputs](#outputs)) |

## Setup

- **Data:** the official 50,000 / 10,000 split. The 50,000 training images are split once
  (stratified, seed 42) into **45,000 train / 5,000 validation** (500 per class).
  Only the 45,000 are used for back-propagation; validation is used for early stopping;
  the 10,000 test images only for the final evaluation.
  The dataset is downloaded to `data/` automatically if it is missing.
- **Normalization** (`--norm`): `raw` (0–255, no normalization) or `unit` (divide by 255,
  as in the interim; the default).
- **Data augmentation** (`--augment`, training batches only): random 32×32 crop from the
  image padded with 4 zero pixels on each side, and random horizontal flip.
- **Training:** cross-entropy loss, Adam or SGD (momentum 0.9), mini-batches in a new random
  order every epoch, up to `--epochs` epochs with **early stopping** on the validation loss
  (`--patience` epochs without improvement). At the end the weights of the best epoch are
  restored and train / validation / test accuracy are computed with them.

## Run

From the repo root.

### One configuration

```bash
python assignment-1/train.py --model mlp --hidden 512 512 512
python assignment-1/train.py --model cnn --channels 32 64 128 --batchnorm --dropout 0.25 --augment
python assignment-1/train.py --model mlp --hidden 512 --train-subset 10000 --epochs 5   # quick CPU check
```

Main options (defaults in brackets): `--model {mlp,cnn}` [mlp], `--hidden` [512 512 512],
`--channels` [32 64 128], `--kernel-size` [3], `--pool {max,avg}` [max], `--batchnorm`,
`--dropout` [0], `--augment`, `--activation` [relu], `--norm {raw,unit}` [unit],
`--optimizer {adam,sgd}` [adam], `--lr` [1e-3], `--batch-size` [128], `--epochs` [50],
`--patience` [8], `--seed` [42], `--train-subset N`, `--name`, `--out-dir`, `--save-model`.
See `python assignment-1/train.py --help`.

### Experiments

```bash
python assignment-1/experiments.py --exp all --dry-run   # list the runs
python assignment-1/experiments.py --exp E1              # one experiment
python assignment-1/experiments.py --exp E1 E3           # several
python assignment-1/experiments.py --exp all             # everything
python assignment-1/experiments.py --exp all --smoke     # smoke test: 2,000 images, 2 epochs
```

Runs whose `metrics.json` already exists are skipped, so after a disconnect the same
command continues where it stopped. `--force` reruns them, `--verbose` prints every epoch.
The smoke test writes to `results_smoke/` (gitignored).

One factor changes at a time; everything else stays at the default
(ReLU, Adam, lr 1e-3, batch 128, max 50 epochs, patience 8, `unit` normalization, seed 42,
no augmentation / dropout / batchnorm).
Reference models: **MLP_REF** = 3 hidden layers × 512, **CNN_REF** = conv channels
[32, 64, 128], kernel 3, max-pool, then a 256-unit fully connected layer.

| Exp | What changes | Runs |
|---|---|---|
| E1 | MLP width, 1 hidden layer: 32, 64, 128, 256, 512, 1024, 2048 | 7 |
| E2 | MLP depth {2, 3, 4} × width {64, 128, 256, 512} | 12 |
| E3 | Activation: sigmoid, tanh, relu, leaky_relu, elu, gelu (MLP_REF and CNN_REF) | 12 |
| E4 | Input normalization: raw vs unit (MLP_REF and CNN_REF) | 4 |
| E5 | CNN_REF variations: kernel 5/7, channels ×0.5/×2, 2/4 blocks, avg-pool, batchnorm, dropout 0.25/0.5, augmentation (+ the reference) | 12 |
| E6 | Learning rate 1e-4/1e-2, batch size 32/512, SGD (+ the reference), for MLP_REF and CNN_REF | 12 |
| | **Total** | **59** |

## Outputs

For every run, in `results/runs/<run_name>/`:

- `history.csv`: per epoch, training loss/accuracy (running average over the epoch),
  validation loss/accuracy and epoch time
- `metrics.json`: the full config, final train/validation/test accuracy and loss, best
  epoch, epochs run, training time, test inference time, number of parameters, device
- `test_predictions.npy`: the predicted label of each of the 10,000 test images
- `model.pt`: the weights, only with `--save-model`

Plus one row per run in `results/summary.csv`.
