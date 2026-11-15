# Assignment 1: Feedforward neural network (MLP / CNN) on CIFAR-10

Due: 22 November 2026, 23:59

Train feedforward networks (an MLP and a CNN) with back-propagation and compare them
against the 1-NN, 3-NN and Nearest Class Centroid results from [the interim assignment](../interim/)
(best: 1-NN 39.08%, 3-NN 37.47% with PCA 50, NCC ~27.7%).

Framework: PyTorch. Runs on CPU or a CUDA GPU (selected automatically).

## Structure

| File | Contents |
|---|---|
| [`data.py`](data.py) | Load CIFAR-10, train/validation split, normalization, data augmentation |
| [`models.py`](models.py) | The `MLP` and `CNN` models |
| [`train.py`](train.py) | Default settings, training loop, evaluation, saving the results of one run |
| [`experiments.py`](experiments.py) | Experiments E1–E6 and the command line (the only entry point) |
| [`colab_experiments.ipynb`](colab_experiments.ipynb) | Runs the experiments on Google Colab (see [Running on Colab](#running-on-colab)) |
| `results/` | Output of the experiments (see [Outputs](#outputs)) |

## Setup

- **Data:** the official 50,000 / 10,000 split. The 50,000 training images are split once
  (stratified, seed 42) into **45,000 train / 5,000 validation** (500 per class).
  Only the 45,000 are used for back-propagation; validation is used for early stopping;
  the 10,000 test images only for the final evaluation.
  The dataset is downloaded to `data/` automatically if it is missing.
- **Normalization:** `raw` (pixels 0–255, no normalization) or `unit` (pixels / 255, as in
  the interim; the default).
- **Data augmentation** (training batches only): random 32×32 crop from the image padded
  with 4 zero pixels on each side, and random horizontal flip.
- **Training:** cross-entropy loss, Adam or SGD (momentum 0.9), mini-batches in a new random
  order every epoch, up to 50 epochs with **early stopping** on the validation loss
  (stop after 8 epochs without improvement). At the end the weights of the best epoch are
  restored and train / validation / test accuracy are computed with them.

## Run

From the repo root:

```bash
python assignment-1/experiments.py --exp E1              # one experiment
python assignment-1/experiments.py --exp E1 E3           # several
python assignment-1/experiments.py --exp all             # everything
python assignment-1/experiments.py --exp E1 --smoke      # quick test: 2,000 images, 2 epochs
```

Runs whose `metrics.json` already exists are skipped, so after a disconnect (e.g. on Colab)
the same command continues where it stopped. To rerun a run, delete its folder in
`results/runs/` and its row in `results/summary.csv`.
`--smoke` writes to `results_smoke/` (gitignored) instead of `results/`.

## Running on Colab

[`colab_experiments.ipynb`](colab_experiments.ipynb) runs all experiments on a free Colab T4 GPU
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonistsiorvas/cifar10-classifiers/blob/main/assignment-1/colab_experiments.ipynb)

1. Open the notebook from GitHub in Colab (badge above) and select *Runtime → Change runtime
   type → T4 GPU*.
2. Run cells 1–6 (GPU check, mount Drive, clone the repo, install missing packages,
   download CIFAR-10, link `assignment-1/results/` to
   `MyDrive/cifar10-classifiers/assignment-1-results/`). Every finished run is written
   straight to Drive.
3. Cell 7 runs a smoke test of E1 and E5 and estimates the epoch times and the total time.
4. Run the experiment cells (one per experiment, cheapest first: E1, E2, E4, E3, E6, E5),
   in one session or across several.
5. After a disconnect, run cells 1–6 again and continue with the next experiment cell:
   finished runs are skipped, only the interrupted run starts again.
6. Cell 9 shows `summary.csv` and how many runs are finished; cell 10 zips the Drive folder
   into `MyDrive/cifar10-classifiers/results.zip` and downloads it. Extract it into
   `assignment-1/results/` locally (it contains `runs/` and `summary.csv`) and commit.

## Experiments

One factor changes at a time; everything else stays at the defaults in `train.py`
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

- `history.csv`: per epoch, training loss/accuracy (average over the epoch),
  validation loss/accuracy and epoch time
- `metrics.json`: the full config and the results: final train/validation/test accuracy,
  test loss, best epoch, epochs run, training time, test inference time, number of
  parameters, device
- `test_predictions.npy`: the predicted label of each of the 10,000 test images
- `test_probabilities.npy`: the predicted class probabilities (10,000 × 10, softmax)

Plus one row per run (config + results) in `results/summary.csv`.
