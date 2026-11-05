# Neural Networks & Deep Learning

Coursework for the course «Νευρωνικά Δίκτυα – Βαθιά Μάθηση».
All work uses the **CIFAR-10** dataset (10 classes, 32×32 colour images, 50,000 train / 10,000 test).

## Structure

| Folder | Assignment | Due |
|---|---|---|
| [`interim/`](interim/) | Interim: 1-NN and 3-NN vs Nearest Class Centroid | 8 Nov 2026 |
| [`assignment-1/`](assignment-1/) | Feedforward NN (MLP/CNN) trained with back-propagation | 22 Nov 2026 |
| [`assignment-2/`](assignment-2/) | TBA | |
| [`assignment-3/`](assignment-3/) | TBA | |
| [`common/`](common/) | Code shared by all assignments (dataset loading) | |
| `data/` | CIFAR-10 dataset (not in git, see [How to get the data](#how-to-get-the-data)) | |
| [`report/`](report/) | Material for the written reports (references) | |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Run scripts from the repo root, e.g. `python interim/knn_ncc.py`.

## Dataset

This project uses the CIFAR-10 dataset [1]: 60,000 colour images of size 32×32
in 10 classes, with the official split of 50,000 training and 10,000 test images.
Website: https://www.cs.toronto.edu/~kriz/cifar.html

### How to get the data

1. Download the **CIFAR-10 python version** (`cifar-10-python.tar.gz`) from the website above.
2. Extract it.
3. Place the `cifar-10-batches-py/` folder in `data/`, so you have
   `data/cifar-10-batches-py/data_batch_1`, ..., `data_batch_5` and `test_batch`.

The dataset is not committed to the repo (`data/` is in `.gitignore`).

## References

[1] Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.
