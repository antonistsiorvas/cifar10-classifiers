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
| `data/` | Dataset, downloaded automatically on first run (not in git) | |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Run scripts from the repo root, e.g. `python interim/knn_ncc.py`.
The first run downloads CIFAR-10 (~163 MB) into `data/`.
