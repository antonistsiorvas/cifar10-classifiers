# Interim assignment: Nearest Neighbour vs Nearest Class Centroid

Due: 8 November 2026, 23:59

Compares three classifiers on CIFAR-10 using the official train/test split:

- **1-NN**: label of the closest training image
- **3-NN**: majority vote of the 3 closest training images (ties go to the class with the closest neighbour)
- **Nearest Class Centroid (NCC)**: each class is represented by the mean of its training images; a test image gets the class of the closest mean

All three use Euclidean distance on raw pixels scaled to [0, 1] and are implemented
from scratch with NumPy in [`knn_ncc.py`](knn_ncc.py).

## Run

From the repo root:

```bash
python interim/knn_ncc.py                 # full test set (10,000 images)
python interim/knn_ncc.py --subset 1000   # quick run on 1,000 test images
python interim/knn_ncc.py --pca 100       # PCA to 100 dimensions first
```

Results (accuracy, timings, per-class accuracy) are printed and saved as JSON in [`results/`](results/).

## Results

_To be filled in after the full run._
