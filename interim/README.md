# Interim assignment: Nearest Neighbour vs Nearest Class Centroid

Due: 8 November 2026, 23:59

Compares three classifiers on CIFAR-10 using the official train/test split:

- **1-NN**: label of the closest training image
- **3-NN**: majority vote of the 3 closest training images
- **Nearest Class Centroid (NCC)**: each class is represented by the mean of its training images; a test image gets the class of the closest mean

All three use Euclidean distance on raw pixels scaled to [0, 1]. They use
scikit-learn's `KNeighborsClassifier` and `NearestCentroid` in [`knn_ncc.py`](knn_ncc.py).

## Run

From the repo root:

```bash
python interim/knn_ncc.py                 # full test set (10,000 images)
python interim/knn_ncc.py --subset 1000   # quick run on 1,000 test images
python interim/knn_ncc.py --pca 100       # PCA to 100 dimensions first
python interim/knn_ncc.py --examples      # also save example predictions
```

The script prints the accuracy, training time and test time of each classifier.
With `--examples` it also saves, for every classifier and setting (raw and PCA
20/50/100/200/500), a figure with 2 correctly and 2 wrongly classified test
images to `results/examples/`, plus an `examples_index.csv` listing them.

## Results

Full official split: 50,000 training and 10,000 test images. "Variance kept" is
the fraction of the training data's variance kept by PCA. Times are in seconds
and do not include fitting the PCA itself.

### Accuracy

| Setting | Variance kept | 1-NN | 3-NN | NCC |
|---|---|---|---|---|
| Raw pixels (3072) | 100% | 35.39% | 33.03% | 27.74% |
| PCA 20 | 74.5% | 37.31% | 37.00% | 27.35% |
| PCA 50 | 84.3% | **39.10%** | **37.63%** | 27.60% |
| PCA 100 | 90.1% | 38.55% | 36.80% | 27.66% |
| PCA 200 | 94.6% | 37.38% | 35.31% | 27.73% |
| PCA 500 | 98.3% | 36.18% | 33.66% | 27.73% |

### Training and testing time (s)

| Setting | 1-NN train | 1-NN test | 3-NN train | 3-NN test | NCC train | NCC test |
|---|---|---|---|---|---|---|
| Raw pixels (3072) | 0.12 | 22.06 | 0.16 | 20.58 | 2.67 | 0.11 |
| PCA 20 | 0.01 | 1.98 | 0.01 | 0.48 | 0.03 | 0.01 |
| PCA 50 | 0.02 | 1.50 | 0.01 | 0.75 | 0.06 | 0.01 |
| PCA 100 | 0.02 | 1.94 | 0.02 | 1.24 | 0.10 | 0.01 |
| PCA 200 | 0.05 | 2.61 | 0.04 | 1.84 | 0.16 | 0.01 |
| PCA 500 | 0.09 | 5.58 | 0.09 | 4.64 | 0.39 | 0.04 |

### Observations

- 1-NN is the best classifier in every setting, and NCC is the worst. NCC is
  still well above chance (10%).
- PCA helps k-NN: the best result is 1-NN with PCA 50 (39.10%, compared with
  35.39% on raw pixels). Keeping more components moves accuracy back towards
  the raw-pixel result.
- NCC stays at about 27.5% in every setting, so PCA barely affects it.
- k-NN does almost nothing during training (it only stores the data) and does
  all its work at test time. NCC is the other way round: training computes the
  10 class means, and testing only compares each image with those 10 means.
- PCA makes k-NN testing about 4 to 40 times faster than on raw pixels
  (20–22 s down to 0.5–5.6 s).
