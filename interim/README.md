# Interim assignment: Nearest Neighbour vs Nearest Class Centroid

Due: 8 November 2026, 23:59

Compares three classifiers on CIFAR-10 using the official train/test split:

- **1-NN**: label of the closest training image
- **3-NN**: majority vote of the 3 closest training images
- **Nearest Class Centroid (NCC)**: each class is represented by the mean of its training images; a test image gets the class of the closest mean

Pixels are scaled to [0, 1], and all three classifiers use Euclidean distance,
computed either on the raw pixels (3072 dimensions) or, with `--pca`, on the
PCA-reduced features. scikit-learn's `KNeighborsClassifier` and `NearestCentroid`
were employed in [`knn_ncc.py`](knn_ncc.py).

## Run

From the repo root:

```bash
python interim/knn_ncc.py                 # full test set (10,000 images)
python interim/knn_ncc.py --subset 1000   # quick run on 1,000 test images
python interim/knn_ncc.py --pca 100       # PCA to 100 dimensions first
python interim/knn_ncc.py --examples      # also save example predictions
python interim/knn_ncc.py --per-class     # also save per-class accuracy
```

The script prints the accuracy, training time and test time of each classifier.
With `--examples` it also saves, for every classifier and setting (raw and PCA
10/20/50/100/200/500), a figure with 2 correctly and 2 wrongly classified test
images to `results/examples/`, plus an `examples_index.csv` listing them.
With `--per-class` it saves the accuracy of each class for every classifier and
setting to `results/per_class_accuracy.csv`.

## Results

Full official split: 50,000 training and 10,000 test images. "Variance kept" is
the fraction of the training data's variance kept by PCA. Times are in seconds
and do not include fitting the PCA itself.

### Accuracy

Accuracy is the fraction of test images whose predicted class matches their
true class:

$$\text{accuracy} = \frac{\text{number of correctly classified test images}}{\text{total number of test images}}$$

It is computed on the 10,000 test images with scikit-learn's `score` method.
Since every class has exactly 1,000 test images, it also equals the average of
the per-class accuracies below.

| Setting | Variance kept | 1-NN | 3-NN | NCC |
|---|---|---|---|---|
| Raw pixels (3072) | 100% | 35.39% | 33.03% | 27.74% |
| PCA 10 | 65.5% | 30.72% | 30.85% | 26.37% |
| PCA 20 | 74.5% | 37.27% | 36.95% | 27.33% |
| PCA 50 | 84.3% | **39.08%** | **37.47%** | 27.60% |
| PCA 100 | 90.1% | 38.61% | 36.80% | 27.66% |
| PCA 200 | 94.6% | 37.37% | 35.25% | 27.72% |
| PCA 500 | 98.3% | 36.24% | 33.70% | 27.74% |

### Training and testing time (s)

| Setting | 1-NN train | 1-NN test | 3-NN train | 3-NN test | NCC train | NCC test |
|---|---|---|---|---|---|---|
| Raw pixels (3072) | 0.11 | 21.65 | 0.14 | 20.46 | 2.43 | 0.13 |
| PCA 10 | 0.09 | 1.45 | 0.10 | 1.28 | 0.04 | 0.89 |
| PCA 20 | 0.01 | 1.33 | 0.01 | 0.56 | 0.03 | 0.01 |
| PCA 50 | 0.01 | 1.46 | 0.01 | 0.71 | 0.06 | 0.01 |
| PCA 100 | 0.04 | 1.81 | 0.03 | 1.04 | 0.10 | 0.01 |
| PCA 200 | 0.05 | 2.57 | 0.04 | 1.78 | 0.15 | 0.01 |
| PCA 500 | 0.09 | 4.55 | 0.09 | 3.77 | 0.33 | 0.02 |

### Per-class accuracy

Fraction of the test images of each class (1,000 per class) that are classified
correctly. Produced with `python interim/knn_ncc.py --per-class`, which also
saves them to `results/per_class_accuracy.csv`. Every run fixes the PCA random
seed (`random_state=42`), so the per-class averages match the overall
accuracies above.

#### 1-NN

| Class | Raw | PCA 10 | PCA 20 | PCA 50 | PCA 100 | PCA 200 | PCA 500 |
|---|---|---|---|---|---|---|---|
| airplane | 48.50% | 40.40% | 47.60% | 52.10% | 50.90% | 50.10% | 48.80% |
| automobile | 21.80% | 31.30% | 36.60% | 35.50% | 31.00% | 25.90% | 23.30% |
| bird | 38.40% | 26.50% | 35.10% | 38.00% | 39.30% | 39.40% | 38.60% |
| cat | 24.00% | 21.30% | 24.50% | 23.80% | 24.20% | 24.00% | 24.00% |
| deer | 45.70% | 26.20% | 33.60% | 40.30% | 43.50% | 44.50% | 45.90% |
| dog | 29.00% | 26.20% | 29.60% | 30.50% | 29.90% | 28.70% | 28.70% |
| frog | 35.30% | 34.00% | 43.90% | 45.80% | 45.20% | 42.00% | 38.90% |
| horse | 29.40% | 29.50% | 37.10% | 38.10% | 36.10% | 34.20% | 31.00% |
| ship | 61.90% | 39.20% | 49.80% | 55.30% | 57.90% | 60.50% | 61.80% |
| truck | 19.90% | 32.60% | 34.90% | 31.40% | 28.10% | 24.40% | 21.40% |

#### 3-NN

| Class | Raw | PCA 10 | PCA 20 | PCA 50 | PCA 100 | PCA 200 | PCA 500 |
|---|---|---|---|---|---|---|---|
| airplane | 57.30% | 56.30% | 60.40% | 60.40% | 59.70% | 57.90% | 57.70% |
| automobile | 24.30% | 39.50% | 46.00% | 41.40% | 36.20% | 30.60% | 26.10% |
| bird | 45.00% | 35.10% | 41.60% | 45.50% | 45.80% | 46.00% | 44.60% |
| cat | 22.50% | 24.00% | 25.70% | 23.00% | 22.20% | 23.30% | 22.60% |
| deer | 43.60% | 22.90% | 29.70% | 38.40% | 41.50% | 43.40% | 44.20% |
| dog | 21.30% | 21.40% | 24.70% | 23.80% | 21.70% | 21.70% | 21.50% |
| frog | 22.90% | 26.70% | 36.70% | 37.90% | 36.40% | 31.60% | 26.30% |
| horse | 19.90% | 21.20% | 31.40% | 28.10% | 26.80% | 23.00% | 20.70% |
| ship | 61.20% | 36.80% | 45.90% | 53.40% | 56.80% | 58.60% | 59.60% |
| truck | 12.30% | 24.60% | 27.40% | 22.80% | 20.90% | 16.40% | 13.70% |

#### NCC

| Class | Raw | PCA 10 | PCA 20 | PCA 50 | PCA 100 | PCA 200 | PCA 500 |
|---|---|---|---|---|---|---|---|
| airplane | 53.90% | 53.70% | 53.70% | 53.90% | 53.90% | 53.90% | 53.90% |
| automobile | 18.60% | 17.20% | 18.00% | 18.40% | 18.60% | 18.60% | 18.50% |
| bird | 10.70% | 7.90% | 10.10% | 10.80% | 10.70% | 10.70% | 10.70% |
| cat | 5.60% | 3.30% | 5.60% | 5.30% | 5.60% | 5.70% | 5.70% |
| deer | 11.90% | 8.40% | 11.20% | 11.60% | 11.80% | 11.80% | 11.90% |
| dog | 28.60% | 29.00% | 28.30% | 28.60% | 28.60% | 28.60% | 28.70% |
| frog | 53.80% | 52.60% | 53.30% | 53.30% | 53.70% | 53.80% | 53.80% |
| horse | 16.60% | 14.60% | 16.00% | 16.40% | 16.30% | 16.60% | 16.60% |
| ship | 37.00% | 36.70% | 36.50% | 37.20% | 36.80% | 36.90% | 37.00% |
| truck | 40.70% | 40.30% | 40.60% | 40.50% | 40.60% | 40.60% | 40.60% |
