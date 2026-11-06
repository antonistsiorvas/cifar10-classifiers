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

| Setting | Variance kept | 1-NN | 3-NN | NCC |
|---|---|---|---|---|
| Raw pixels (3072) | 100% | 35.39% | 33.03% | 27.74% |
| PCA 10 | 65.5% | 30.71% | 30.85% | 26.37% |
| PCA 20 | 74.5% | 37.31% | 37.00% | 27.35% |
| PCA 50 | 84.3% | **39.10%** | **37.63%** | 27.60% |
| PCA 100 | 90.1% | 38.55% | 36.80% | 27.66% |
| PCA 200 | 94.6% | 37.38% | 35.31% | 27.73% |
| PCA 500 | 98.3% | 36.18% | 33.66% | 27.73% |

### Training and testing time (s)

| Setting | 1-NN train | 1-NN test | 3-NN train | 3-NN test | NCC train | NCC test |
|---|---|---|---|---|---|---|
| Raw pixels (3072) | 0.12 | 22.06 | 0.16 | 20.58 | 2.67 | 0.11 |
| PCA 10 | 0.07 | 0.77 | 0.06 | 1.04 | 0.02 | 1.79* |
| PCA 20 | 0.01 | 1.98 | 0.01 | 0.48 | 0.03 | 0.01 |
| PCA 50 | 0.02 | 1.50 | 0.01 | 0.75 | 0.06 | 0.01 |
| PCA 100 | 0.02 | 1.94 | 0.02 | 1.24 | 0.10 | 0.01 |
| PCA 200 | 0.05 | 2.61 | 0.04 | 1.84 | 0.16 | 0.01 |
| PCA 500 | 0.09 | 5.58 | 0.09 | 4.64 | 0.39 | 0.04 |

\* NCC's test time with PCA 10 was 0.75 s in a second run, so it is
consistently higher than in the other settings, but still under 2 s.

### Per-class accuracy

Fraction of the test images of each class (1,000 per class) that are classified
correctly. Produced with `python interim/knn_ncc.py --per-class`, which also
saves them to `results/per_class_accuracy.csv`. That run fixes the PCA random
seed, so its per-class averages can differ from the overall accuracies above in
the second decimal (e.g. 39.08% vs 39.10% for 1-NN, PCA 50).

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

### Observations

- 1-NN is the best classifier in every setting except PCA 10, where 3-NN is
  marginally ahead (30.85% vs 30.71%). NCC is the worst everywhere, but still
  well above chance (10%).
- PCA helps k-NN, but only with enough components. With PCA 10 (65.5% of the
  variance) 1-NN drops to 30.71%, below raw pixels (35.39%). Accuracy then
  rises to the best result, 1-NN with PCA 50 (39.10%), and keeping more
  components moves it back towards the raw-pixel result.
- NCC stays at 27.4–27.7% from PCA 20 upwards and drops only to 26.4% with
  PCA 10, so PCA barely affects it.
- k-NN does almost nothing during training (it only stores the data) and does
  all its work at test time. NCC is the other way round: training computes the
  10 class means, and testing only compares each image with those 10 means.
- PCA makes k-NN testing about 4 to 40 times faster than on raw pixels
  (20–22 s down to 0.5–5.6 s).
- Per class, k-NN does best on ship and airplane (about 50–62% from PCA 20
  upwards), whose images often have plain blue backgrounds (sea, sky). Cat is
  poor in every setting (21–26%), and truck is the worst class on raw pixels
  (12–20%).
- The classifiers fail on different classes. NCC is very weak on cat (3–6%),
  bird and deer (8–12%), but beats k-NN on frog and truck (about 54%
  and 41%). A single class mean cannot represent classes whose images vary a
  lot.
- PCA's gain for k-NN comes mostly from automobile, truck, horse and frog
  (e.g. 1-NN automobile: 21.8% raw, 36.6% with PCA 20). Ship and deer instead
  get clearly worse with few components (1-NN ship: 61.9% raw, 49.8% with
  PCA 20).
- 3-NN beats 1-NN on airplane, bird and automobile, but loses on frog, horse,
  truck and dog, which explains its lower overall accuracy.
