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
```

The script prints the accuracy, training time and test time of each classifier.

## Results

_To be filled in after the full run._
