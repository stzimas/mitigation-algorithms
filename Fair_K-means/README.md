# Fair K-Means

This repository implements **Fair K-Means**, a fairness-aware extension of the classic K-Means clustering algorithm.

The method modifies the standard assignment step of K-Means in order to produce **more balanced clusters with respect to a binary sensitive attribute** (e.g., gender, race), while preserving good clustering quality.

---

## Motivation

Standard K-Means clusters points based only on distance.
As a result, clusters may become highly imbalanced with respect to sensitive attributes.

**Fair K-Means** addresses this issue by introducing a **fairness-aware force** that influences point assignments, encouraging clusters to move toward balanced compositions.

---

## Key Idea

Each point is influenced by:

1. **Geometric attraction** to nearby centroids (as in standard K-Means)
2. **Fairness-aware interaction** based on the imbalance of each cluster (imbalance = 1- balance(c))

These two effects are combined using a single parameter λ, which controls the trade-off between clustering quality and fairness.

---

## How It Works

* Each data point has a binary sensitive attribute (e.g., red / blue)
* Each cluster has imbalance based on balance metric
* Clusters with strong imbalance exert a strong influence
* Points are assigned to the cluster with the strongest combined influence/force
* Centroids are updated as usual
* The process repeats until convergence

When ( λ = 0 ), the algorithm behaves exactly like standard K-Means.

---

## Usage Example

```python
model = KMeansBalanced(
    n_clusters=3,
    lambda_=0.3,
    init_mode="kmeans"
)
model.fit(X, sensitive_attributes)
labels = model.labels_
```

---

## Notes

* Designed for binary sensitive attributes
* Keeps the simplicity and scalability of K-Means
* Introduces fairness without hard constraints
* Easy to tune through a single parameter


---

## Dependencies

* numpy
* scikit-learn

