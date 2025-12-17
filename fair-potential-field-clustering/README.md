# Potential Field Clustering (f-PFC)

This repository implements **Potential Field Clustering (f-PFC)**, a physics-inspired clustering algorithm based on **Potential Fields**.

## Overview

Each data point is modeled as a particle that moves under the influence of:

* **Gravitational attraction** (geometric similarity)
* **Electrostatic interaction** (fairness constraints)

The algorithm iteratively moves points according to the resultant force until convergence, then merges nearby points to form clusters.

## Core Concepts

* Gravitational force: pulls points together based on distance
* Electrostatic force: attracts or repels points based on cluster imbalance
* Work–energy principle controls displacement magnitude
* Automatic or manual neighborhood radius (`h`)
* Adaptive merging based on movement thresholds

## Main Parameters

* `G`: gravitational constant
* `K`: electrostatic constant
* `lamda`: balance between geometry and fairness
* `W`: work parameter controlling displacement
* `h`: interaction radius (optional, auto-estimated)
* `min_step`: convergence threshold

## Usage

```python
result = f_pfc(
    points=X,
    attributes=attributes,
    lamda=0.5,
    G=1.0,
    K=1.0
)
labels = result["labels"]
```

## Output

The algorithm returns:

* final cluster centers
* point labels
* force and displacement history
* convergence diagnostics
* fairness metrics per cluster

## Dependencies

* numpy
* scipy
* scikit-learn

## Notes

Setting `lamda = 0` removes fairness effects, yielding purely geometry-based clustering.
