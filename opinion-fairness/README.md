# This repository contains the code and datasets used in the paper: Fairness in Opinion-Formation Dynamics

## Contents

- `Algorithms.py`: Implements all core algorithms used to optimize fairness across groups in a network.
- `Apply_Algorithms.py`: Applies the algorithms to real-world graphs located in the `Real_World_Data` folder and computes the experimental results.
- `Make_Real_World_Plots.py`: Generates the Real World Figures (2,3,4) used in the paper.
- `Make_Synthetic.py`: Generates the Synthetic graphs.
- `Make_Synthetic_Plots.py`: Generates the Synthetic Figure (1) used in the paper.

## Data

The `Real_World_Data/` folder contains real-world graphs used in our experiments. Each subfolder corresponds to a graph dataset and contains:
- `A_uniform.npz`: The uniform stubbornness matrix \( A \)
- `W.npz`: The network's weighted adjacency matrix \( W \)
- `labels.csv`: Node group labels (0 = Red or 1 = Blue, Teams)

The `Make_Synthetic.py` generates the Synthetic graphs in the folder: synthetic_graphs_VARYING_a_P=[0.2, 0.1]n=1000, which contains:
-synthetic_graphs_R=[{split}]_P=[0.2, 0.1]_a_ratio={ratio}

where:

{ratio} = [1, 2, 3, 5, 1over2, 1over3, 1over5]

{split} = [0.1,0.9, 0.2,0.8, 0.3,0.7, 0.4,0.6, 0.5,0.5, 0.6,0.4, 0.7,0.3, 0.8,0.2, 0.9,0.1]

## Reproducing Results

To reproduce the results and plots from the paper:

1. Run `Make_Synthetic.py` to generate the synthetic graphs
2. Run `Make_Synthetic_Plots.py` to generate the Synthetic Figure (1)
3. Run `Apply_Algorithms.py` to apply the optimization algorithms to the real world datasets
4. Run `Make_Real_World_Plots.py` to generate the real world Figures (2,3,4)

All scripts are written in Python 3 and rely on standard libraries such as NumPy, SciPy, Matplotlib and Cupy for GPU speedups.
To install the dependencies, run:
pip install numpy scipy matplotlib cupy

---

## Table of Results for Real world graphs

| Graph            | Metric     | Se-Rand | Se-Greedy | Se-SM | GA-NMA | GA-SM |
|------------------|------------|-------------------|---------------------|-----------------|--------|-------|
| karate           | Time(s)    | 0.008             | **0.002**           | 0.004           | 0.025  | 0.306 |
| (n=34)           | Iterations | 12                | **2**               | 3               | 31     | 214   |
|                  | Cost       | 1.760             | 1.071               | 0.582           | 0.335  | **0.175** |
| residence        | Time(s)    | 0.117             | **0.074**           | 0.088           | 0.659  | 1.944 |
| (n=217)          | Iterations | 58                | **32**              | 47              | 434    | 1766  |
|                  | Cost       | 27.679            | 22.928              | 20.054          | 19.043 | **14.527** |
| twitter          | Time(s)    | 0.232             | **0.097**           | 0.129           | 1.653  | 5.842 |
| (n=475)          | Iterations | 92                | **24**              | 36              | 566    | 2242  |
|                  | Cost       | 23.431            | 15.892              | 9.202           | 3.255  | **2.620** |
| blogs            | Time(s)    | 7.051             | **2.025**           | 3.121           | 65.624 | 173.28 |
| (n=1,222)        | Iterations | 808               | **167**             | 210             | 7057   | 17902 |
|                  | Cost       | 258.43            | 88.167              | 63.866          | 92.278 | **49.822** |
| trinity          | Time(s)    | 77.612            | **32.085**          | 39.930          | 247.92 | 1352.9 |
| (n=2,613)        | Iterations | 764               | **320**             | 492             | 3345   | 16887 |
|                  | Cost       | 365.45            | 209.43              | 162.108         | 118.35 | **90.937** |
| facebook         | Time(s)    | 218.54            | **50.656**          | 59.081          | 787.20 | 4927.9 |
| (n=4,039)        | Iterations | 1813              | **266**             | 383             | 5231   | 29544 |
|                  | Cost       | 502.36            | 162.82              | 107.50          | 47.154 | **33.892** |
| northwestern     | Time(s)    | 2397.6            | **1234.5**          | 2134.2          | 13879  | 81088 |
| (n=10,537)       | Iterations | 2095              | **1012**            | 1704            | 12764  | 73350 |
|                  | Cost       | 802.36            | 720.49              | 553.62          | 345.94 | **276.14** |

---

This repository is provided anonymously to support reproducibility of the paper's results.
