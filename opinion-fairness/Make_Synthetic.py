# ================================ #
# Code for making Synthetic Graphs #
# ================================ #

import numpy as np
import os
import scipy.sparse as sp

def generate_network(n, k, R, P, undirected=False):
    rng = np.random.default_rng(seed=None)
    labels = rng.choice(k, size=n, p=R)
    adjacency_list = [list() for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < P[labels[i]][labels[j]]:
                adjacency_list[i].append(j)
    
    for i in range(n):
        adjacency_list[i].append((i + 1) % n)
    
    for i in range(n):
        adjacency_list[i] = np.unique(adjacency_list[i]).tolist()
    
    return labels, adjacency_list

def adjacency_list_to_sparse_matrix(adjacency_list, n):
    rows, cols, data = [], [], []
    for i, neighbors in enumerate(adjacency_list):
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            data.append(1)
    
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    inv_row_sums = 1.0 / row_sums
    W = A.multiply(inv_row_sums[:, np.newaxis])
    
    return A, W

# ======================= #
# VARYING a RATIOS Graphs #
# ======================= #

# Define fixed probabilities P
P_fixed = [[0.2, 0.1], [0.1, 0.2]]

# Different ratios of red vs blue nodes (R)
team_ratios = [[round(r, 1), round(1 - r, 1)] for r in np.arange(0.1, 1, 0.1)]

# Define the (a_R, a_B) value ranges
value_ranges = {
    "low": (0.0, 0.33),
    "medium": (0.33, 0.66),
    "high": (0.66, 1.0),
}

# (a_R, a_B) mappings based on the desired ratios 
a_ratios = {
    "5":       (value_ranges["high"], value_ranges["low"]),   # high / low
    "3":       (value_ranges["medium"], value_ranges["low"]), # medium / low
    "2":       (value_ranges["high"], value_ranges["medium"]),  # high / medium
    "1":       (value_ranges["medium"], value_ranges["medium"]),  # medium / medium (only one green line kept)
    "1over2":  (value_ranges["medium"], value_ranges["high"]),  # medium / high
    "1over3":  (value_ranges["low"], value_ranges["medium"]),  # low / medium
    "1over5":  (value_ranges["low"], value_ranges["high"]),  # low / high
}

def save_graph_data(n, k, R, P, a_r_range, a_b_range, ratio_name):
    """ Saves SBM-generated graph data with given (a_R, a_B) values. """
    
    root_folder = f"synthetic_graphs_VARYING_a_P={P[0]}_n={n}"
    os.makedirs(root_folder, exist_ok=True)
    
    R_clean = [f"{float(x):.1f}" for x in R]  # ensures e.g., ['0.9', '0.1']
    graph_folder = os.path.join(root_folder, f"synthetic_graphs_R=[{','.join(R_clean)}]_P={P[0]}_a_ratio={ratio_name}")
    os.makedirs(graph_folder, exist_ok=True)
    
    graph_subfolder = os.path.join(graph_folder, f"graph_{n}_nodes")
    os.makedirs(graph_subfolder, exist_ok=True)
    
    for run in range(1, 6):
        labels, adjacency_list = generate_network(n=n, k=k, R=R, P=P, undirected=False)
        A, W = adjacency_list_to_sparse_matrix(adjacency_list, n)
        
        a_values = np.zeros(n)
        red_mask = labels == 0
        blue_mask = labels == 1
        
        # Assign a values based on the selected (a_R, a_B) range
        a_values[red_mask] = np.random.uniform(a_r_range[0], a_r_range[1], size=np.sum(red_mask))
        a_values[blue_mask] = np.random.uniform(a_b_range[0], a_b_range[1], size=np.sum(blue_mask))
        A_diag = sp.diags(a_values)
        
        run_folder = os.path.join(graph_subfolder, f"run_{run}")
        os.makedirs(run_folder, exist_ok=True)
        
        sp.save_npz(os.path.join(run_folder, "A.npz"), A_diag)
        sp.save_npz(os.path.join(run_folder, "W.npz"), W)
        np.savetxt(os.path.join(run_folder, "labels.csv"), labels, delimiter=",", fmt="%d", header="Label", comments="")
        
        print(f"Graph {run} saved in '{run_folder}' with sparse matrices A and W.")

# Generate graphs for all combinations of node ratios and (a_R, a_B) settings
fixed_n = 1000
for R in team_ratios:
    for ratio_name, (a_r_range, a_b_range) in a_ratios.items():
        save_graph_data(n=fixed_n, k=2, R=R, P=P_fixed, a_r_range=a_r_range, a_b_range=a_b_range, ratio_name=ratio_name)

print("\n✅ All graphs saved!")
