# Import Algorithms

from Algorithms import linear_sol_neum
from Algorithms import linear_sol_sm
from Algorithms import constructive_random
from Algorithms import constructive_greedy
from Algorithms import constructive_neuman
from Algorithms import constructive_sm

# Test Algorithms

import numpy as np
import scipy.sparse as sp
import time
import os
import csv
import traceback

# === Import your algorithm functions === #
from Optimization_Algorithms import (
    linear_sol_sm,
    linear_sol_neum,
    constructive_random,
    constructive_greedy,
    constructive_sm
)

# === Set tolerance / epsilon === #
phi_tol = 1e-3        # Convergence tolerance for GA = linear_sol
epsilon = 1e-8        # For Selective = Constructive algorithms

# === Define folders === #
base_path = "Real_World_Data"
results_dir = "Real_World_Results"
a_primes_dir = os.path.join(results_dir, "a_primes")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(a_primes_dir, exist_ok=True)

# === Prepare CSV result file === #
csv_file = os.path.join(results_dir, "results.csv")
write_header = not os.path.exists(csv_file)

# === Graphs and their initial Q_R values === #
graph_names = ["karate", "residence", "twitter", "blogs", "trinity", "facebook", "northwestern"]
Q_R_init = [0.552, 0.443, 0.517, 0.479, 0.478, 0.373, 0.472]

# Map each graph to its phi_target = Q_R_init + 0.1
phi_targets = {name: q_init + 0.1 for name, q_init in zip(graph_names, Q_R_init)}

print("✅ Starting RW Batch Job...", flush=True)

with open(csv_file, mode="a", newline="") as f_csv:
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(["graph", "method", "objective_score", "elapsed_time", "iterations", "delta_Q_R"])

    # === Prioritize small graph like 'karate' === #
    all_graphs = os.listdir(base_path)
    ordered_graphs = sorted(all_graphs, key=lambda name: 0 if name == "karate" else 1)

    for graph_name in ordered_graphs:
        graph_folder = os.path.join(base_path, graph_name) + os.sep

        if not os.path.isdir(graph_folder):
            continue  # Skip non-directory entries

        print(f"\n=== Processing Graph: {graph_name} ===", flush=True)

        try:
            # Load A, W, labels
            A = sp.load_npz(graph_folder + "A_uniform.npz")
            W = sp.load_npz(graph_folder + "W.npz")
            labels = np.loadtxt(graph_folder + "labels.csv", delimiter=",", skiprows=1, dtype=int)

            # Determine phi_target for this graph
            if graph_name in phi_targets:
                phi_target = phi_targets[graph_name]
            else:
                phi_target = 0.6  # baseline phi=0.6 if graph not in list

            # Apply all algorithms
            methods = [
                ("linear_sol_sm", linear_sol_sm, (W, A, labels, phi_target, phi_tol), True),
                ("linear_sol_neum", linear_sol_neum, (W, A, labels, phi_target, phi_tol), True),
                ("constructive_random", constructive_random, (W, A, labels, phi_target, epsilon), False),
                ("constructive_greedy", constructive_greedy, (W, A, labels, phi_target, epsilon), False),
                ("constructive_sm", constructive_sm, (W, A, labels, phi_target, epsilon), False)
            ]

            for method_name, method_func, args, has_delta in methods:
                print(f"\n--> Running {method_name}...", flush=True)
                try:
                    result = method_func(*args)
                    a_prime = result[0]
                    np.save(os.path.join(a_primes_dir, f"{graph_name}__{method_name}.npy"), a_prime)

                    row = [
                        graph_name,
                        method_name,
                        f"{result[1]:.6f}",
                        f"{result[2]:.3f}",
                        result[3]
                    ]
                    if has_delta:
                        row.append(f"{result[4]:.6f}")
                    else:
                        row.append("N/A")

                    writer.writerow(row)
                    f_csv.flush()

                except Exception as e:
                    print(f"❌ Error in {method_name} on {graph_name}: {e}", flush=True)
                    writer.writerow([graph_name, method_name, "ERROR", "ERROR", "ERROR", "ERROR"])
                    f_csv.flush()

        except Exception as e:
            print(f"❌ Failed to process graph {graph_name}: {e}", flush=True)
            for method_name, *_ in methods:
                writer.writerow([graph_name, method_name, "LOAD_FAIL", "LOAD_FAIL", "LOAD_FAIL", "LOAD_FAIL"])
                f_csv.flush()

print("\n✅ All Real-World Graphs Processed.", flush=True)
