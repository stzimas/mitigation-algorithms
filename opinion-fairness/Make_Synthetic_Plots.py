# =================== #
# MAKE VARYING A PLOT #
# =================== #

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.sparse as sp

# Function to load graph data
def load_graph(graph_folder):
    """Load graph data from the given folder."""
    W_sparse = sp.load_npz(os.path.join(graph_folder, "W.npz"))
    A_sparse = sp.load_npz(os.path.join(graph_folder, "A.npz"))
    
    # Convert sparse matrices back to dense format
    W = W_sparse.toarray()
    A = A_sparse.toarray()
    
    labels_path = os.path.join(graph_folder, "labels.csv")
    labels = np.loadtxt(labels_path, delimiter=",", skiprows=1)  # Skip header
    
    return W, A, labels

# Function to compute Q_R
def compute_Q_R(W, a_prime, labels):
    """Compute Q_R efficiently."""
    n = len(labels)
    red_mask = labels == 0

    A_prime = np.diag(a_prime)
    B = np.eye(n) - (np.eye(n) - A_prime) @ W

    # Solve B @ X = A_prime using a direct solver
    Q = np.linalg.solve(B, A_prime)

    return np.sum(Q[:, red_mask]) / n  # Sum all elements in Red columns

# ============================================== #
# 🔹 Plot Initial Q_R for Varying Team Ratios 🔹 #
# ============================================== #

# Define base folder containing graphs
base_folder = "synthetic_graphs_VARYING_a_P=[0.2, 0.1]_n=1000"

# Extract all subfolders
all_folders = sorted([
    folder for folder in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, folder))
])

# Extract and sort unique team ratios
team_ratios_set = set()
for folder in all_folders:
    try:
        ratio_str = folder.split("R=")[1].split("_P=")[0]
        team_ratios_set.add(ratio_str)
    except IndexError:
        print(f"⚠️ Skipping invalid folder format: {folder}")

# Fix team ratio formatting
team_ratios = sorted(list(team_ratios_set), key=lambda x: float(x.split(",")[0].replace("[", "")))
team_ratios_formatted = [f"{int(float(r.split(',')[0].replace('[', '')) * 100)}%" for r in team_ratios]  # X-axis labels

# Define color mapping based on **new** a_R / a_B ratios
color_mapping = {
    "5": '#8B0000',      # Dark red
    "3": '#FF0000',      # Bright red
    "2": '#FF6347',      # Orange-red
    "1": '#32CD32',      # Green (neutral)
    "1over2": '#1E90FF', # Strong blue
    "1over3": '#4169E1', # Royal blue
    "1over5": '#0000FF'  # Deep blue
}

# Store results for each a_R / a_B combination
results = {ratio: {"mean_Q_R": [], "std_Q_R": []} for ratio in color_mapping.keys()}

# Iterate over each a_R / a_B combination
for ratio in color_mapping.keys():
    for team_ratio in team_ratios:
        folder_name = f"synthetic_graphs_R={team_ratio}_P=[0.2, 0.1]_a_ratio={ratio}"
        folder_path = os.path.join(base_folder, folder_name, "graph_1000_nodes")

        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"⚠️ Missing folder: {folder_path}")
            results[ratio]["mean_Q_R"].append(None)
            results[ratio]["std_Q_R"].append(None)
            continue

        runs = [f for f in os.listdir(folder_path) if f.startswith("run_")]
        Q_R_values = []

        # Compute Q_R for each run
        for run in runs:
            graph_path = os.path.join(folder_path, run)
            W, A, labels = load_graph(graph_path)
            a_prime = np.diag(A)  # Extract the diagonal of A

            Q_R = compute_Q_R(W, a_prime, labels)
            Q_R_values.append(Q_R)

        # Compute mean and std of Q_R values
        results[ratio]["mean_Q_R"].append(np.mean(Q_R_values))
        results[ratio]["std_Q_R"].append(np.std(Q_R_values))

formatted_labels = {
    "5": r"$r=5$",
    "3": r"$r=3$",
    "2": r"$r=2$",
    "1": r"$r=1$",
    "1over2": r"$r=1/2$",
    "1over3": r"$r=1/3$",
    "1over5": r"$r=1/5$"
}

# ======================================= #
# 🔹 Plot the results for all r values 🔹 #
# ======================================= #

plt.figure(figsize=(12, 6))

for ratio in results.keys():
    label = formatted_labels[ratio]  # Use r notation
    plt.errorbar(
        team_ratios_formatted, results[ratio]["mean_Q_R"],
        yerr=results[ratio]["std_Q_R"], fmt="o-",
        color=color_mapping[ratio], label=label
    )

# Increase the font size of axis labels
plt.xlabel("Percentage of Red Group Nodes", fontsize=16)  # Bigger x-axis title
plt.ylabel(r"Red Group Influence ($Q_R$)", fontsize=16)   # Bigger y-axis title

# Keep x-axis labels horizontal
plt.xticks(team_ratios_formatted, ha="center", fontsize=14)  # Bigger x-axis tick labels
plt.yticks(fontsize=14)  # Bigger y-axis tick labels

# Increase the size of the legend labels
plt.legend(title=r"Ratio: $r = \alpha_R / \alpha_B$", title_fontsize=14, fontsize=14)

plt.grid(True)
output_path = "Initial_Q_R_vs_Team_Ratio.pdf"
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
plt.close()


# ============= #
# Make the PLOT #
# ============= #

import pandas as pd
import matplotlib.pyplot as plt

# --- Load results ---
df = pd.read_csv("experiment_results_costs_by_aratio.csv")

# --- Define desired a_ratio order ---
ratio_order = ["1over5", "1over3", "1over2", "1", "2", "3", "5"]
df["a_ratio"] = pd.Categorical(df["a_ratio"], categories=ratio_order, ordered=True)
df = df.sort_values("a_ratio")

# --- Mapping of internal names to display names (in desired order) ---
algorithm_display_order = [
    ("constructive_random", "Se-Rand"),
    ("constructive_greedy", "Se-Greedy"),
    ("constructive_neuman", "Se-NMA"),
    ("constructive_sm", "Se-SM"),
    ("linear_sol_neum", "GA-NMA"),
    ("linear_sol_sm", "GA-SM"),
]

# --- Plot setup ---
plt.figure(figsize=(10, 6))
colors = plt.cm.get_cmap("tab10", len(algorithm_display_order))
markers = ['o', 's', '^', 'D', 'v', '*']

# --- Plot each algorithm ---
for i, (algo_key, display_name) in enumerate(algorithm_display_order):
    data = df[df["algorithm"] == algo_key]
    x = range(len(ratio_order))  # use numeric x-coordinates
    y = data["mean_cost"]
    yerr = data["std_cost"]
    plt.errorbar(
        x, y, yerr=yerr,
        label=display_name,
        color=colors(i),
        marker=markers[i % len(markers)],
        markersize=6,
        capsize=3,
        elinewidth=1.5,
        linewidth=2.5,
        linestyle='-',
        alpha=0.95
    )

# --- Axis labels ---
plt.xlabel(r"$\mathrm{ratio} \; r = \frac{\alpha_R}{\alpha_B}$", fontsize=15)
plt.ylabel("Cost", fontsize=15)

# --- Custom x-tick labels (LaTeX fractions) ---
custom_labels = [r"$\frac{1}{5}$", r"$\frac{1}{3}$", r"$\frac{1}{2}$", "1", "2", "3", "5"]
plt.xticks(ticks=range(len(ratio_order)), labels=custom_labels, fontsize=11)

# --- Styling ---
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Algorithm", fontsize=15)
plt.tight_layout()

# --- Save and show ---
plt.savefig("cost_vs_aratio_plot.pdf")
plt.show()
