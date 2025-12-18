# ==================================== #
# Make KDE Grid Plots of all RW graphs #
# ==================================== #

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse

# --- Configuration --- #
base_data_path = "Real_World_Data"
base_result_path = "Real_World_Results/a_primes"

graph_names = [
    "karate", "residence", "twitter",
    "blogs", "trinity", "facebook", "northwestern"
]

algorithms = [
    ("constructive_random", "Se-Rand"),
    ("greedy", "Se-Greedy"),
    ("constructive_sm", "Se-SM"),
    ("linear_sol_neum", "GA-NMA"),
    ("linear_sol_sm", "GA-SM")
]


# --- Set up plot grid --- #
fig, axes = plt.subplots(
    nrows=len(graph_names),
    ncols=len(algorithms),
    figsize=(5 * len(algorithms), 2.5 * len(graph_names)),
    sharey='row'
)

# --- Plotting loop --- #
for row_idx, graph_name in enumerate(graph_names):
    initial_a_path = os.path.join(base_data_path, graph_name, "A_uniform.npz")

    if not os.path.exists(initial_a_path):
        for ax in axes[row_idx]: ax.axis("off")
        continue

    A_initial_sparse = scipy.sparse.load_npz(initial_a_path)
    a_initial = A_initial_sparse.diagonal()

    for col_idx, (algo_key, algo_label) in enumerate(algorithms):
        ax = axes[row_idx, col_idx]
        a_prime_file = f"{graph_name}__{algo_key}.npy"
        a_prime_path = os.path.join(base_result_path, a_prime_file)

        if not os.path.exists(a_prime_path):
            ax.axis("off")
            continue

        a_prime = np.load(a_prime_path)

        # Plot KDEs with smoothness and limited x-axis
        sns.kdeplot(a_initial, ax=ax, color='blue', fill=True, alpha=0.3, bw_adjust=1, clip=(0, 1))
        sns.kdeplot(a_prime, ax=ax, color='red', fill=True, alpha=0.3, bw_adjust=1, clip=(0, 1))

        ax.set_xlim(0, 1)

        # Set titles for top row
        if row_idx == 0:
            ax.set_title(algo_label, fontsize=33, pad=10)  # HERE SIZE of alg names

        # Keep bold graph name only on first column
        if col_idx == 0:
            ax.set_ylabel(graph_name.title(), fontsize=22, fontweight='bold', labelpad=8)
        else:
            ax.set_ylabel("")
        
        # Hide y-axis ticks and tick labels (for all plots)
        ax.tick_params(left=False, labelleft=False)

        # Show x-axis label ONLY a , only at the bottom row
        if row_idx == len(graph_names) - 1:
            ax.set_xlabel("Stubbornness (a)", fontsize=18)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)  # Hide x-axis ticks and tick labels

        ax.grid(True)

# --- Add shared legend on top --- #
blue_patch = plt.Line2D([0], [0], color='blue', lw=4, label='Initial a')
red_patch = plt.Line2D([0], [0], color='red', lw=4, label="Final a'")
fig.legend(handles=[blue_patch, red_patch], loc='upper center', ncol=2, fontsize=24)  # HERE SIZE of legend

# --- Adjust layout --- #
plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = "Combined_KDE_Distributions_METIS_FINAL.pdf"
plt.savefig(output_path, bbox_inches='tight')
plt.close()

output_path


# ==================================================================== #
# Make Histograms and Scatter Plots for Degree corelation of RW graphs #
# ==================================================================== #

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import matplotlib.ticker as ticker
from scipy.stats import pearsonr 

# --- Configuration --- #
algorithms = [
    ("constructive_random", "Se-Rand"),
    ("greedy", "Se-Greedy"),
    ("constructive_sm", "Se-SM"),
    ("linear_sol_neum", "GA-NMA"),
    ("linear_sol_sm", "GA-SM")
]
base_data_path = "Real_World_Data"
base_result_path = "Real_World_Results/a_primes"
graph_names = [
    "karate", "residence", "twitter",
    "blogs", "trinity", "facebook", "northwestern"
]

# --- Set up grid for all graphs and algorithms --- #
fig, axes = plt.subplots(
    nrows=len(graph_names),
    ncols=len(algorithms),
    figsize=(5 * len(algorithms), 2.5 * len(graph_names)),
    sharex=False,
    sharey=False
)

for row_idx, graph_name in enumerate(graph_names):
    print(f"Processing graph: {graph_name}")

    W_path = os.path.join(base_data_path, graph_name, "W.npz")
    A_path = os.path.join(base_data_path, graph_name, "A_uniform.npz")

    if not os.path.exists(W_path) or not os.path.exists(A_path):
        for ax in axes[row_idx]: ax.axis("off")
        continue

    W = scipy.sparse.load_npz(W_path).tocsr()
    A = scipy.sparse.load_npz(A_path)
    degrees = np.diff(W.indptr)
    a_initial = A.diagonal()

    for col_idx, (algo_key, algo_label) in enumerate(algorithms):
        ax = axes[row_idx, col_idx]
        a_prime_file = f"{graph_name}__{algo_key}.npy"
        a_prime_path = os.path.join(base_result_path, a_prime_file)

        if not os.path.exists(a_prime_path):
            ax.axis("off")
            continue

        a_prime = np.load(a_prime_path)

        if col_idx < 3:  # Changed from 4 to 2 based on where scatter plots are
            # Constructive algorithm — histogram of degrees of changed nodes
            changed_mask = a_initial != a_prime
            changed_degrees = degrees[changed_mask]

            ax.hist(changed_degrees, bins=40, color='skyblue', edgecolor='black')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Integer-only y ticks

            if col_idx == 0:
                ax.set_ylabel(r"$\mathbf{" + graph_name.title() + r"}$" + "\nNode Count", fontsize=20)
        else:
            # SL algorithm — scatter of |a_i - a_i'| vs degree
            delta_abs = np.abs(a_initial - a_prime)

            ax.scatter(degrees, delta_abs, alpha=0.6, s=10, rasterized=True)

            # Compute and annotate Pearson correlation and p-value
            if len(degrees) > 1 and np.std(degrees) > 0 and np.std(delta_abs) > 0:
                corr, p_val = pearsonr(degrees, delta_abs)
                ax.annotate(f"$r$ = {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=25, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.8))

                sig_text = "SIGNIFICANT" if p_val < 0.05 else "not significant"
                print(f"{graph_name} - {algo_label}: r = {corr:.3f}, p = {p_val:.4g} → {sig_text}")

            if col_idx == 3: # Changed from 4 to 2 based on where scatter plots are
                ax.set_ylabel("|aᵢ - aᵢ′|", fontsize=18)

        if row_idx == 0:
            ax.set_title(algo_label, fontsize=33, pad=10)

        if row_idx == len(graph_names) - 1:
            ax.set_xlabel("Node Degree", fontsize=20)
        else:
            ax.set_xlabel("")

        ax.grid(True)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = "Combined_Degree_METIS_FINAL.pdf"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Figure saved to: {output_path}")


# ====================================================================== #
# Make Histograms and Scatter Plots for InitialA corelation of RW graphs #
# ====================================================================== #

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import matplotlib.ticker as ticker
from scipy.stats import pearsonr  # <-- For correlation + p-value

# --- Configuration --- #
algorithms = [
    ("constructive_random", "Se-Rand"),
    ("greedy", "Se-Greedy"),
    ("constructive_sm", "Se-SM"),
    ("linear_sol_neum", "GA-NMA"),
    ("linear_sol_sm", "GA-SM")
]
base_data_path = "Real_World_Data"
base_result_path = "Real_World_Results/a_primes"
graph_names = [
    "karate", "residence", "twitter",
    "blogs", "trinity", "facebook", "northwestern"
]

# --- Set up grid for all graphs and algorithms ---
fig, axes = plt.subplots(
    nrows=len(graph_names),
    ncols=len(algorithms),
    figsize=(5 * len(algorithms), 2.5 * len(graph_names)),
    sharex=False,
    sharey=False
)

for row_idx, graph_name in enumerate(graph_names):
    print(f"Processing graph: {graph_name}")

    W_path = os.path.join(base_data_path, graph_name, "W.npz")
    A_path = os.path.join(base_data_path, graph_name, "A_uniform.npz")

    if not os.path.exists(W_path) or not os.path.exists(A_path):
        for ax in axes[row_idx]:
            ax.axis("off")
        continue

    W = scipy.sparse.load_npz(W_path).tocsr()
    A = scipy.sparse.load_npz(A_path)
    a_initial = A.diagonal()

    for col_idx, (algo_key, algo_label) in enumerate(algorithms):
        ax = axes[row_idx, col_idx]
        a_prime_file = f"{graph_name}__{algo_key}.npy"
        a_prime_path = os.path.join(base_result_path, a_prime_file)

        if not os.path.exists(a_prime_path):
            ax.axis("off")
            continue

        a_prime = np.load(a_prime_path)

        if col_idx < 3: # Changed from 4 to 2 based on where scatter plots are
            # Histogram of initial a_i for changed nodes
            changed_mask = a_initial != a_prime
            changed_ainit = a_initial[changed_mask]

            ax.hist(changed_ainit, bins=40, color='lightcoral', edgecolor='black')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            if col_idx == 0:
                ax.set_ylabel(r"$\mathbf{" + graph_name.title() + r"}$" + "\nNode Count", fontsize=20)

        else:
            # Scatter plot: |a_i - a_i'| vs initial a_i
            delta_abs = np.abs(a_initial - a_prime)
            ax.scatter(a_initial, delta_abs, alpha=0.6, s=10, rasterized=True)

            # Pearson correlation and p-value
            if len(a_initial) > 1 and np.std(a_initial) > 0 and np.std(delta_abs) > 0:
                corr, p_val = pearsonr(a_initial, delta_abs)
                ax.annotate(f"$r$ = {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=25, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.8))

                sig_text = "SIGNIFICANT" if p_val < 0.05 else "not significant"
                print(f"{graph_name} - {algo_label}: r = {corr:.3f}, p = {p_val:.4g} → {sig_text}")

            if col_idx == 3: # Changed from 4 to 2 based on where scatter plots are
                ax.set_ylabel("|aᵢ - aᵢ′|", fontsize=18)

        if row_idx == 0:
            ax.set_title(algo_label, fontsize=33, pad=10)

        if row_idx == len(graph_names) - 1:
            ax.set_xlabel("Initial $a_i$", fontsize=20)
        else:
            ax.set_xlabel("")

        ax.grid(True)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = "InitialA_Correlation_NoSel_METIS_FINAL.pdf"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Figure saved to: {output_path}")





