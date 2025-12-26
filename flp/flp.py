import random
import networkx as nx
from collections import Counter, defaultdict, deque
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd


def init_balance_data(labeling, G, attribute):
    balance_data = {}
    for node, com in labeling.items():
        node_attr = G.nodes[node][attribute]
        red = 1 if node_attr == 1 else 0
        blue = 1 - red
        balance_data[com] = {
            "red": red,
            "blue": blue,
            "size": 1,
            "balance": 0,
            "more_red": red
        }
    return balance_data


def find_balance(red, blue):
    if red > 0 and blue > 0:
        return (blue / red, 1) if red >= blue else (red / blue, 0)
    return (0, 1) if red > 0 else (0, 0)


def update_balance_data(operation, balance_data, com, node_attr):
    if operation == "addition":
        if com not in balance_data:
            balance_data[com] = {"red": 0, "blue": 0, "size": 0} 

        balance_data[com]["red"] += 1 if node_attr == 1 else 0
        balance_data[com]["blue"] += 1 if node_attr == 0 else 0
        balance_data[com]["size"] += 1
    elif operation == "reduction":
        balance_data[com]["red"] -= 1 if node_attr == 1 else 0
        balance_data[com]["blue"] -= 1 if node_attr == 0 else 0
        balance_data[com]["size"] -= 1

    red, blue = balance_data[com]["red"], balance_data[com]["blue"]
    balance, more_red = find_balance(red, blue)
    balance_data[com]["balance"] = balance
    balance_data[com]["more_red"] = more_red
    if balance_data[com]["size"] == 0:
        balance_data.pop(com)


def delta(node, com, balance_data, G, attribute):
    node_attr = G.nodes[node][attribute]
    return -1 if node_attr == balance_data[com]["more_red"] else 1


def find_coms(labeling):
    clusters = defaultdict(set)
    for node, label in labeling.items():
        clusters[label].add(node)
    coms = frozenset(frozenset(nodes) for nodes in clusters.values())
    return coms


def find_coms_dict(labeling):
    clusters = defaultdict(set)
    for node, label in labeling.items():
        clusters[label].add(node)
    return clusters


def fsp(G, k_gravity, k_coul, seed=None, max_iter= 1000):
    attribute = "protected"
    coloring = _color_network(G)
    random.seed(seed)
    nodes = list(G.nodes())
    random_labels = list(range(len(nodes)))
    random.shuffle(random_labels)
    labeling = {node: label for node, label in zip(nodes, random_labels)}
    labeling_cp = labeling 
    balance_data = init_balance_data(labeling_cp, G, attribute)
    iterations = 0
    window_size = 20
    previous_coms = deque(maxlen=window_size)
    nodes = set(G.nodes())

    flag = None

    while not _labeling_complete(labeling, G, balance_data, k_gravity, k_coul):
        iterations += 1
        current_com = find_coms(labeling)
        print(f"iterations {iterations}")
        if current_com in previous_coms:
            flag = "oscillation_{iterations}"
            print(f"coms Oscillation detected at iteration {iterations}!")
            break
       
        if iterations == max_iter:
            flag = "max_iterations_{iterations}"
            break
        previous_coms.append(current_com)

        for color,nodes in coloring.items():
            for node in nodes:
                _update_label(node, labeling, G, balance_data, k_gravity, k_coul, attribute)

    clusters = defaultdict(set)
    for node, label in labeling.items():
        clusters[label].add(node)

    df = pd.DataFrame.from_dict(balance_data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'community'}, inplace=True)
    return clusters.values(), df, flag


def _color_network(G):
    coloring = {}
    colors = nx.coloring.greedy_color(G)
    for node, color in colors.items():
        if color in coloring:
            coloring[color].add(node)
        else:
            coloring[color] = {node}
    return coloring


def _labeling_complete(labeling, G, balance_data, k_gravity, k_coul):
    return all(
        labeling[v] in _best_labels(v, labeling, G, balance_data, k_gravity, k_coul)
        for v in G
        if len(G[v]) > 0)


def _best_labels(node, labeling, G, balance_data, k_gravity, k_coul):
    if not G[node]:
        return {labeling[node]}
    freqs = Counter(labeling[neighbor] for neighbor in G[node])
    metrics = {}
    for com, n_cx in freqs.items():

        if(labeling[node]==com):
            node_attr = G.nodes[node]["protected"]
            update_balance_data("reduction", balance_data, labeling[node], node_attr)
            delta = delta(node, com, balance_data, G, "protected")
            imbalance_c = 1 - balance_data[com]["balance"]
            update_balance_data("addition", balance_data, labeling[node], node_attr)
        else: 
            delta = delta(node, com, balance_data, G, "protected")
            imbalance_c = 1 - balance_data[com]["balance"]
        metrics[com] = n_cx * (k_gravity + k_coul * delta * imbalance_c)

    max_value = max(metrics.values())
    return {label for label, value in metrics.items() if value == max_value}


def _update_label(node, labeling, G, balance_data, k_gravity, k_coul, attribute):
    best_labels = _best_labels(node, labeling, G, balance_data, k_gravity, k_coul)
    current_label = labeling[node]
    node_attr = G.nodes[node][attribute]
    if len(best_labels) == 1:
        update_balance_data("reduction", balance_data, current_label, node_attr)
        labeling[node] = best_labels.pop()
        update_balance_data("addition", balance_data, labeling[node], node_attr)
    elif len(best_labels) > 1 and current_label not in best_labels:
        update_balance_data("reduction", balance_data, current_label, node_attr)
        labeling[node] = max(best_labels)
        update_balance_data("addition", balance_data, labeling[node], node_attr)


def show_graph(G, labeling, iterations):
    shapes = {0: "^", 1: "o"}
    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True) 
    pos = nx.spring_layout(G, seed=42, k=0.4 )
    unique_labels = set(labeling.values())
    color_map = {label: plt.cm.tab10(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

    for protected_value, marker in shapes.items():
        node_list = [node for node in G.nodes if G.nodes[node]['protected'] == protected_value]
        node_colors = [color_map[labeling[node]] for node in node_list] 
        nx.draw_networkx(G, pos, nodelist=node_list, node_color=node_colors,
                           node_shape=marker, node_size=500, edgecolors="black")
    
    plt.savefig(f"{save_dir}/graph{iterations}.png", dpi=300, bbox_inches='tight')
    plt.show()
###