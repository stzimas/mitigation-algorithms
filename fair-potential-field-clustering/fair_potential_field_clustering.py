import numpy as np
from sklearn.cluster import estimate_bandwidth
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time
from scipy.spatial.distance import pdist


def compute_fairness(points, attributes=None, labels_mapping=None):
    points = np.asarray(points, dtype=np.float64)

    if labels_mapping is None:
        labels_mapping = np.arange(points.shape[0], dtype=np.int32)

    labels_mapping = np.asarray(labels_mapping, dtype=np.int64)

    if labels_mapping.size == 0:
        k = max(0, points.shape[0])
        compact_labels = labels_mapping
    else:
        unique_labels = np.unique(labels_mapping)
        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        compact_labels = np.array([label_to_idx[lbl] for lbl in labels_mapping], dtype=np.int64)
        k = len(unique_labels)

    counts = np.zeros((k, 2), dtype=np.int32)
    imb = np.ones(k, dtype=np.float64)
    majority = np.zeros(k, dtype=np.float64)

    if attributes is None:
        return dict(imb=imb, majority=majority, counts=counts)
    
    attributes = np.asarray(attributes, dtype=np.int32)
    if attributes.shape[0] != labels_mapping.shape[0]:
        raise ValueError("attributes and labels_mapping must have the same length (one attribute per sample)")

    mask_red = (attributes == 0)
    mask_blue = (attributes == 1)

    if np.any(mask_red):
        counts[:, 0] = np.bincount(compact_labels[mask_red], minlength=k)[:k]
    if np.any(mask_blue):
        counts[:, 1] = np.bincount(compact_labels[mask_blue], minlength=k)[:k]

    red = counts[:, 0].astype(np.float64)
    blue = counts[:, 1].astype(np.float64)

    both_pos = (red > 0) & (blue > 0)
    if np.any(both_pos):
        ratio_min = np.minimum(red[both_pos] / blue[both_pos], blue[both_pos] / red[both_pos])
        imb[both_pos] = 1.0 - ratio_min

    majority[red > blue] = 1.0
    majority[blue > red] = -1.0

    return dict(imb=imb, majority=majority, counts=counts)#, cluster_attrs=cluster_attrs)

def compute_forces(points, masses, fairness, h=None, G=1.0, K=1.0, lamda=1.0):
    #points: current points (k, dim)
    #masses: current masses (k,)
    #fairness: dict with 'imb' and 'signs' (per current point)
    #h: interaction radius; None means all-to-all
    #G, K, lamda: coefficients
    #dim_exp: exponent for distance denominator; defaults to dim


    points = np.asarray(points, dtype=np.float64)
    k, dim_exp = points.shape

    masses = np.ones(k, dtype=np.float64) if masses is None else np.asarray(masses, dtype=np.float64)
    imb = fairness.get('imb', np.zeros(k, dtype=np.float64))
    maj = fairness.get('majority', np.zeros(k, dtype=np.float64))

    forces = np.zeros((k, dim_exp), dtype=np.float64)

    if h is None:
        d_vec = points[None, :, :] - points[:, None, :]
        r_sq = np.sum(d_vec ** 2, axis=2)
        r = np.sqrt(r_sq)
        mask = r > 0.0

        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.where(mask, r ** dim_exp, 1.0)

        mprod = masses[:, None] * masses[None, :]
        f_grav_coeff = G * mprod  / denom
        f_grav_coeff = np.where(mask, f_grav_coeff, 0.0)

        maj_i = maj[:, None]
        maj_j = maj[None, :]
        sign_pair = np.where((maj_i == 0) | (maj_j == 0), 0.0, np.where(maj_i == maj_j, -1.0, 1.0))
        imb_i = imb[:, None]
        imb_j = imb[None, :]
        f_elec_coeff = K * imb_i * imb_j * sign_pair / denom
        f_elec_coeff = np.where(mask, f_elec_coeff, 0.0)

        total_coeff = (1.0 - lamda)  * f_grav_coeff + lamda * f_elec_coeff
        forces = np.einsum('ij,ijk->ik', total_coeff, d_vec)
        
    else:
        tree = cKDTree(points)
        neighs = tree.query_ball_point(points, h)
        for i in range(k):
            idxs = [j for j in neighs[i] if j != i]
            if len(idxs) == 0:
                continue
            idxs = np.array(idxs, dtype=np.int32)
            d_vecs = points[idxs] - points[i]
            r = np.linalg.norm(d_vecs, axis=1)
            valid = r > 0
            if not np.any(valid):
                continue
            idxs = idxs[valid]
            d_vecs = d_vecs[valid]
            r = r[valid]

            mp = masses[i] * masses[idxs]
            f_grav_coeff = G * mp / (r ** dim_exp)

            maj_i = maj[i]
            maj_j = maj[idxs]
            sign_local = np.where((maj_i == 0) | (maj_j == 0), 0.0, np.where(maj_i == maj_j, -1.0, 1.0))
            imb_i = imb[i]
            imb_j = imb[idxs]
            f_elec_coeff = K * imb_i * imb_j * sign_local / (r ** dim_exp)
            total_coeff = (1.0 - lamda) * f_grav_coeff + lamda * f_elec_coeff
            forces[i] = np.sum(total_coeff[:, None] * d_vecs, axis=0)

    return forces

def compute_displacements(forces, W=1.0):
    #Delta x_i = W * F_i / ||F_i||^2

    forces = np.asarray(forces, dtype=np.float64)
    normF_sq = np.sum(forces ** 2, axis=1)  # (n,)
    valid = normF_sq > 0.0
    deltas = np.zeros_like(forces)
    deltas[valid] = (W / normF_sq[valid, None]) * forces[valid]
    
    return deltas

def clustering_fairness(labels, attributes, masses=None):
    #ratio = red / (red + blue)
    #balance = min(red/blue, blue/red) (defined as 0 when either count is 0)
    # Returns:
    #ratios: list of per-cluster ratios (red/total)
    #balances: list of per-cluster balances (min(red/blue, blue/red))
    #avg_ratio: weighted by cluster mass (if provided), else simple mean
    #avg_balance: weighted by cluster mass (if provided), else simple mean
    #min_balance: minimum balance across clusters (ignoring None)

    if attributes is None:
        return {'ratios': None, 'avg_ratio': None}
    
    attr_array = np.asarray(attributes, dtype=np.int32)
    n_clusters = int(np.max(labels)) + 1
    
    ratios = []
    balances = []
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        if not np.any(cluster_mask):
            ratios.append(None)
            balances.append(None)
            continue
        
        cluster_attrs = attr_array[cluster_mask]
        red_count = np.sum(cluster_attrs == 0)
        blue_count = np.sum(cluster_attrs == 1)
        total = red_count + blue_count
        
        if total > 0:
            ratio = float(red_count) / float(total)
            ratios.append(ratio)
        else:
            ratios.append(None)
        
        # balance = min(red/blue, blue/red) if either count is 0, balance=0
        if red_count > 0 and blue_count > 0:
            r_over_b = float(red_count) / float(blue_count)
            b_over_r = float(blue_count) / float(red_count)
            balances.append(float(min(r_over_b, b_over_r)))
        else:
            balances.append(0.0)
    
    # Aggregates
    avg_ratio = None
    avg_balance = None
    min_balance = None
    
    # Weighted averages when masses are provided
    if masses is not None:
        valid_ratios = [(ratios[i], masses[i]) for i in range(len(ratios)) if ratios[i] is not None and i < len(masses)]
        valid_balances = [(balances[i], masses[i]) for i in range(len(balances)) if balances[i] is not None and i < len(masses)]
        
        if valid_ratios:
            total_mass = sum(m for _, m in valid_ratios)
            if total_mass > 0:
                avg_ratio = sum(r * m for r, m in valid_ratios) / total_mass
        if valid_balances:
            total_mass_b = sum(m for _, m in valid_balances)
            if total_mass_b > 0:
                avg_balance = sum(b * m for b, m in valid_balances) / total_mass_b
    else:
        # Unweighted means
        valid_ratios = [r for r in ratios if r is not None]
        valid_balances = [b for b in balances if b is not None]
        if valid_ratios:
            avg_ratio = float(np.mean(valid_ratios))
        if valid_balances:
            avg_balance = float(np.mean(valid_balances))
    
    # Minimum balance across clusters (ignore None)
    valid_balances_min = [b for b in balances if b is not None]
    if valid_balances_min:
        min_balance = float(np.min(valid_balances_min))
    
    return {
        'ratios': ratios,
        'balances': balances,
        'avg_ratio': float(avg_ratio) if avg_ratio is not None else None,
        'avg_balance': float(avg_balance) if avg_balance is not None else None,
        'min_balance': float(min_balance) if min_balance is not None else None
    }

def merge_points(points, threshold_merge=None, masses=None):
    # agglomerative BFS merge: points within threshold_merge are in same cluster.

    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    if threshold_merge is None or np.isnan(threshold_merge) or threshold_merge <= 0:
        threshold_merge = 1e-4

    dist_matrix = cdist(points, points, metric='euclidean') #(n,n)
    
    labels = -np.ones(n_points, dtype=int)
    cluster_centers = []
    cluster_masses = []
    current_label = 0

    for i in range(n_points):
        if labels[i] != -1:
            continue
            
        labels[i] = current_label
        cluster_points_idx = [i]
        queue = [i]
        
        while queue:
            idx = queue.pop(0)
            close_points = np.where((labels == -1) & (dist_matrix[idx] < threshold_merge))[0]
            for j in close_points:
                labels[j] = current_label
                cluster_points_idx.append(j)
                queue.append(j)
        
        idx = np.array(cluster_points_idx, dtype=int)
        total_mass = float(np.sum(masses[idx])) if masses is not None else float(len(idx))
        
        if total_mass > 0 and masses is not None:
            cluster_center = np.sum(points[idx] * masses[idx][:, None], axis=0) / total_mass
        else:
            cluster_center = np.mean(points[idx], axis=0)
            
        cluster_centers.append(cluster_center)
        cluster_masses.append(total_mass)
        current_label += 1
        
    return labels, np.array(cluster_centers), np.array(cluster_masses)

def f_pfc(
    points,
    G=1.0,
    max_iter=300,
    W=1.0,
    h=None,
    quantile=0.3,
    min_step=None,
    min_step_multiplier=1e-4,
    min_step_default=1e-6,
    threshold_merge=None,
    threshold_merge_mode=None,
    merge_quantile=0.3,
    merge_factor=1.0,
    K=1.0,
    lamda=1.0,
    attributes=None
):

    t0 = time.perf_counter()
    pts = np.asarray(points, dtype=np.float64)

    params = dict(
        G=float(G),
        max_iter=int(max_iter),
        W=float(W),
        h=h,
        quantile=float(quantile),
        min_step=min_step,
        min_step_multiplier=float(min_step_multiplier),
        min_step_default=float(min_step_default),
        threshold_merge=threshold_merge,
        threshold_merge_mode=str(threshold_merge_mode),
        merge_quantile=float(merge_quantile),
        merge_factor=float(merge_factor),
        K=float(K),
        lamda=float(lamda),
    )
    
    # Set h
    h_used, h_source = set_h(h, pts, quantile)
    min_step_used, min_step_source = set_min_step(min_step, h_used, min_step_multiplier, min_step_default)
    
    history = [pts.copy()]
    cur_pts = pts
    forces = None
    displacements = None
    iter_stats = []
    labels_history = []
    labels_mapping = np.arange(len(pts), dtype=np.int32)
    cur_masses = np.ones(len(pts), dtype=np.float64)

    converged = False
    fair_info = None

    # Main loop
    for it in range(1, max_iter + 1):
        step_t0 = time.perf_counter()
        
        fair_info = compute_fairness(
            points=pts,
            attributes=attributes,
            labels_mapping=labels_mapping,

        )
        forces = compute_forces(
            points=cur_pts,
            masses=cur_masses,
            fairness=fair_info,
            h=h_used,
            G=G,
            K=K,
            lamda=lamda,
        )

        before_displacements = cur_pts.copy()
        before_tuples = [tuple(row) for row in before_displacements]

        displacements = compute_displacements(forces, W=W)
        cur_pts = cur_pts + displacements  # Update in-place
 
        # merge points with min_step threshold
        thr_used, thr_source, merge_labels, cluster_centers, masses_out = set_threshold_merge(
            cur_pts=cur_pts,
            threshold_merge_mode='min_step',
            min_step_used=min_step_used,
            threshold_merge=threshold_merge,
            h_used=h_used,
            merge_quantile=merge_quantile,
            merge_factor=merge_factor,
            masses=cur_masses
        )
        
        # update labels mapping
        labels_mapping = merge_labels[labels_mapping]
        labels_history.append(labels_mapping.copy())
        
        cur_pts = cluster_centers
        cur_masses = masses_out

        after_displacements = cur_pts.copy()
        after_tuples = [tuple(row) for row in after_displacements]

        disp_norms = np.linalg.norm(displacements, axis=1)
        max_disp = float(np.max(disp_norms))
        mean_disp = float(np.mean(disp_norms))
        n_large = int(np.sum(disp_norms >= min_step_used))

        history.append(cur_pts.copy())      
        iter_stats.append({
            'iter': it,
            'max_disp': max_disp,
            'mean_disp': mean_disp,
            'n_large_disp': n_large,
            'time_s': time.perf_counter() - step_t0,
            "thr_used": thr_used,
            "thr_source": thr_source
        })

        before_set = frozenset(before_tuples)
        after_set = frozenset(after_tuples)


        if before_set == after_set:
            converged = "True and same sets"
            n_iters = it
            break
        if max_disp <= min_step_used:
            converged = True
            n_iters = it

            break
    else:
        n_iters = max_iter

    t1 = time.perf_counter()
    duration = t1 - t0
    
    # final merge according to threshold_merge_mode
    if threshold_merge_mode is not None and threshold_merge_mode != 'min_step':
        thr_used_final, thr_source_final, labels_extra, cluster_centers_extra, masses_out_final = set_threshold_merge(
                                cur_pts=cur_pts,
                                threshold_merge_mode=threshold_merge_mode,
                                min_step_used=min_step_used,
                                threshold_merge=threshold_merge,
                                h_used=h_used,
                                merge_quantile=merge_quantile,
                                merge_factor=merge_factor,
                                masses=cur_masses
                            )
                                                                                                                
        # update labels and masses
        final_labels = labels_extra[labels_mapping]
        final_points = cluster_centers_extra
        final_point_labels = labels_extra
        masses_out_final = masses_out_final
    else:
        # no extra merge
        thr_used_final = thr_used
        thr_source_final = thr_source
        final_labels = labels_mapping
        final_points = cur_pts
        final_point_labels = np.arange(len(cur_pts), dtype=np.int32)
        masses_out_final = cur_masses
    
    
    # compute clustering fairness info
    ratio_info = clustering_fairness(final_labels, attributes, masses_out_final)

    result = {
        'final_points': final_points,
        'forces': forces,
        'displacements': displacements,
        'history': history,
        'labels_history': labels_history,
        'labels': final_labels,
        'final_points_labels': final_point_labels,
        'cluster_centers': final_points,
        'params': params,
        'derived': {
            'h_used': h_used,
            'h_source': h_source,
            'min_step_used': min_step_used,
            'min_step_source': min_step_source,
            'threshold_merge_used': thr_used,
            'threshold_merge_source': thr_source,
            'Work': W,
            'cluster_size_final': len(final_points),
            'clusters_size_before_merge': len(cur_pts),
            'masses_out': masses_out_final,
            'final_ratios': ratio_info['ratios'],
            'final_avg_ratio': ratio_info['avg_ratio'],
            'final_avg_balance': ratio_info['avg_balance'],
            'final_min_balance': ratio_info['min_balance'],
            'balances': ratio_info['balances'],
            'extra_merge': {
                "thr_used_final": thr_used_final,
                "thr_source_final": thr_source_final,
            }
        },
        'convergence': {
            'converged': str(converged),
            'n_iters': int(n_iters),
            'final_max_disp': float(iter_stats[-1]['max_disp']) if iter_stats else 0.0
        },
        'timing': {
            'start_time_s': t0,
            'end_time_s': t1,
            'duration_s': duration
        },
        'iter_stats': iter_stats
    }

    return result

#######################################################################################################

def set_h(h, pts, quantile):
    h_source = "none"
    if h == "limit_max_q":
        h_est = estimate_bandwidth(pts, quantile=quantile)
        h_used = float(h_est) if (h_est is not None and not np.isnan(h_est)) else None
        h_source = f"estimated with quantile max {quantile}"
    elif h == "limit_avg_q":
        h_est = estimate_bandwidth_avg(pts, quantile=quantile)
        h_used = float(h_est) if (h_est is not None and not np.isnan(h_est)) else None
        h_source = f"estimated with quantile avg {quantile}"
    elif h is None:
        h_used = None
        h_source = "there is no limit in points interaction"
    else:
        h_used = float(h)
        h_source = "manual"
    return h_used, h_source

def set_min_step(min_step, h_used, min_step_multiplier, min_step_default):
    if min_step is not None:
        min_step_used = float(min_step)
        min_step_source = f"manual value provided {min_step}"
    else:
        if h_used is not None:
            min_step_used = float(h_used) * float(min_step_multiplier)
            min_step_source = f"derived_from_h with min_step_multiplier {min_step_multiplier}"
        else:
            min_step_used = float(min_step_default)
            min_step_source = f"default value min_step_default {min_step_default}"
    return min_step_used, min_step_source

def knn_bandwidth(X, k=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return distances[:, 1:].mean()

def estimate_bandwidth_avg(X, quantile=0.3):
    X = np.asarray(X)
    n = len(X)
    if n < 2:
        return 1.0

    dists = pdist(X, metric="euclidean")
    cutoff = np.quantile(dists, quantile)
    selected = dists[dists <= cutoff]

    if len(selected) == 0:
        return cutoff
    
    return np.mean(selected)

def set_threshold_merge(cur_pts, threshold_merge_mode, min_step_used=None, threshold_merge=None,
                       h_used=None, merge_quantile=None, merge_factor=1, k=None, merge_mass=False, masses=None):
    thr_used = None
    thr_source = None

    if threshold_merge_mode == 'manual':
        thr_used = float(threshold_merge)
        thr_source = f"manual value provided{threshold_merge}"
        labels, cluster_centers, masses_out = merge_points(cur_pts, threshold_merge=thr_used, masses=masses)
    
    elif threshold_merge_mode == 'min_step':
        thr_used = min_step_used
        thr_source = f"same previous min_step used {min_step_used}"
        labels, cluster_centers, masses_out = merge_points(cur_pts, threshold_merge=thr_used, masses=masses)

    elif threshold_merge_mode == 'h':
        if h_used is None:
            h_used = estimate_bandwidth(cur_pts, quantile=merge_quantile)
        thr_used = float(h_used) * float(merge_factor)
        thr_source = f"same previous h used as first points multiplied by merge_factor {merge_factor}"
        labels, cluster_centers, masses_out = merge_points(cur_pts, threshold_merge=thr_used, masses=masses)

    elif threshold_merge_mode == 'final_quantile_max':
        thr = estimate_bandwidth(cur_pts, quantile=merge_quantile)
        thr_used = float(thr)
        thr_source = f"estimated h on final points with quantile MAX distance{merge_quantile}"
        labels, cluster_centers, masses_out = merge_points(cur_pts, threshold_merge=thr_used, masses=masses)

    elif threshold_merge_mode == 'final_quantile_avg':
        thr = estimate_bandwidth_avg(cur_pts, quantile=merge_quantile)
        thr_used = float(thr)
        thr_source = f"estimated h on final points with quantile AVG distance{merge_quantile}"
        labels, cluster_centers, masses_out = merge_points(cur_pts, threshold_merge=thr_used, masses=masses)

    elif threshold_merge_mode == 'iterating_merge_max':
        labels_1, cluster_centers, masses_1 = merge_points(cur_pts, threshold_merge=min_step_used, masses=masses)
        th = estimate_bandwidth(cluster_centers, quantile=merge_quantile)
        labels_2, cluster_centers, masses_2 = merge_points(cluster_centers, threshold_merge=th, masses=masses_1)
        labels = labels_2[labels_1]
        thr_used = th
        masses_out = masses_2
        thr_source = f"iterative merging with min_step_used and then estimated bandwidth on centers with quantile MAX{merge_quantile}"
    
    elif threshold_merge_mode == 'iterating_merge_avg':
        labels_1, cluster_centers, masses_1 = merge_points(cur_pts, threshold_merge=min_step_used, masses=masses)
        th = estimate_bandwidth_avg(cluster_centers, quantile=merge_quantile)
        labels_2, cluster_centers, masses_2 = merge_points(cluster_centers, threshold_merge=th, masses=masses_1)
        labels = labels_2[labels_1]
        thr_used = th
        masses_out = masses_2
        thr_source = f"iterative merging with min_step_used and then estimated bandwidth on centers with quantile AVG {merge_quantile}"
    
    elif threshold_merge_mode == 'iterating_merge_nn':
        labels_1, cluster_centers, masses_1 = merge_points(cur_pts, threshold_merge=min_step_used, masses=masses)
        th = knn_bandwidth(cluster_centers, k=2)
        labels_2, cluster_centers, masses_2 = merge_points(cluster_centers, threshold_merge=th, masses=masses_1)
        labels = labels_2[labels_1]
        thr_used = th
        masses_out = masses_2
        thr_source = f"iterative merging with min_step_used and then estimated bandwidth on centers with K=2 NN"
    
    else:
        thr_used = 1e-4
        thr_source = f"default threshold for merging{thr_used}"
        labels, cluster_centers, masses_out = merge_points(cur_pts, threshold_merge=thr_used, masses=masses)
    
    return thr_used, thr_source, labels, cluster_centers, masses_out


