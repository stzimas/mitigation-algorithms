import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import cupy as cp

def exact_Q(A, W):
    """
    Compute Q = B^-1 * A where B = I - (I - A)W, using sparce matrix operations
    """
    N = A.shape[0]
    I = sp.identity(N, format='csr')
    B = I - (I - A) @ W
    B_csc = B.tocsc()

    # LU factorization is faster than solving from scratch
    lu = spla.splu(B_csc)
    Q = lu.solve(A.toarray())
    #Q = spla.spsolve(B_csc, A)   # Alternative: Q = spla.spsolve(B_csc, A) # Slower

    return Q

def exact_Q_gpu(A_sparse, W_sparse, precision='float32'):
    """
    Compute Q = B^-1 * A on the GPU using dense matrix algebra.
    Converts sparse matrices to dense CuPy arrays and solves B Q = A.

    Parameters:
    - A_sparse: scipy sparse diagonal matrix (n x n)
    - W_sparse: scipy sparse matrix (n x n)
    - precision: 'float32' (default)

    Returns:
    - Q: NumPy array (n x n), solution to B Q = A
    """
    n = A_sparse.shape[0]

    # Choose dtype
    dtype = cp.float32 if precision == 'float32' else cp.float64

    # Construct B = I - (I - A)W
    I_sparse = sp.identity(n, format='csr')
    B_sparse = I_sparse - (I_sparse - A_sparse) @ W_sparse

    # Convert to dense CuPy arrays
    B_dense = cp.array(B_sparse.toarray(), dtype=dtype)
    A_dense = cp.array(A_sparse.toarray(), dtype=dtype)

    # Solve on GPU
    Q_gpu = cp.linalg.solve(B_dense, A_dense)

    # Move back to CPU
    return cp.asnumpy(Q_gpu)

def calc_QR(Q, labels):
    """
    Calculate the red team influence Q_R = sum of influences of red nodes divided by n
    """
    n = Q.shape[0]
    red_mask = labels == 0
    Q_R = Q[:, red_mask].sum() / n
    return Q_R

def calculate_c_i_exact(a_prime, Q, labels, epsilon=1e-6):
    """
    Calculate exact gradients (c_i values) for linearized optimization using Sherman-Morrison
    """
    red_mask = labels == 0
    blue_mask = labels == 1

    c = np.zeros_like(a_prime)

    Q_i = np.array(Q.sum(axis=0)).flatten()
    a_one_minus_a = a_prime * (1 - a_prime)

    numerator_sum_red = Q[:, red_mask].sum(axis=1)
    numerator_sum_red = np.array(numerator_sum_red).flatten()

    c = Q_i / (a_one_minus_a + epsilon)

    c[blue_mask] = -c[blue_mask] * numerator_sum_red[blue_mask]
    c[red_mask] = c[red_mask] * (1 - numerator_sum_red[red_mask])
    
    return c

def calculate_c_i_neum(W, a_prime, labels):
    """
    Calculate approximate gradients (c_i values) for linearized optimization using Neumann Series Approximation
    """
    # Ensure W is in CSR format for slicing
    if not sp.isspmatrix_csr(W):              # TODO : Monkey fix, maybe can be implemented Faster
        W = W.tocsr()

    red_mask = labels == 0
    blue_mask = labels == 1

    c = np.zeros_like(a_prime)
    c[red_mask] = W[red_mask][:, blue_mask] @ a_prime[blue_mask]
    c[blue_mask] = - (W[blue_mask][:, red_mask] @ a_prime[red_mask])

    return c


def linear_sol_sm(W, A, labels, phi, tolerance=1e-3):
    """
    Linearized optimization algorithm to find a' values such that Q_R ≈ phi
    """

    n = len(labels)
    a_original = A.diagonal()
    a_prime = np.copy(a_original)

    iter = 0
    start_time = time.time()

    while True:
        iter += 1
        Q = exact_Q_gpu(A, W) 
        Q_R = calc_QR(Q, labels)

        delta_Q_R = abs(Q_R - phi)
        if delta_Q_R < tolerance:
            print(f"Converged at iteration {iter} with |Q_R - phi| = {delta_Q_R:.6f}")
            break

        c = calculate_c_i_exact(a_prime, Q, labels)
        denominator_sum = np.sum(c**2)
        ratio = (phi - Q_R) / denominator_sum
        a_prime += ratio * c

        a_prime = np.clip(a_prime, 1e-4, 1 - 1e-4)
        A = sp.diags(a_prime, format='csr')

        # print(
        #     f"[Iter {iter:>3}] Q_R = {Q_R:.6f} | "
        #     f"Delta = {delta_Q_R:.6e} | "
        #     f"Objective = {np.sum((a_prime - a_original)**2):.6f}"
        # )

    elapsed_time = time.time() - start_time
    objective_score = np.sum((a_prime - a_original) ** 2)
    final_Q_R = Q_R
    delta_Q_R = abs(Q_R - phi)

    print(f"\n Run Completed")
    print(f"Actual Final Q_R = {final_Q_R:.6f}, Target phi = {phi:.6f}, Difference = {delta_Q_R:.6f}")
    print(f"Objective Score = {objective_score:.6f}")
    print(f"Elapsed Time = {elapsed_time:.6f} seconds")
    print(f"Total Iterations = {iter}\n")

    return a_prime, objective_score, elapsed_time, iter, delta_Q_R


def linear_sol_neum(W, A, labels, phi, tolerance=1e-3):
    """
    Linearized optimization algorithm to find a' values such that Q_R ≈ phi
    """

    n = len(labels)
    a_original = A.diagonal()
    a_prime = np.copy(a_original)

    iter = 0
    start_time = time.time()

    while True:
        iter += 1
        Q = exact_Q_gpu(A, W)
        Q_R = calc_QR(Q, labels)

        delta_Q_R = abs(Q_R - phi)
        if delta_Q_R < tolerance:
            print(f"Converged at iteration {iter} with |Q_R - phi| = {delta_Q_R:.6f}")
            break

        c = calculate_c_i_neum(W, a_prime, labels)
        denominator_sum = np.sum(c**2)
        ratio = (phi - Q_R) / denominator_sum
        a_prime += ratio * c

        a_prime = np.clip(a_prime, 1e-4, 1 - 1e-4)
        A = sp.diags(a_prime, format='csr')

        print(
            f"[Iter {iter:>3}] Q_R = {Q_R:.6f} | "
            f"Delta = {delta_Q_R:.6e} | "
            f"Objective = {np.sum((a_prime - a_original)**2):.6f}"
        )

    elapsed_time = time.time() - start_time
    objective_score = np.sum((a_prime - a_original) ** 2)
    final_Q_R = Q_R
    delta_Q_R = abs(Q_R - phi)

    # print(f"\n Run Completed")
    # print(f"Actual Final Q_R = {final_Q_R:.6f}, Target phi = {phi:.6f}, Difference = {delta_Q_R:.6f}")
    # print(f"Objective Score = {objective_score:.6f}")
    # print(f"Elapsed Time = {elapsed_time:.6f} seconds")
    # print(f"Total Iterations = {iter}\n")

    return a_prime, objective_score, elapsed_time, iter, delta_Q_R


# ==================== #
# Selective Algorithms #
# ==================== #

def constructive_random(W, A, labels, phi, epsilon=1e-8):

    print(f"\n🚀 Running Constructive Random...\n")

    n = len(labels)
    a_original = A.diagonal()    # Original resistance values
    a_prime = a_original.copy()  # Copy initial resistance values

    Q = exact_Q_gpu(A, W)     # Compute initial Q
    Q_R = calc_QR(Q, labels)  # Compute initial Q_R

    V_avail = np.arange(n).tolist()  # Nodes available for adjustment

    print(f"Initial Q_R: {Q_R:.6f}, Target φ: {phi:.6f}")

    if np.abs(Q_R - phi) < epsilon:
        print("✅ Already Fair: No Changes Needed")
        return a_prime, True  # Already fair, return early

    # Initialize iteration counter
    iter = 0

    # Start timing
    start_time = time.time()
    
    # =============================== #
    # Case 1: Increase Red Influence  #
    # =============================== #
    if Q_R < phi:
        print("🔺 Increasing Red Influence (Q_R lower than Target)")
        while Q_R < phi:
            i = np.random.choice(V_avail)  # Randomly select node
            V_avail.remove(i)  # Remove it from available nodes

            if labels[i] == 0:  # If Red node
                a_prime[i] = 1 - epsilon  # Maximize resistance
            else:  # If Blue node
                a_prime[i] = epsilon  # Minimize resistance
            
            # Recompute Q and Q_R
            A = sp.diags(a_prime, format='csr')  # Rebuild A
            Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
            Q_R = calc_QR(Q, labels)

            # Increment iteration counter
            iter += 1  

    # =============================== #
    # Case 2: Decrease Red Influence  #
    # =============================== #
    elif Q_R > phi:
        print("🔻 Decreasing Red Influence (Q_R higher than Target)")
        while Q_R > phi:
            i = np.random.choice(V_avail)  # Randomly select node
            V_avail.remove(i)  # Remove it from available nodes

            if labels[i] == 0:  # If Red node
                a_prime[i] = epsilon  # Minimize resistance
            else:  # If Blue node
                a_prime[i] = 1 - epsilon  # Maximize resistance

            # Recompute Q and Q_R
            A = sp.diags(a_prime, format='csr')  # Rebuild A
            Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
            Q_R = calc_QR(Q, labels)

            # Increment iteration counter
            iter += 1 


    # =========================================== #
    # Undo Last Change and Solve for Precise a_i' #
    # =========================================== #

    a_prime[i] = a_original[i]  # Undo last change

    # Recompute Q and Q_R
    A = sp.diags(a_prime, format='csr')  # Rebuild A
    Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
    Q_R = calc_QR(Q, labels)

    q_ii = Q[i, i]  # Self Influence
    Q_i = np.sum(Q[:, i])/n  # Influence of node i

    # Do final precise altering of final a_i
    if labels[i] == 1:  # If Blue
        a_prime[i] *= ((Q_R - phi) * (1 - a_prime[i])) / (Q_i * np.sum(Q[i, labels == 0]) - (Q_R - phi) * (q_ii - a_prime[i]) + epsilon) + 1
    else:  # If Red
        a_prime[i] *= ((phi - Q_R) * (1 - a_prime[i])) / (Q_i * np.sum(Q[i, labels == 1]) - (phi - Q_R) * (q_ii - a_prime[i]) + epsilon) + 1
    
    # Time passed
    elapsed_time = time.time() - start_time
    
    # Calculate Objective Score
    objective_score = np.sum((a_prime - a_original) ** 2)

    # Compute final Q_R
    A = sp.diags(a_prime, format='csr')  # Rebuild A
    Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
    Q_R = calc_QR(Q, labels)

    print(f"\n🔍 Run Completed")
    print(f"  Final Q_R       = {Q_R:.6f}")
    print(f"  Target phi      = {phi:.6f}")
    print(f"  Objective Score = {objective_score:.6f}")
    print(f"  Elapsed Time    = {elapsed_time:.3f}s")
    print(f"  Total Iterations= {iter}\n")

    return a_prime, objective_score, elapsed_time, iter


def constructive_neuman(W, A, labels, phi, epsilon=1e-8):

    print(f"\n🚀 Running Constructive Neuman...\n")

    n = len(labels)
    a_original = A.diagonal()    # Original resistance values
    a_prime = a_original.copy()  # Copy initial resistance values

    Q = exact_Q_gpu(A, W)     # Compute initial Q
    Q_R = calc_QR(Q, labels)  # Compute initial Q_R

    V_avail = np.arange(n).tolist()  # Nodes available for adjustment

    print(f"Initial Q_R: {Q_R:.6f}, Target φ: {phi:.6f}")

    if np.abs(Q_R - phi) < epsilon:
        print("✅ Already Fair: No Changes Needed")
        return a_prime, True  # Already fair, return early
    
    # Initialize iteration counter
    iter = 0

    # Start timing
    start_time = time.time()
    
    # =============================== #
    # Case 1: Increase Red Influence  #
    # =============================== #
    if Q_R < phi:
        print("🔺 Increasing Red Influence (Q_R lower than Target)")
        while Q_R < phi:
            c_i = calculate_c_i_neum(W, a_prime, labels)  # Compute c_i values
            i = max(V_avail, key=lambda idx: abs(c_i[idx]))  # Select node with max |c_i|
            V_avail.remove(i)  # Remove it from available nodes

            if labels[i] == 0:  # If Red node
                a_prime[i] = 1 - epsilon  # Maximize resistance
            else:  # If Blue node
                a_prime[i] = epsilon  # Minimize resistance
 
            # Recompute Q and Q_R
            A = sp.diags(a_prime, format='csr')  # Rebuild A
            Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
            Q_R = calc_QR(Q, labels)

            # Increment iteration counter
            iter += 1 

    # =============================== #
    # Case 2: Decrease Red Influence  #
    # =============================== #
    elif Q_R > phi:
        print("🔻 Decreasing Red Influence (Q_R higher than Target)")
        while Q_R > phi:
            c_i = calculate_c_i_neum(W, a_prime, labels)  # Compute c_i values
            i = max(V_avail, key=lambda idx: abs(c_i[idx]))  # Select node with max |c_i|
            V_avail.remove(i)  # Remove it from available nodes

            if labels[i] == 0:  # If Red node
                a_prime[i] = epsilon  # Minimize resistance
            else:  # If Blue node
                a_prime[i] = 1 - epsilon  # Maximize resistance

            # Recompute Q and Q_R
            A = sp.diags(a_prime, format='csr')  # Rebuild A
            Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
            Q_R = calc_QR(Q, labels)

            # Increment iteration counter
            iter += 1 

    # =========================================== #
    # Undo Last Change and Solve for Precise a_i' #
    # =========================================== #

    a_prime[i] = a_original[i]  # Undo last change

    # Recompute Q and Q_R
    A = sp.diags(a_prime, format='csr')  # Rebuild A
    Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
    Q_R = calc_QR(Q, labels)

    q_ii = Q[i, i]  # Self Influence
    Q_i = np.sum(Q[:, i])/n  # Influence of node i

    # Do final precise altering of final a_i
    if labels[i] == 1:  # If Blue
        a_prime[i] *= ((Q_R - phi) * (1 - a_prime[i])) / (Q_i * np.sum(Q[i, labels == 0]) - (Q_R - phi) * (q_ii - a_prime[i]) + epsilon) + 1
    else:  # If Red
        a_prime[i] *= ((phi - Q_R) * (1 - a_prime[i])) / (Q_i * np.sum(Q[i, labels == 1]) - (phi - Q_R) * (q_ii - a_prime[i]) + epsilon) + 1
    
    # Time passed
    elapsed_time = time.time() - start_time
    
    # Calculate Objective Score
    objective_score = np.sum((a_prime - a_original) ** 2)

    # Compute final Q_R
    A = sp.diags(a_prime, format='csr')  # Rebuild A
    Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
    Q_R = calc_QR(Q, labels)

    print(f"\n🔍 Run Completed")
    print(f"  Final Q_R       = {Q_R:.6f}")
    print(f"  Target phi      = {phi:.6f}")
    print(f"  Objective Score = {objective_score:.6f}")
    print(f"  Elapsed Time    = {elapsed_time:.3f}s")
    print(f"  Total Iterations= {iter}\n")

    return a_prime, objective_score, elapsed_time, iter


def constructive_sm(W, A, labels, phi, epsilon=1e-8):

    print(f"\n🚀 Running Constructive SM...\n")

    n = len(labels)
    a_original = A.diagonal()    # Original resistance values
    a_prime = a_original.copy()  # Copy initial resistance values

    Q = exact_Q_gpu(A, W)     # Compute initial Q
    Q_R = calc_QR(Q, labels)  # Compute initial Q_R

    V_avail = np.arange(n).tolist()  # Nodes available for adjustment

    print(f"Initial Q_R: {Q_R:.6f}, Target φ: {phi:.6f}")

    if np.abs(Q_R - phi) < epsilon:
        print("✅ Already Fair: No Changes Needed")
        return a_prime, True  # Already fair, return early
    
    # Initialize iteration counter
    iter = 0

    # Start timing
    start_time = time.time()
    
    # =============================== #
    # Case 1: Increase Red Influence  #
    # =============================== #
    if Q_R < phi:
        print("🔺 Increasing Red Influence (Q_R lower than Target)")
        while Q_R < phi:
            c_i = calculate_c_i_exact(a_prime, Q, labels)  # Compute c_i values
            i = max(V_avail, key=lambda idx: abs(c_i[idx]))  # Select node with max |c_i|
            V_avail.remove(i)  # Remove it from available nodes

            if labels[i] == 0:  # If Red node
                a_prime[i] = 1 - epsilon  # Maximize resistance
            else:  # If Blue node
                a_prime[i] = epsilon  # Minimize resistance
 
            # Recompute Q and Q_R
            A = sp.diags(a_prime, format='csr')  # Rebuild A
            Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
            Q_R = calc_QR(Q, labels)

            # Increment iteration counter
            iter += 1 

    # =============================== #
    # Case 2: Decrease Red Influence  #
    # =============================== #
    elif Q_R > phi:
        print("🔻 Decreasing Red Influence (Q_R higher than Target)")
        while Q_R > phi:
            c_i = calculate_c_i_exact(a_prime, Q, labels)  # Compute c_i values
            i = max(V_avail, key=lambda idx: abs(c_i[idx]))  # Select node with max |c_i|
            V_avail.remove(i)  # Remove it from available nodes

            if labels[i] == 0:  # If Red node
                a_prime[i] = epsilon  # Minimize resistance
            else:  # If Blue node
                a_prime[i] = 1 - epsilon  # Maximize resistance

            # Recompute Q and Q_R
            A = sp.diags(a_prime, format='csr')  # Rebuild A
            Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
            Q_R = calc_QR(Q, labels)

            # Increment iteration counter
            iter += 1 

    # =========================================== #
    # Undo Last Change and Solve for Precise a_i' #
    # =========================================== #

    a_prime[i] = a_original[i]  # Undo last change

    # Recompute Q and Q_R
    A = sp.diags(a_prime, format='csr')  # Rebuild A
    Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
    Q_R = calc_QR(Q, labels)

    q_ii = Q[i, i]  # Self Influence
    Q_i = np.sum(Q[:, i])/n  # Influence of node i

    # Do final precise altering of final a_i
    if labels[i] == 1:  # If Blue
        a_prime[i] *= ((Q_R - phi) * (1 - a_prime[i])) / (Q_i * np.sum(Q[i, labels == 0]) - (Q_R - phi) * (q_ii - a_prime[i]) + epsilon) + 1
    else:  # If Red
        a_prime[i] *= ((phi - Q_R) * (1 - a_prime[i])) / (Q_i * np.sum(Q[i, labels == 1]) - (phi - Q_R) * (q_ii - a_prime[i]) + epsilon) + 1
    
    # Time passed
    elapsed_time = time.time() - start_time
    
    # Calculate Objective Score
    objective_score = np.sum((a_prime - a_original) ** 2)

    # Compute final Q_R
    A = sp.diags(a_prime, format='csr')  # Rebuild A
    Q = exact_Q_gpu(A, W) # New Q computation (CAN BE DONE FASTER WITH EXACT SM FORMULA (only 1 a_i has been altered): TODO)
    Q_R = calc_QR(Q, labels)

    print(f"\n🔍 Run Completed")
    print(f"  Final Q_R       = {Q_R:.6f}")
    print(f"  Target phi      = {phi:.6f}")
    print(f"  Objective Score = {objective_score:.6f}")
    print(f"  Elapsed Time    = {elapsed_time:.3f}s")
    print(f"  Total Iterations= {iter}\n")

    return a_prime, objective_score, elapsed_time, iter


# Greedy Version helpful functions

def calculate_QR_prime_incr(a, Q, labels, phi, epsilon=1e-8):
    """
    Compute Q_R' values if each node i were to be altered to its extreme value (0 or 1).
    """
    n = len(a)
    Q_R = np.sum(Q[labels == 0]) / n
    Q_i_vec = np.sum(Q, axis=0) / n
    q_ii_diag = Q.diagonal()
    QR_prime = np.zeros(n)

    for i in range(n):
        a_i = a[i]
        Q_i = Q_i_vec[i]
        q_ii = q_ii_diag[i]
        if labels[i] == 0:  # Red node: increase influence => set a_i' = 1 - epsilon
            a_i_prime = 1 - epsilon
            sum_qij = np.sum(Q[i, labels == 1])  # sum over Blue
            sign = +1
        else:  # Blue node: decrease influence => set a_i' = epsilon
            a_i_prime = epsilon
            sum_qij = np.sum(Q[i, labels == 0])  # sum over Red
            sign = -1

        delta = a_i_prime - a_i
        denom = 1 + (delta / (a_i * (1 - a_i))) * (q_ii - a_i + epsilon)
        numer = Q_i / (a_i * (1 - a_i)) * sum_qij
        delta_QR = sign * delta * numer / denom

        QR_prime[i] = Q_R + delta_QR

    return QR_prime

def calculate_QR_prime_decr(a, Q, labels, phi, epsilon=1e-8):
    """
    Compute Q_R' values if each node i were to be altered to its extreme value (0 or 1).
    """
    n = len(a)
    Q_R = np.sum(Q[labels == 0]) / n
    Q_i_vec = np.sum(Q, axis=0) / n
    q_ii_diag = Q.diagonal()
    QR_prime = np.zeros(n)
  
    for i in range(n):
        a_i = a[i]
        Q_i = Q_i_vec[i]
        q_ii = q_ii_diag[i]
        if labels[i] == 0:  # Red node: decrease influence => set a_i' = epsilon
            a_i_prime = epsilon
            sum_qij = np.sum(Q[i, labels == 1])  # sum over Blue
            sign = +1
        else:  # Blue node: increase influence => set a_i' = 1 - epsilon
            a_i_prime = 1 - epsilon
            sum_qij = np.sum(Q[i, labels == 0])  # sum over Red
            sign = -1

        delta = a_i_prime - a_i
        denom = 1 + (delta / (a_i * (1 - a_i))) * (q_ii - a_i + epsilon)
        numer = Q_i / (a_i * (1 - a_i)) * sum_qij
        delta_QR = sign * delta * numer / denom

        QR_prime[i] = Q_R + delta_QR

    return QR_prime


def constructive_greedy(W, A, labels, phi, epsilon=1e-8):
    print(f"\n🚀 Running Constructive Greedy...\n")

    n = len(labels)
    a_original = A.diagonal()
    a_prime = a_original.copy()

    Q = exact_Q_gpu(A, W)
    #Q = exact_Q(A, W)
    Q_R = calc_QR(Q, labels)

    V_avail = np.arange(n).tolist()

    print(f"Initial Q_R: {Q_R:.6f}, Target φ: {phi:.6f}")

    if np.abs(Q_R - phi) < epsilon:
        print("✅ Already Fair: No Changes Needed")
        return a_prime, True

    iter = 0
    start_time = time.time()

    if Q_R < phi:
        print("🔺 Increasing Red Influence (Q_R lower than Target)")
        while Q_R < phi:
            QR_prime = calculate_QR_prime_incr(a_prime, Q, labels, phi, epsilon)
            i = min(V_avail, key=lambda idx: abs(QR_prime[idx] - phi))
            V_avail.remove(i)

            a_prime[i] = 1 - epsilon if labels[i] == 0 else epsilon
            A = sp.diags(a_prime, format='csr')
            Q = exact_Q_gpu(A, W)
            #Q = exact_Q(A, W)  
            Q_R = calc_QR(Q, labels)
            iter += 1

    elif Q_R > phi:
        print("🔻 Decreasing Red Influence (Q_R higher than Target)")
        while Q_R > phi:
            QR_prime = calculate_QR_prime_decr(a_prime, Q, labels, phi, epsilon)
            i = min(V_avail, key=lambda idx: abs(QR_prime[idx] - phi))
            V_avail.remove(i)

            a_prime[i] = epsilon if labels[i] == 0 else 1 - epsilon
            A = sp.diags(a_prime, format='csr')
            Q = exact_Q_gpu(A, W)
            #Q = exact_Q(A, W) 
            Q_R = calc_QR(Q, labels)
            iter += 1

    a_prime[i] = a_original[i]
    A = sp.diags(a_prime, format='csr')
    Q = exact_Q_gpu(A, W)
    #Q = exact_Q(A, W) 
    Q_R = calc_QR(Q, labels)

    q_ii = Q[i, i]
    Q_i = np.sum(Q[:, i]) / n

    if labels[i] == 1:
        a_prime[i] *= ((Q_R - phi) * (1 - a_prime[i])) / (
            Q_i * np.sum(Q[i, labels == 0]) - (Q_R - phi) * (q_ii - a_prime[i]) + epsilon) + 1
    else:
        a_prime[i] *= ((phi - Q_R) * (1 - a_prime[i])) / (
            Q_i * np.sum(Q[i, labels == 1]) - (phi - Q_R) * (q_ii - a_prime[i]) + epsilon) + 1

    elapsed_time = time.time() - start_time
    objective_score = np.sum((a_prime - a_original) ** 2)

    A = sp.diags(a_prime, format='csr')
    Q = exact_Q_gpu(A, W)
    #181Q = exact_Q(A, W) 
    Q_R = calc_QR(Q, labels)

    print(f"\n🔍 Run Completed")
    print(f"  Final Q_R       = {Q_R:.6f}")
    print(f"  Target phi      = {phi:.6f}")
    print(f"  Objective Score = {objective_score:.6f}")
    print(f"  Elapsed Time    = {elapsed_time:.3f}s")
    print(f"  Total Iterations= {iter}\n")

    return a_prime, objective_score, elapsed_time, iter


