import numpy as np
from munkres import Munkres
from collections import Counter
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score


def compute_cost(zt, zt_real):
    """
    Input:
        zt: list of predicted hidden states
        zt_real: list of real hidden states
    Output:
        total: total cost
        indexes: list of tuples representing the optimal mapping of states to real states

    This function computes the cost between 2 sets of hidden states
    Uses Munkres to find the optimal assignment
    Then compares the predicted hidden states with the real hidden states 
    
    """

    zt = np.array(zt)
    zt_real = np.array(zt_real)

    all_states = np.unique( np.concatenate((zt, zt_real)) )

    K_use = len(all_states)

    cost_mat = np.zeros((K_use, K_use))

    for ii in range(K_use):  ## real
        for jj in range(K_use):
            cost_mat[ii, jj] = ((np.abs((zt_real == ii) * 1 - (zt == jj) * 1)).sum()) # TODO check

    m = Munkres()
    indexes = m.compute(cost_mat.tolist())

    total = sum([cost_mat[row][column] for row, column in indexes])
    return total, indexes


def euclidean_distance(A, B):
    assert A.shape == B.shape, "two matrices should have the same shape"
    return np.sqrt(np.sum((A - B) ** 2))


def difference(A, B):
    assert len(A) == len(B), "two hidden states set should have the same length"
    miss_sum = 0
    tot_num = 0
    for i in range(len(A)):
        assert len(A[i]) == len(B[i])
        miss_sum += np.sum(np.array(A[i]) != np.array(B[i]))
        tot_num += len(A[i])
    return miss_sum, tot_num


def kl_divergence(P, Q):
    assert P.shape == Q.shape, "two matrices should have the same shape"
    mask = (P != 0) & (Q != 0)
    filtered_P = P[mask]
    filtered_Q = Q[mask]
    return np.sum(filtered_P * np.log(filtered_P / filtered_Q))


def calculate_entropy(cluster):
    """Calculate the entropy of a clustering."""
    total_points = len(cluster)
    if total_points == 0:
        return 0
    label_counts = Counter(cluster)
    probabilities = [count / total_points for count in label_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

def calculate_mutual_information(U, V):
    """Calculate the mutual information between two clusterings."""
    total_points = len(U)
    mutual_info = 0
    U_labels, V_labels = set(U), set(V)
    for u in U_labels:
        for v in V_labels:
            intersection_size = sum(1 for i in range(total_points) if U[i] == u and V[i] == v)
            if intersection_size == 0:
                continue
            p_u = sum(1 for x in U if x == u) / total_points
            p_v = sum(1 for x in V if x == v) / total_points
            p_uv = intersection_size / total_points
            mutual_info += p_uv * np.log2(p_uv / (p_u * p_v))
    return mutual_info

def calculate_variation_of_information(U, V):
    """Calculate the variation of information between two clusterings."""
    entropy_U = calculate_entropy(U)
    entropy_V = calculate_entropy(V)
    mutual_information = calculate_mutual_information(U, V)
    variation_of_information = entropy_U + entropy_V - 2 * mutual_information
    return variation_of_information, variation_of_information / (entropy_U + entropy_V)

def calculate_v_measure(true_labels, predicted_labels):
    homo_score = homogeneity_score(true_labels, predicted_labels)
    comp_score = completeness_score(true_labels, predicted_labels)
    v_score = v_measure_score(true_labels, predicted_labels)
    return homo_score, comp_score, v_score

