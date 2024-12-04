import numpy as np
from collections import Counter
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score


def viterbi(observations, num_states, transition_prob, emission_prob):
    V = np.zeros((num_states, len(observations)))
    path = {}
    # first column of V is the transition prob from state 0
    V[:, 0] = transition_prob[0, :]

    for t in range(1, len(observations)):
        for s in range(num_states):
            prob = V[:, t - 1] * transition_prob[:, s] * emission_prob[s - 1, observations[t]]
            V[s, t] = np.max(prob)
            path[s, t] = np.argmax(prob)

    optimal_path = []
    last_state = np.argmax(V[:, -1])
    optimal_path.append(last_state)

    for t in range(len(observations) - 1, 1, -1):
        last_state = path[last_state, t]
        optimal_path.insert(0, last_state)

    optimal_path.insert(0, 0)

    return optimal_path


def calculate_entropy(cluster):
    """Calculate the entropy of a clustering."""
    total_points = len(cluster)
    if total_points == 0:
        return 0
    label_counts = Counter(cluster)
    probabilities = [count / total_points for count in label_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy


def calculate_v_measure(true_labels, predicted_labels):
    homo_score = homogeneity_score(true_labels, predicted_labels)
    comp_score = completeness_score(true_labels, predicted_labels)
    v_score = v_measure_score(true_labels, predicted_labels)
    print(adjusted_mutual_info_score(true_labels, predicted_labels))
    return homo_score, comp_score, v_score

