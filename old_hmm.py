import os
from typing import List, Dict, Tuple

from numpy import log


def get_transition_probs(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation.
        Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state.
        The table must include probability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """

    # Count the number of times each state appears
    #               Divide by the count of all transitions from that state

    types = set(i for j in hidden_sequences for i in j)
    total_freq = {s: 0 for s in types}
    total_transitions = {(i, j): 0 for i in types for j in types}

    for sequence in hidden_sequences:

        for typeA in types:
            total_freq[typeA] += sequence.count(typeA)

        for i in range(1, len(sequence)):
            total_transitions[(sequence[i - 1], sequence[i])] += 1

    for transition in total_transitions.keys():

        total_transitions[transition] /= total_freq[transition[0]]

    return total_transitions


def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[
    Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation.
     Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state.
      The table must include proability values for all state-observation pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """

    types = set(i for j in hidden_sequences for i in j)
    emissions = set(i for j in observed_sequences for i in j)

    total_freq = {s: 0 for s in types}
    total_freq_emissions = {s: 0 for s in emissions}

    total_transitions = {(i, j): 0 for i in types for j in emissions}

    for index in range(len(hidden_sequences)):

        for sequence, out in zip(hidden_sequences[index], observed_sequences[index]):
            total_transitions[(sequence, out)] += 1

    for sequence in hidden_sequences:
        for typeA in types:
            total_freq[typeA] += sequence.count(typeA)

    for transition in total_transitions.keys():
        total_transitions[transition] /= total_freq[transition[0]]

    return total_transitions


def estimate_hmm(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities. We use 'B' for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]

def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float],
            emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.
    Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """

    transition_probs = {k: log(v) if v > 0 else float("-inf") for k, v in transition_probs.items()}

    emission_probs = {k: log(v) if v > 0 else float("-inf") for k, v in emission_probs.items()}

    states = set(sum(transition_probs.keys(), ()))

    states = list(states - {'B', 'Z'})

    path_probabilities = {key: [] for key in states}

    psi = {('B', -1): None}

    path_probabilities['B'] = [0]

    # Add first element:

    first = observed_sequence[0]

    for stateA in states:  # State A is current state, could be any,

        b = emission_probs[(stateA, first)]
        a = transition_probs[('B', stateA)]  # How to get to state A

        aS = (a + path_probabilities['B'][-1])

        psi[(stateA, 0)] = 'B'

        delta = aS + b
        path_probabilities[stateA].append(delta)

    for i in range(1, len(observed_sequence)):

        current = observed_sequence[i]  # Current observation

        for stateA in states:  # State A is current state

            b = emission_probs[(stateA, current)]

            aS = []

            for stateB in states:  # State B is previous state
                a = transition_probs[(stateB, stateA)]  # How to get to state A

                aS.append(a + b + path_probabilities[stateB][i-1])


            delta = max(aS) # This is non-deterministic

            psi[(stateA, i)] = states[aS.index(max(aS))]  # Choosing previous state
            path_probabilities[stateA].append(delta)

    b = 0
    aS = []

    for stateB in states:  # State B is previous state
        a = transition_probs[(stateB, 'Z')]  # How to get to state A

        aS.append(a + b + path_probabilities[stateB][-1])

    delta = max(aS)

    last = states[aS.index(max(aS))]
    psi[('Z', i + 1)] = last
    path_probabilities['Z'] = [delta]

    path = [last]

    for i in range(len(observed_sequence) - 1, -1, -1):
        temp = psi[path[0], i]
        path = [temp] + path



    return path[1:]

def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state),
        i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    d = confusion_matrix(pred, true)

    TPc = d[0][0]
    FPc = d[0][1]

    return TPc / (FPc + TPc)
def confusion_matrix(predicted_sentiments, actual_sentiments) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """

    pospos = 0
    posneg = 0
    negpos = 0
    negneg = 0

    for j in range(len(predicted_sentiments)):

        for i in range(len(predicted_sentiments[j])):

            if predicted_sentiments[j][i] == actual_sentiments[j][i]:

                if actual_sentiments[j][i] == 1:
                    pospos += 1

                else:
                    negneg += 1

            if predicted_sentiments[j][i] != actual_sentiments[j][i]:

                if actual_sentiments[j][i] == 1:
                    posneg += 1

                else:
                    negpos += 1

    return [[pospos, negpos], [posneg, negneg]]


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """

    d = confusion_matrix(pred, true)

    TPc = d[0][0]
    FN = d[1][0]

    return TPc / (FN + TPc)

    return 2
    pass


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """

    R = recall_score(pred, true)

    P = precision_score(pred, true)

    return (2 * P * R) / (P + R)
    pass


def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
                      unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development,
     and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """

    l = int(len(unlabeled_data) / (num_iterations - 1))

    unlabelled = {'observed': [], 'hidden': []}

    out = []

    train = training_data.copy()

    for i in range(num_iterations + 1):

        print(len(train))

        transition_probs, emission_probs = estimate_hmm(train)

        dev_observed_sequences = [x['observed'] for x in dev_data]
        dev_hidden_sequences = [x['hidden'] for x in dev_data]

        predictions = []
        for sample in dev_observed_sequences:
            prediction = viterbi(sample, transition_probs, emission_probs)
            predictions.append(prediction)

        predictions_binarized = [[1 if x == 'M' else 0 for x in pred] for pred in predictions]
        dev_hidden_sequences_binarized = [[1 if x == 'M' else 0 for x in dev] for dev in dev_hidden_sequences]

        out.append({'recall': recall_score(predictions_binarized, dev_hidden_sequences_binarized),
                    'precision': precision_score(predictions_binarized, dev_hidden_sequences_binarized),
                    'f1': f1_score(predictions_binarized, dev_hidden_sequences_binarized)})

        train = (training_data.copy())
        to_move = unlabeled_data

        for sample in to_move:
            prediction = viterbi(sample, transition_probs, emission_probs)
            train.append({'observed': sample, 'hidden': prediction})

    print(out, "MINE")
    print(num_iterations, len(out))
    return out