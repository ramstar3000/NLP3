NOTE: ctrl-shift-m to preview markdown in VSCode

# Research 




## How HMMs work recap

We have the words are the observations O

POS tags are the hidden states S

Goal is to find the most likely sequence of POS tags given the words

 ($S_1, S_2, ..., S_n$ for a sequence of words $O_1, O_2, ..., O_n$)

Transisition probabilities:
* $P(S_i | S_{i-1})$ - probability of transitioning from state $S_{i-1}$ to state $S_i$

Emission probabilities:
* $P(O_i | S_i)$ - probability of observing word $O_i$ given state $S_i$

Viterbi Algorithm:
* Find the most likely sequence of states given the observations
* Works by doing dynamic programming on the trellis of states

Forward Algorithm:
* Find the probability of the observation sequence given the model
* Used to calculate the likelihood of the observation sequence given the model
* Parametwer Estimation


## Munkres

What is Munkres
- Combinatorial optimization algorithm for solving the assignment problem
- Assigns a column to each row in a matrix such that the sum of the assigned elements is minimized
- O(n^3)
- Applied to HMMs for POS tagging

* Decoding Multiple possible outputs
  * Munkres can help to select an optimal 1-1 aligment of words to tages
  * Using the Munkres algorithm would result in the optimal alignment between words and tags to minimize cost (maximize likelihood).


## Metrics

Homogenity Score
  - Measures how much each cluster contains only members of a single class
  - Higher the score, the more each cluster corresponds to  true class
  - $ 1- \frac{H(C|K)}{H(C)} $
    - $H(C|K)$ is the conditional entropy of the class distribution given the cluster assignments
    - $H(C)$ is the entropy of the class distribution

Completeness Score
    - Measures how much each class is assigned to a single cluster
    - Higher the score, the more each class is assigned to a single cluster
    - $ 1- \frac{H(K|C)}{H(K)} $
        - $H(K|C)$ is the conditional entropy of the cluster assignments given class distribution
        - $H(K)$ is the entropy of the cluster assignments


V_measure Score
    - Harmonic mean of homogenity and completeness
    - $2 * \frac{homogenity * completeness}{homogenity + completeness}$
    - Balanced score that considers both homogenity and completeness

## Functions
TODO: Add docstrings to the functions

### Compute Cost
Initilialise the cost matrix where each element represents the cost of assigning a predicted state to a real state
The cost is computed by comparing the predicted state with the real state

Apply munkres to find the optimal assignment of predicted states to real states


## Forward Backward Algorithm

Learning: Given an observation sequence O and the set of possible
states in the HMM, learn the HMM parameters A and B.




### Mapping from UPOS to XPOS :(                     ):
