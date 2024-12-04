## Introduction

This file aims to guide through the implementation of the Hidden Markov Model (HMM) for Part of Speech (POS) tagging. 

The HMM model is implemented in Python using the numpy library. The model is trained on the Penn Treebank dataset and tested on the same dataset.


## HMM Model
This is located in the HMM.py file. The model is implemented using the following methods:

- Viterbi Algorithm
- Baum-Welch Algorithm as the training algorithm

## Running this project
To run this project, you can use the following command:

```
cd hmm
python hmm.py
```

To adapt any hyperparameters, you can change the values in the `hmm.py` file within the main function.


## Implementation

I have chosen to follow the implementation from the book "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. The book provides a clear and concise explanation of the HMM model and its implementation. I have adapted this algorithm to work with the Penn Treebank dataset.


## More details

For more details on the implementation, you can refer to the `research.md` file in the `assignment03` folder. This file contains more detailed information on the HMM model and its implementation.
Additional details can be found in the report