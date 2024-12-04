# About this Project

This project is a test of unsupervised learning algorithms for POS tagging for the English language. 

The goal is to compare the performance of different algorithms on the same dataset. The algorithms that will be tested are:
- KMeans
- HMM

# Dataset

We used the Penn Treebank dataset for training and testing the algorithms.

 The dataset contains 1 million words and their corresponding POS tags.


This data is a collection of tagged sentences from the Wall Street Journal.

# How to Initialize the Project

To run the project, you can use the following command:

```
pip install -r requirements.txt
```
The most important thing to download is conllu, numpy for the HMM model.


The important modules for the KMeans model are numpy, sklearn, pytorch and transformers.

The modules used for graphing are matplotlib and seaborn.

# How to Run the Project

To run the project, you can use the following command:

```
cd hmm
python hmm.py

cd k_means
python k_means.py
```
These 2 will run the baseline models for the project.

More details can be found below.

# Run HMM

### HMM Model
This is located in the HMM.py file. The model is implemented using the following methods:

- Viterbi Algorithm
- Baum-Welch Algorithm as the training algorithm

### Running this project
To run this project, you can use the following command:

```
cd hmm
python hmm.py
```

To adapt any hyperparameters, you can change the values in the `hmm.py` file within the main function.


### Implementation

I have chosen to follow the implementation from the book "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. The book provides a clear and concise explanation of the HMM model and its implementation. I have adapted this algorithm to work with the Penn Treebank dataset.



# Run K-means with BERT embeddings


### Main File
The main file for this implementation is `k_means.py`. This file contains the implementation of the k-means clustering algorithm using BERT embeddings.


### Running this project
To run this project, you can use the following command:

```
cd kmeans
python k_means.py
```

To adapt any hyperparameters, you can change the values in the `k_means.py` file within the main function.

Additionally, the implementation to generate the embeddings can be found by running the BERT_encoder file. This file contains the implementation to generate the BERT embeddings for the Penn Treebank dataset. This has a hyperparameter basic to choose between styles of embeddings (mean or final layer) and the number of sentences to generate embeddings for.


### Implementation

I have chosen to follow the implementation from the book "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. The book provides a clear and concise explanation of the HMM model and its implementation. I have adapted this algorithm to work with the Penn Treebank dataset.



# How to Run the Graphing

To run the graphing, you can use the following command:

```
cd graphing
python *.py
```

This will run the graphing for the project. There are various files generated for the different metrics that are calculated.

# License

This project is open source, no license exists for this project.

Penn Treebank dataset is used for this project.
- Created by Marcus et al. at 1995, the The Penn Treebank Project Naturally occurring text annotated for linguistic structure., in English language

# Authors
Ram Vinjamuri

