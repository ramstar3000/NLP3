# About this Project

This project is a test of unsupervised learning algorithms for POS tagging for the English language. 

The goal is to compare the performance of different algorithms on the same dataset. The algorithms that will be tested are:
- KMeans
- HMM

## Dataset

We used the Penn Treebank dataset for training and testing the algorithms.

 The dataset contains 1 million words and their corresponding POS tags.


This data is a collection of tagged sentences from the Wall Street Journal.

## How to Initialize the Project

To run the project, you can use the following command:

```
pip install -r requirements.txt
```
The most important thing to download is conllu, numpy for the HMM model.


The important modules for the KMeans model are numpy, sklearn, pytorch and transformers.

The modules used for graphing are matplotlib and seaborn.

## How to Run the Project

To run the project, you can use the following command:

```
cd hmm
python hmm.py

cd k_means
python k_means.py
```
These 2 will run the baseline models for the project.

More details can be found in the respective folders, with more detailed readmes for each model.

## How to Run the Graphing

To run the graphing, you can use the following command:

```
cd graphing
python *.py
```

This will run the graphing for the project. There are various files generated for the different metrics that are calculated.


## License

This project is open source, no license exists for this project.

Penn Treebank dataset is used for this project.
- Created by Marcus et al. at 1995, the The Penn Treebank Project Naturally occurring text annotated for linguistic structure., in English language

## Authors
Ram Vinjamuri

