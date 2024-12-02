# Aim is to analyse the HMM class
import numpy as np
import matplotlib.pyplot as plt
from HMM import HMM
from HMM import PostProcessingHMM

def predict(A, B, observations, num_states):

    paths  = []

    for obs in observations:
        
        # Can use HMM().viterbi() to get the most likely sequence of states
        path = HMM().viterbi(obs, num_states, A, B)
        paths.append(path)

    return paths


def analyse():

    # Load the matrices
    A = np.load("C:/Users/RamVi/Downloads/NLP-LAPTOP-GNHTUGQO/3/assignment03/states/A2_fine_39832.npy")
    B = np.load("C:/Users/RamVi/Downloads/NLP-LAPTOP-GNHTUGQO/3/assignment03/states/B2_fine_39832.npy")

    # Try and make a prediction using the matrices
    # We will use the first 10 sentences
    file_path = "ptb-train.conllu"

    vocab, observations, state_mapping, real_states_sentence = PostProcessingHMM.parse_conllu(file_path)

    # Flip vocab and state_mapping
    vocab = {v: k for k, v in vocab.items()}
    state_mapping = {v: k for k, v in state_mapping.items()}

    # print("Real States Sentence: ", real_states_sentence[:10])

    # Map real_states_sentence to the state_mapping
    real_states_sentence = [[state_mapping[state] for state in sentence] for sentence in real_states_sentence]
    # print("Real States Sentence: ", real_states_sentence[:10])

    # For the first 10 sentences feed them into the A and B matrices
    predictions = predict(A, B, observations[:n], num_states = len(state_mapping.keys()))


    # Real States
    real_states = [state for sentence in real_states_sentence for state in sentence]
    real_states = np.array(real_states)
    unique_real, counts_real = np.unique(real_states, return_counts=True)

    # Sort real states
    sorted_indices_real = np.argsort(counts_real)
    unique_real = unique_real[sorted_indices_real]
    counts_real = counts_real[sorted_indices_real]

    # Predicted States
    predicted_states = [state for sentence in predictions for state in sentence]
    predicted_states = np.array(predicted_states)
    unique_pred, counts_pred = np.unique(predicted_states, return_counts=True)

    # Sort predicted states
    sorted_indices_pred = np.argsort(counts_pred)
    unique_pred = unique_pred[sorted_indices_pred]
    counts_pred = counts_pred[sorted_indices_pred]

    # Detailed analysis, take a sentence and compare the real and predicted states
    # For each sentence, compare the real and predicted states by the rank of the states in the real and predicted states
    sentence = observations[0]
    real_states = real_states_sentence[0]
    predicted_states = predictions[0]

    # Map the states to the real and predicted states to their ranks
    real_states_rank = {state: rank for rank, state in enumerate(unique_real)}
    predicted_states_rank = {state: rank for rank, state in enumerate(unique_pred)}

    # Map the real and predicted states to their ranks for the first sentence
    real_states = [real_states_rank[state] for state in real_states]
    predicted_states = [predicted_states_rank[state] for state in predicted_states]

    print("Real States: ", real_states)
    print("Predicted States: ", predicted_states)

    # Print the number of states
    print("Number of Real States: ", len(unique_real))
    print("Number of Predicted States: ", len(unique_pred))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Plotting
    colors_real = plt.cm.Blues(np.linspace(0.5, 1, len(unique_real)))
    colors_pred = colors_real

    # Real States Bar Chart
    axes[0].bar(unique_real, counts_real, color=colors_real, edgecolor='black', alpha=0.8)
    axes[0].set_title("Real States", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines
    axes[0].tick_params(bottom=False, labelbottom=False)  # Hide x-axis labels

    # Predicted States Bar Chart
    axes[1].bar(unique_real[:len(counts_pred)], counts_pred, color=colors_pred, edgecolor='black', alpha=0.8)
    axes[1].set_title("Predicted States", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines
    axes[1].tick_params(bottom=False, labelbottom=False)  # Hide x-axis labels

    # Add overall title and adjust layout
    fig.suptitle("Frequency of Real vs Predicted States", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    plt.show()
    print(sum(counts_real), sum(counts_pred))

    # Adjust layout and show
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    analyse()