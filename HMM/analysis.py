# Aim is to analyse the HMM class
import numpy as np
import matplotlib.pyplot as plt
from processing import ProcessingHMM
from utils import compute_cost
from collections import defaultdict

# The purpose of this file is to analyse the HMM class, given we have trained the model
# This will involve loading the A and B matrices from memory and then using them to make predictions on the sentences



def predict(A, B, observations, num_states):

    paths  = []

    for obs in observations:
        path = ProcessingHMM.viterbi(obs, num_states, A, B)
        paths.append(path)

    return paths


def analyse():

    # Load the matrices from memory
    A = np.load("C://Users/RamVi/Downloads/NLP-LAPTOP-GNHTUGQO/3/assignment03/hmm/states/A2_fine_39832.npy")
    B = np.load("C://Users/RamVi/Downloads/NLP-LAPTOP-GNHTUGQO/3/assignment03/hmm/states/B2_fine_39832.npy")

    # Try and make a prediction using the matrices
    # We will use the first 10 sentences
    file_path = "ptb-train.conllu"

    vocab, observations, state_mapping, real_states_sentence = ProcessingHMM.parse_conllu_hmm(file_path)


    # Flip vocab and state_mapping dictionaries
    vocab = {v: k for k, v in vocab.items()}
    state_mapping = {v: k for k, v in state_mapping.items()}

    n = len(observations)

    # For the first n sentences feed them into the A and B matrices
    predictions = predict(A, B, observations[:n], num_states = len(state_mapping.keys()))

    print("Predictions: ", len(predictions))


    n = len(observations)

    real_states = real_states_sentence[:n]
    predicted_states = [a for b in predictions for a in b][:n]

    print("Real States: ", real_states)
    print("Predicted States: ", predicted_states)

    _, map = compute_cost(real_states, predicted_states)
    print("Map: ", map)


    predicted_map = []
    # Need to use the Hungarian Algorithm to map
    for letter in predicted_states:
        # Use the reverse of the map
        for key, value in map:
            if value == letter:
                predicted_map.append(state_mapping[key])
                break
    
    plot_mistakes = True

    if plot_mistakes: 
        mistakes = defaultdict(int)
        miss_y = defaultdict(int)

        for i in range(n):
            print(f"R: {state_mapping[real_states[i]]}, P: {predicted_map[i]}", end = "     ")

            if state_mapping[real_states[i]] != predicted_map[i]:
                mistakes[state_mapping[real_states[i]]] += 1
                miss_y[predicted_map[i]] += 1

        
        print("Mistakes: ", mistakes)

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Predicted Mistakes
        ax[0].bar(miss_y.keys(), miss_y.values(), color='skyblue', edgecolor='black', width=0.6)
        ax[0].set_title("Counts of Incorrectly Predicted States", fontsize=14, fontweight='bold')
        ax[0].set_xlabel("Predicted States", fontsize=12)
        ax[0].set_ylabel("Count", fontsize=12)
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)
        ax[0].tick_params(axis='x', labelrotation=45)

        # Plot 2: Real Mistakes
        ax[1].bar(mistakes.keys(), mistakes.values(), color='salmon', edgecolor='black', width=0.6)
        ax[1].set_title("Counts of True States with Prediction Errors", fontsize=14, fontweight='bold')
        ax[1].set_xlabel("True States", fontsize=12)
        ax[1].set_ylabel("Count", fontsize=12)
        ax[1].grid(axis='y', linestyle='--', alpha=0.7)
        ax[1].tick_params(axis='x', labelrotation=45)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save or display the figure
        plt.savefig("mistakes_graph.png", dpi=300)  # Save as high-resolution image for report
        plt.show()

    else:

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

    # # Add overall title and adjust layout
        fig.suptitle("Frequency of Real vs Predicted States", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
        plt.show()
        print(sum(counts_real), sum(counts_pred))

        # Adjust layout and show
        plt.tight_layout()
        plt.show()





if __name__ == "__main__":
    analyse()