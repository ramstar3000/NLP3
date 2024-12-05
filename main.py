from conllu import parse_incr
# from HMM import HMM
# from utils import compute_cost
import numpy as np

# from HMM import HMM

# Create a HMM tagger, we have auxiliary functions in utils
from matplotlib import pyplot as plt
import seaborn as sns
    


def parse_conllu(file_path, debug=False):

    word_data = []
    vocab = dict()
    observations = []
    real_states_sentence = []
    all_sentences=  []


    state_mapping = dict()


    count = 2
    state_count = 2
    
    state_mapping["<start>"] = 0
    state_mapping["<end>"] = 1

    vocab["<start>"] = 0
    vocab["<end>"] = 1

    # Note need to add auxiliary end and start states
        # We can say 0 is start and 1 is end



    with open(file_path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            current_sentence = [0]
            state_sentence = ["<start>"]
            real_sentence = []

            for token in sentence:
                word_data.append({
                    "form": token["form"],
                    "upos": token["upos"],
                    "xpos": token["xpos"],
                })

                if token["form"] not in vocab:
                    vocab[token["form"]] = count
                    count += 1

                if token["upos"] not in state_mapping:
                    state_mapping[token["upos"]] = state_count
                    state_count += 1

                # states.add(token["xpos"])

                # Check for X or unkown token
                if token["upos"] == "X":
                    print(token["form"], token["xpos"])
                
                current_sentence.append(vocab[token["form"]])
                state_sentence.append(token["upos"])
                real_sentence.append(token["form"])


            current_sentence.append(1)
            state_sentence.append("<end>")
            observations.append(current_sentence)
            real_states_sentence += state_sentence
            all_sentences.append(real_sentence)

            if debug:
                return word_data, vocab, observations, state_mapping, real_states_sentence, all_sentences
    raise Exception
    return word_data, vocab, observations, state_mapping, real_states_sentence, all_sentences


if __name__ == "__main__":

    file_path = "ptb-train.conllu"

    word_data, vocab, observations, state_mapping, real_states_sentence, all_sentences = parse_conllu(file_path)

    # Want to plot labels for each sentence by freuqency of occurence
    # We can use real states sentence to get the labels

    # We can use the real states sentence to get the labels and use numpy
    real_states_sentence = np.array(real_states_sentence)

    # Sort the array
    unique, counts = np.unique(real_states_sentence, return_counts=True)

    print(unique, counts)

    sorted_indices = np.argsort(counts)[::-1]
    unique = unique[sorted_indices]
    counts = counts[sorted_indices]


    sns.set_palette("muted")
    sns.set_style("white")

    # Create a bar plot
    plt.figure(figsize=(12, 7))
    sns.barplot(x=unique, y=counts, palette="viridis")

    # Add title and axis labels
    plt.title("Frequency of Tags in Treebank | XPOS", fontsize=16, weight="bold")
    plt.xlabel("Tag Type", fontsize=14, weight="bold")
    plt.ylabel("Frequency", fontsize=14, weight="bold")

    plt.xticks(fontsize=11, rotation=45, ha="right")  # Rotate for better readability
    plt.yticks(fontsize=10, rotation=-20)

    plt.text(0.5, counts[0], f"{counts[0]}", ha='center', fontsize=9, color='black', weight='semibold')

    for i, value in enumerate(counts):
        if i % 2 == 1:
            plt.text(i + 0.2, value + 1.9, f"{value}", ha='center', fontsize=9, color='black', weight='semibold')

    plt.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.3, alpha=0.9)

    plt.tight_layout()
    # Show the plot
    plt.savefig("graphing/xpos_frequency.png")
    plt.show()





# def main():
    
#     file_path = "ptb-train.conllu"


#     _, vocab, observations, state_mapping, real_states_sentence, _ = parse_conllu(file_path)

#     number = 50

#     states = list(state_mapping.values()) # Should be replaced with just the number of states
#     print(f"States: {len(states)}")

#     hmm = HMM()


#     A, B = hmm.train(observations[:number], vocab, states, max_iter=20, tol = 1e-5)

#     raise Exception("Stop here")
#     # Extract the raw probabilities
  
#     print(np.round(np.exp(A), 3)) # print with precission of 1dp: 
#     print(np.round(np.exp(B), 3)) # print with precission of 1dp:

#     # raise Exception("Stop here")

#     symbols = []

#     for i in range(number):
#         path = hmm.viterbi(observations[i], len(states), A, B)
#         symbols += path

#     cost, indexes = compute_cost(symbols, real_states_sentence[:len(symbols)])

#     # print("Cost: ", cost)
#     # print("Indexes: ", indexes) # This definitely does not work
#     print("Indexes: ", indexes) # This definitely does not work
#     # Use the mapping on path to get the real states
#     changed_path = []
#     for p in real_states_sentence[:len(symbols)]:
#         # Use indices which is a pair of (real_state, predicted_state)
#         for index in indexes:
#             if p in index:
#                 if p == index[1]:
#                     changed_path.append(index[0])


#     assert len(changed_path) == len(symbols), f"Lengths do not match {len(changed_path)} != {len(symbols)}"

#     same = 0
#     for i in range(len(symbols)):
#         if changed_path[i] == symbols[i]:
#             same += 1
#     print(same, len(changed_path))

#     print("Accuracy: ", same / len(changed_path))

#     # Now attempt on the test set

#     test_number = int(len(observations) * test)

#     symbols = []

#     for i in range(number, number + test_number):
#         path = hmm.viterbi(observations[i], len(states), A, B)
#         symbols += path

#     print("PATH:", symbols)

#     print("REAL PATH:", real_states_sentence[number:number + test_number])

#     # Use same indexes from above?

#     changed_path = []
#     for p in real_states_sentence[number:number + test_number]:
#         # Use indices which is a pair of (real_state, predicted_state)
#         for index in indexes:
#             if p in index:
#                 if p == index[1]:
#                     changed_path.append(index[0])

    
#     same = 0
#     for i in range(len(symbols)):
#         if changed_path[i] == symbols[i]:
#             same += 1
#     print(same, len(changed_path))
#     print("Accuracy TEST: ", same / len(changed_path))




# if __name__ == "__main__":


#     main()

#     # # Some code to test the HMM class training method | Spome dummy examples

#     # hmm = HMM()

#     # A = np.random.rand(2, 2)
#     # B = np.random.rand(2, 2)

#     # observations = [[3, 4, 3], [3, 3, 4, 3]]
    
#     # for i in range(len(observations)):
#     #    observations[i] = [0] + observations[i] + [1]

#     # real_states_sentence = [0, 2, 3, 2, 1, 0, 2, 2, 3, 2, 1]


#     # output_vocab = {0: 0, 1: 1, 2:2, 3:3, 4:4, 5:5}
#     # states = [0, 1, 2, 3, 4, 5, 6, 7 ]

#     # A, B = hmm.train(observations, output_vocab, states, max_iter=57)

#     # print(np.round(np.exp(A), 4)) 
#     # print(np.round(np.exp(B), 4))

#     # A_weird = np.exp(A)
#     # B_weird = np.exp(B)

#     # symbols = []
#     # number = len(observations)
#     # full_info = []



#     # for i in range(number):
        
#     #     path = hmm.viterbi(observations[i], len(states), A_weird, B_weird)
#     #     # path = hmm.viterbi(observations[i], transition_prob_dict, emission_prob_dict)
#     #     # print(observations[i], path)
#     #     symbols += path
#     #     full_info.append(path)


#     # print("FULL INFO: ", full_info)
#     # print("PATH:", symbols)
#     # print("REAL PATH:", real_states_sentence[:len(symbols)])

#     # cost, indexes = compute_cost(symbols, real_states_sentence[:len(symbols)])

#     # print("Cost: ", cost)
#     # print("Indexes: ", indexes) # This definitely does not work

#     # # Use the mapping on path to get the real states

#     # changed_path = []

#     # for p in symbols:
#     #     # Use indices which is a pair of (real_state, predicted_state)
#     #     for index in indexes:
#     #         if p in index:
#     #             if p == index[1]:
#     #                 changed_path.append(index[0])
#     #                 break
#     #             elif p == index[0]:
#     #                 print("BROKEN")
#     #                 changed_path.append(index[1])
#     #                 break
#     #             else:
#     #                 raise Exception("This should not happen")
                
#     # print("Changed path: ", changed_path)

#     # same = 0
#     # for i in range(len(changed_path)):
#     #     if changed_path[i] == real_states_sentence[i]:
#     #         same += 1
#     # print(same, len(changed_path))


#     # hmm = HMM()

#     # transition_prob = np.log(np.array([
#     # [0.0, 0.6, 0.4],  # From state 0
#     # [0.0, 0.7, 0.3],  # From state 1
#     # [0.0, 0.4, 0.6]   # From state 2
#     #         ]))
#     # emission_prob = np.log(np.array([
#     # [0.5, 0.5],  # State 1
#     # [0.4, 0.6],  # State 2
#     # [0.7, 0.3]   # State 3
#     #     ]))
    

#     # # Convert into dictionaries for viterbi, index is (state1, state2) and (state, value)
#     # transition_prob_dict = {}
#     # emission_prob_dict = {}

#     # for i in range(transition_prob.shape[0]):
#     #     for j in range(transition_prob.shape[1]):
#     #         transition_prob_dict[(i, j)] = transition_prob[i, j]

#     # for i in range(emission_prob.shape[0]):
#     #     for j in range(emission_prob.shape[1]):
#     #         emission_prob_dict[(i, j)] = emission_prob[i, j]

 

    

    

#     # observations = [0, 1, 0]



#     # path = hmm.viterbi(observations, transition_prob_dict, emission_prob_dict)

#     # print(path)