from conllu import parse_incr
import numpy as np
from utils import calculate_v_measure

class ProcessingHMM:

    @staticmethod
    def viterbi(observations, num_states, transition_prob, emission_prob): 
        """
        This function will implement the viterbi algorithm in logarithmic space
        Input:
        - Observations: The sequence of observations : List
        - num_states: The number of states : int
        - transition_prob: The transition probabilities : np.array
        - emission_prob: The emission probabilities : np.array
        Output:
        - The optimal path : List
        """

        V = np.full((num_states, len(observations)), -np.inf)
        path = {}

        V[:, 0] = emission_prob[:, observations[0]]

        for t in range(1, len(observations)):
            for s in range(1, num_states):
                prob = V[:, t - 1] + transition_prob[:, s] + emission_prob[s - 1, observations[t]] # Changed to logs
                V[s, t] = np.max(prob)
                path[s, t] = np.argmax(prob)

        optimal_path = []
        last_state = np.argmax(V[:, -1])

        if last_state == 0:
            last_state = 1
        optimal_path.append(last_state)

        for t in range(len(observations) - 1, 1, -1):
            last_state = path[last_state, t]
            optimal_path.insert(0, last_state)

        optimal_path.insert(0, 0)

        return optimal_path

    @staticmethod
    def full_post_loop(A, B, number, observations, l, real_states_sentence):
        """
        This function runs the full evaluation loop for the HMM model
        Input:
        - A: The transition matrix : np.array
        - B: The emission matrix : np.array
        - number: The number of sentences to evaluate : int
        - observations: The observations : List
        - l: The number of states : int
        - real_states_sentence: The real states of the sentence : List

        Output:
        - The evaluation scores : Tuple[float, float, float]
        """

        path = []
        for obs in observations[:number]:
            path += ProcessingHMM.viterbi(obs, l, A, B)

        ones = np.where(np.array(real_states_sentence) == 1)[0]

        return ProcessingHMM.evaluate(path, real_states_sentence[:ones[number - 1] + 1])


    @staticmethod
    def evaluate(predicted_path, real_path):
        """
        This function will evaluate the predicted path against the real path
        Input:
        - predicted_path: The predicted path : List
        - real_path: The real path : List
        Output:
        - The evaluation scores : Tuple[float, float, float]
        """


        homo_score, comp_score, v_score = calculate_v_measure(predicted_path, real_path)
        print(f"Scores of this pass :: Homo {homo_score}, Comps: {comp_score}, V : {v_score}")
        return homo_score, comp_score, v_score

    @staticmethod
    def parse_conllu_hmm(file_path, debug=False, fine_grained = True):
        """
        This is a function that parses the conllu and outputs the correct information to train a HMM
        Input:
        - file_path: The conllu file
        - debug: To early stop the function and outputs the types
        - fine_grained: Type of tag ; boolean for tags
        
        Output:
        - vocab: Dict
        - observations: 2D array
        - state_mapping: Dict
        - real_state_sentence: array
        """

        word_data = []
        vocab = dict()
        observations = []
        real_states_sentence = []
        all_sentences=  []

        state_mapping = dict()

        count = 2
        state_count = 2
        
        # Note need to add auxiliary end and start states
        state_mapping["<start>"] = 0
        state_mapping["<end>"] = 1
        vocab["<start>"] = 0
        vocab["<end>"] = 1

        tag = "xpos"
        if not(fine_grained):
            tag = "upos"

        with open(file_path, "r", encoding="utf-8") as f:
            for sentence in parse_incr(f):
                current_sentence = [0]
                state_sentence = [0]
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

                    if token[tag] not in state_mapping:
                        state_mapping[token[tag]] = state_count
                        state_count += 1
                    
                    current_sentence.append(vocab[token["form"]])
                    state_sentence.append(state_mapping[token[tag]])
                    real_sentence.append(token["form"])

                current_sentence.append(1)
                state_sentence.append(1)
                observations.append(current_sentence)
                real_states_sentence += state_sentence 
                all_sentences.append(real_sentence)

                if debug:
                    return word_data, vocab, observations, state_mapping, real_states_sentence, all_sentences

        return vocab, observations, state_mapping, real_states_sentence


