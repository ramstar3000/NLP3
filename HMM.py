import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from main import parse_conllu
from utils import calculate_v_measure, viterbi

from conllu import parse_incr
from multiprocessing import Pool



np.random.seed(0)

# TODO Make it work!!!
# TODO Graph the convergence

def normalize_log_probs(log_probs):

    # Only calculate the log sum of the none -inf values
    log_sum = logsumexp(log_probs, axis = 1, keepdims=True)
    return log_probs - log_sum


class HMM:
    "The main class for building HMM models for POS tagging"

    def __init__(self):
        pass    

    @staticmethod
    def viterbi(observations, num_states, transition_prob, emission_prob): 


        V = np.full((num_states, len(observations)), -np.inf)
        path = {}

        V[:, 0] = emission_prob[:, observations[0]]

        for t in range(1, len(observations)):
            for s in range(1, num_states):
                prob = V[:, t - 1] + transition_prob[:, s] + emission_prob[s - 1, observations[t]] # Changed to logs, TODO correctness proof
                V[s, t] = np.max(prob)
                path[s, t] = np.argmax(prob)

        optimal_path = []
        last_state = np.argmax(V[:, -1])

        if last_state == 0:
            last_state = 1
        optimal_path.append(last_state) # Edge case if broken, can be removed


        for t in range(len(observations) - 1, 1, -1):
            last_state = path[last_state, t]
            optimal_path.insert(0, last_state)

        optimal_path.insert(0, 0)

        return optimal_path
    
    def forward(self, observations):
        T = len(observations)
        N = self.A.shape[0]

        alpha = np.full((N, T), -np.inf)  # Log of 0 is -inf

        alpha[0, 0] = 0

        for t in range(1, T):
            for j in range(N):
                alpha[j, t] = np.logaddexp.reduce(alpha[:, t - 1] + self.A[:, j]) + self.B[j, observations[t]]
        return alpha # Does this need to be normalised TODO analyse this properly, would also need to pass in the sclaing factor
    
    def backward(self, observations):
        T = len(observations)
        N = self.A.shape[0]

        beta = np.full((N, T), -np.inf) 
        beta[:, -1] = 0

        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[i, t] = np.logaddexp.reduce(
                    self.A[i, :] + self.B[:, observations[t + 1]] + beta[:, t + 1]
            )
        return beta

    def forward_backward(self, observations, hidden_state_set):

        T = len(observations)
        N = len(hidden_state_set)

        alpha = self.forward(observations)
        beta = self.backward(observations)

        gamma = np.full((N, T), -np.inf)
        for t in range(T):
            denom = logsumexp(alpha[:, t] + beta[:, t])
            for i in range(N):
                gamma[i, t] = alpha[i, t] + beta[i, t] - denom


        # Compute xi
        xi = np.full((T - 1, N, N), -np.inf)
        for t in range(T - 1):
            denom = logsumexp(alpha[:, t] + beta[:, t])

            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = alpha[i, t] + self.A[i, j] + self.B[j, observations[t + 1]] + beta[j, t + 1] - denom
            
            xi[t] -= logsumexp(xi[t])

        return gamma, xi, alpha, beta

    def initialise_matricies(self, load, number = 0):

        if load == False:
            self.A = np.random.rand(self.N, self.N)
            self.B = np.random.rand(self.N, self.M)

            # # Normalise A, B
            self.A /= np.sum(self.A, axis=1, keepdims=True)
            self.B /= np.sum(self.B, axis=1, keepdims=True)

            self.A[:, 0] = 0
        
            # Set the 1st row of A to be 0s, representing the final state, but we can loop in this state
            self.A[1, :] = 0
            self.A[1, 1] = 1

            # Change B so that observations in state 0 is 0 and 1 for 0
            self.B[0, :] = 0
            self.B[0, 0] = 1

            # Change B so that observations in state 1 is 0 and 1 for 1
            self.B[1, :] = 0
            self.B[:, 1] = 0
            self.B[1, 1] = 1

            self.A = np.log(self.A + 1e-10)
            self.B = np.log(self.B + 1e-10)

            self.A = normalize_log_probs(self.A) # TODO Analyse
            self.B = normalize_log_probs(self.B)

            return
        try:
            self.A = np.load(f"states/A2_{number}.npy")
            self.B = np.load(f"states/B2_{number}.npy")
        except:
            print("Error loading A and B, reinitialising")
            self.initialise_matricies(False, number)

    def find_tolerance(self, prev_log_likelihood, alpha):
        log_likelihood = logsumexp(alpha)
        return np.abs(log_likelihood - prev_log_likelihood) / np.abs(prev_log_likelihood) , log_likelihood
    

    def process_sentences(self, sentences):
        xis_numerator = np.full((self.N, self.N), -np.inf)
        xis_denominator = np.full((self.N), -np.inf)
        gamma_numerator = np.full((self.N, self.M), -np.inf)
        gamma_denominator = np.full((self.N), -np.inf)

        for sentence in sentences:
            
            sentence = np.array(sentence)
            gamma, xi, alpha, _ = self.forward_backward(sentence, self.hidden_state_set)

            xis_numerator = np.logaddexp(xis_numerator, logsumexp(xi, axis=0)) # Sum over time
            xis_denominator = np.logaddexp(xis_denominator, logsumexp(xi, axis=(0, 2))) # Sum over time and j state

            for v in set(sentence):
                indices = np.where(sentence == v)[0]

                gamma_numerator[:, v] = np.logaddexp(gamma_numerator[:, v], logsumexp(gamma[:, indices], axis=1))
                gamma_denominator = np.logaddexp(gamma_denominator, logsumexp(gamma, axis=1))

        return xis_numerator, xis_denominator, gamma_numerator, gamma_denominator, alpha

    def train(self, sentences, output_vocab, hidden_state_set, max_iter = 50, tol = 1e-18, r = None, number = 0, load=False, num_states = 19):
        
        self.N = len(hidden_state_set)
        self.M = len(output_vocab)

        # Initialise A and B
        self.initialise_matricies(load, number)        

        # Additionally, 1 can only be observed in state 1
        self.hidden_state_set = hidden_state_set

        prev_log_likelihood = -1e10

        for iteration in tqdm(range(max_iter)):
            
            # Want to index i, j or i, v
            num_processes = 12
            chunk_size = len(sentences) // num_processes

            chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

            with Pool(num_processes) as p:
                results = p.map(self.process_sentences, chunks)


            # xis numerator using logumexp of first element in the tuple
            xis_numerator = logsumexp([x[0] for x in results], axis=0)
            xis_denominator = logsumexp([x[1] for x in results], axis=0)
            gamma_numerator = logsumexp([x[2] for x in results], axis=0)
            gamma_denominator = logsumexp([x[3] for x in results], axis=0)
            alpha = results[0][4]

            # xis_numerator, xis_denominator, gamma_numerator, gamma_denominator, alpha = self.process_sentences(sentences, hidden_state_set)
            

            # M step to update A and B

            # Need to stack the denominators for xis and gamma
            new_A = xis_numerator - np.stack([xis_denominator] * self.N, axis=1)
            new_B = gamma_numerator - np.stack([gamma_denominator] * self.M, axis=1)

            # Calcualte the difference between the new and old A and B, note these are in log space
            real_diff, log_likelihood  = self.find_tolerance(prev_log_likelihood, alpha[:, -1])

            if real_diff < tol:
                print("Converged")
                break
            else:
                print(f"Likelihood diff: {real_diff}") 
                prev_log_likelihood = log_likelihood

            if iteration % 3 == 1:
                PostProcessingHMM.full_post_loop(self.A, self.B, number, sentences, num_states, r)

            self.A = new_A
            self.B = new_B
            
            # Save A and B into a file for debugging
            np.save(f"states/A2_fine_{number}.npy", self.A)
            np.save(f"states/B2_fine_{number}.npy", self.B)


        return self.A, self.B

def main():
    
    file_path = "ptb-train.conllu"

    vocab, observations, state_mapping, real_states_sentence = PostProcessingHMM.parse_conllu(file_path)
    
    states = list(state_mapping.values()) 
    print(f"States: {len(states)}")  # Should be 19 or 49?

    hmm = HMM()


    number = len(observations)
    A, B = hmm.train(observations[:number], vocab, states, max_iter=100, tol = 1e-18, r = real_states_sentence, number = number, load=False, num_states = len(states))

    PostProcessingHMM.full_post_loop(A, B, number, observations, len(states), real_states_sentence)
    

class PostProcessingHMM:

    @staticmethod
    def full_post_loop(A, B, number, observations, l, real_states_sentence):

        path = []
        for obs in observations[:number]:
            path += HMM.viterbi(obs, l, A, B)

        # Analyse / Compare the patha and the real states
        ones = np.where(np.array(real_states_sentence) == 1)[0]

        PostProcessingHMM.evaluate(path, real_states_sentence[:ones[number - 1] + 1])


    @staticmethod
    def evaluate(predicted_path, real_path):
        homo_score, comp_score, v_score = calculate_v_measure(predicted_path, real_path)
        print(f"Scores of this pass :: Homo {homo_score}, Comps: {comp_score}, V : {v_score}")

    @staticmethod
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
                state_sentence = [0]
                real_sentence = []

                for token in sentence:
                    # temp_word_data = {
                    #     "id": token["id"],
                    #     "form": token["form"],
                    #     "lemma": token["lemma"],
                    #     "upos": token["upos"],
                    #     "xpos": token["xpos"],
                    #     "feats": token["feats"],
                    #     "head": token["head"],
                    #     "deprel": token["deprel"],
                    #     "deps": token["deps"],
                    #     "misc": token["misc"]
                    # } TODO Experiment with adding these

                    word_data.append({
                        "form": token["form"],
                        "upos": token["upos"],
                        "xpos": token["xpos"],
                    })

                    if token["form"] not in vocab:
                        vocab[token["form"]] = count
                        count += 1

                    if token["xpos"] not in state_mapping:
                        state_mapping[token["xpos"]] = state_count
                        state_count += 1

                    # states.add(token["xpos"])
                    
                    current_sentence.append(vocab[token["form"]])
                    state_sentence.append(state_mapping[token["xpos"]])
                    real_sentence.append(token["form"])


                current_sentence.append(1)
                state_sentence.append(1)
                observations.append(current_sentence)
                real_states_sentence += state_sentence
                all_sentences.append(real_sentence)

                if debug:
                    return word_data, vocab, observations, state_mapping, real_states_sentence, all_sentences

        return vocab, observations, state_mapping, real_states_sentence


if __name__ == "__main__":

    main()