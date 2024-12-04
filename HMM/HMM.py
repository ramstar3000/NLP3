import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from multiprocessing import Pool
from processing import ProcessingHMM
from utils import normalize_log_probs

# np.random.seed(0) for reproducibility


class HMM:
    "The main class for building HMM models for POS tagging"

    def __init__(self):
        pass    

    
    def forward(self, observations):
        """
        This function will implement the forward algorithm in logarithmic space (Generates the alpha matrix)
        Input:
        - observations: The sequence of observations : List
        Output:
        - The alpha matrix : np.array
            This is the matrix to represent the probability of being in state i at time t given the observations upto time t
        """
        T = len(observations)
        N = self.A.shape[0]

        alpha = np.full((N, T), -np.inf)  # Log of 0 is -inf

        alpha[0, 0] = 0 # Initial state is 0 (log(1) = 0)

        for t in range(1, T):
            for j in range(N):
                alpha[j, t] = np.logaddexp.reduce(alpha[:, t - 1] + self.A[:, j]) + self.B[j, observations[t]]
        return alpha
    
    def backward(self, observations):
        """
        This function will implement the backward algorithm in logarithmic space (Generates the beta matrix)
        Input:
        - observations: The sequence of observations : List
        Output:
        - The beta matrix : np.array
            This is the matrix to represent the probability of observing the observations from t + 1 to T given state i at time t
        """
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
        """
        This function will implement the forward-backward algorithm in logarithmic space
        Input:
        - observations: The sequence of observations : List
        - hidden_state_set: The set of hidden states : List

        Output:
        - The gamma matrix : np.array
            This is the matrix to represent the probability of being in state i at time t given the entire sequence of observations
        - The xi matrix : np.array
            This is the matrix to represent the probability of transitioning from state i to j at time t given the entire sequence of observations
        - The alpha matrix : np.array
        - The beta matrix : np.array
            These are passed back for tolerance calculations
        """

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
        """
        This function will initialise the A and B matricies
        Input:
        - load: Whether to load the matricies from file or not : bool
        - number: The number of the file to load : int

        Output:
        - None

        State
        - Initialises self.A and self.B
        """

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

            self.A = normalize_log_probs(self.A)
            self.B = normalize_log_probs(self.B)

            return
        try:
            self.A = np.load(f"/states/A2_{number}.npy")
            self.B = np.load(f"/states/B2_{number}.npy")
        except:
            print("Error loading A and B, reinitialising")
            self.initialise_matricies(False, number)

    def find_tolerance(self, prev_log_likelihood, alpha):
        """
        This function will calculate the tolerance for the convergence of the model
        This is using the logsumexp of alpha to calculate the log likelihood, detail provided in the report

        Input:
        - prev_log_likelihood: The previous log likelihood : float
        - alpha: The alpha matrix : np.array

        Output:
        - The real difference between the log likelihoods : float
        """
        
        log_likelihood = logsumexp(alpha)
        return np.abs(log_likelihood - prev_log_likelihood) / np.abs(prev_log_likelihood) , log_likelihood
    

    def process_sentences(self, sentences):
        """
        This function will process the sentences in parallel
        This was done to speed up the training process and provided a significant speed up

        Input:
        - sentences: The sentences to process : List

        Output:
        - The xis numerator : np.array
        - The xis denominator : np.array
            - These are the numerators and denominators for updating xi

        - The gamma numerator : np.array
        - The gamma denominator : np.array
            - These are the numerators and denominators for updating gamma

        - The alpha matrix : np.array
        """

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

        """
        This function will train the HMM model using the Baum-Welch algorithm
        Input:
        - sentences: The sentences to train on : List
        - output_vocab: The output vocabulary : List
        - hidden_state_set: The set of hidden states : List
        - max_iter: The maximum number of iterations : int
        - tol: The tolerance for convergence : float
        - r: The real states of the sentences : List
        - number: The number of the file to load : int
        - load: Whether to load the matricies from file or not : bool
        - num_states: The number of states : int

        Output:
        - The transition matrix : np.array
        - The emission matrix : np.array

        File I/O
        - Saves the A and B matricies to file for evaluation
          
        """
        
        self.N = len(hidden_state_set)
        self.M = len(output_vocab)

        # Initialise A and B
        self.initialise_matricies(load, number)        

        self.hidden_state_set = hidden_state_set
        prev_log_likelihood = -1e10

        for iteration in tqdm(range(max_iter)):
            
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
                ProcessingHMM.full_post_loop(self.A, self.B, number, sentences, num_states, r)

            self.A = new_A
            self.B = new_B
            
            # Save A and B into a file for evaluation
            np.save(f"states/A2_fine_{number}.npy", self.A)
            np.save(f"states/B2_fine_{number}.npy", self.B)


        return self.A, self.B



def main():
    """
    This function will run the main training loop for the HMM model
    Input:
    - None

    Output Files
    - Saves the A and B matricies to file for evaluation

    """
    
    file_path = "ptb-train.conllu"

    vocab, observations, state_mapping, real_states_sentence = ProcessingHMM.parse_conllu_hmm(file_path, fine_grained=True)
    
    states = list(state_mapping.values()) 
    print(f"States: {len(states)}")  # 19 or 49

    hmm = HMM()


    number = len(observations)
    A, B = hmm.train(observations[:number], vocab, states, max_iter=100, tol = 1e-6, r = real_states_sentence, number = number, load=True, num_states = len(states) )

    ProcessingHMM.full_post_loop(A, B, number, observations, len(states), real_states_sentence)
    

if __name__ == "__main__":
    main()