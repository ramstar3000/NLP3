import numpy as np
from tqdm import tqdm

np.random.seed(0)

class HMM:
    "The main class for building HMM models for POS tagging"

    def __init__(self):
        pass    

    
    def viterbi(self, observations, num_states, transition_prob, emission_prob): # Copied from utils
        # TODO add state

        V = np.zeros((num_states, len(observations)))
        path = {}
        # first column of V is the transition prob from state 0
        V[:, 0] = transition_prob[0, :]

        for t in range(1, len(observations)):
            for s in range(num_states):
                prob = V[:, t - 1] * transition_prob[:, s] * emission_prob[s - 1, observations[t]]
                V[s, t] = np.max(prob)
                path[s, t] = np.argmax(prob)

        optimal_path = []
        last_state = 1
        optimal_path.append(last_state)

        # print(path)
        for t in range(len(observations) - 1, 1, -1):
            last_state = path[last_state, t]
            optimal_path.insert(0, last_state)

        optimal_path.insert(0, 0)

        return optimal_path
    
    def forward(self, A, B, observations, pi):
        T = len(observations)
        N = A.shape[0]
        alpha = np.zeros((N, T))

        assert B.shape[0] == N, "B should have the same number of rows as A"
        assert max(observations) < B.shape[1], "Observations should be within the range of the columns of B"

        # Initialise alpha
        alpha[:, 0] = pi * B[:, observations[0]] 

        # Recursion
        for t in range(1, T):
            for j in range(N):
                alpha[j, t] = np.sum(alpha[:, t - 1] * A[:, j]) * B[j, observations[t]] # Inner loop can be vectorised
        return alpha # Does this need to be normalised TODO analyse this properly, would also need to pass in the sclaing factor
        # We can also calculate the probability of the observation sequence by summing over the last column of alpha
    
    def backward(self, A, B, observations):
        T = len(observations)
        N = A.shape[0]
        beta = np.zeros((N, T))

        # Initialise beta
        beta[:, -1] = 1

        # Recursion
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[i, t] = np.sum(A[i, :] * B[:, observations[t + 1]] * beta[:, t + 1]) # This can be vectorised

        return beta

    def forward_backward(self, observations, output_vocab, hidden_state_set, max_iter = 100, tol = 1e-4):

        T = len(observations)
        N = len(hidden_state_set)


        alpha = self.forward(self.A, self.B, observations, self.pi)
        beta = self.backward(self.A, self.B, observations)

        gamma = alpha * beta / np.sum(alpha * beta, axis=0, keepdims=True) # TODO check dimensions

        # print("Gamma: ", gamma.shape)
        # print("Alpha: ", alpha.shape)
        # print("Beta: ", beta.shape)

        # raise Exception("Stop here")

        pi = gamma[:, 0]

        A = self.A
        B = self.B
        pi = self.pi

        # Compute xi
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denom = np.sum(alpha[:, t, None] * A * B[:, observations[t + 1]] * beta[:, t + 1]) # Might need to add axis
            xi[t] = alpha[:, t, None] * A * B[:, observations[t + 1]] * beta[:, t + 1] / denom

            # xi[t] /= np.sum(xi[t])
        
        return gamma, xi
        # Update A

    
    def train(self, sentences, output_vocab, hidden_state_set, max_iter = 50, tol = 1e-18):
        
        T = len(sentences)
        N = len(hidden_state_set)
        M = len(output_vocab)
        tol = 1e-16


        # Initialise A and B
        self.A = np.random.rand(N, N)
        self.B = np.random.rand(N, M)
        self.pi = np.random.rand(N)

        # Normalise A, B and pi
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        self.B /= np.sum(self.B, axis=1, keepdims=True)
        self.pi /= np.sum(self.pi)



        # For each 

        for iteration in tqdm(range(max_iter)):
            
            total_gamma = np.zeros((N)) # Sum over sentences
            total_xi = np.zeros((N, N)) # Sum over transsitions

            # For the updating of B
            total_emissions = np.zeros((N, len(output_vocab)))
            total_gamma_per_state = np.zeros((N))

            for sentence in sentences:
                sentence = np.array(sentence)
                gamma, xi = self.forward_backward(sentence, output_vocab, hidden_state_set)
                
                total_gamma += np.sum(gamma, axis=1)
                total_xi += np.sum(xi, axis=0)

                for t, observation in enumerate(sentence):
                    total_emissions[:, observation] += gamma[:, t] # Depends if gamma is nt or tn
                
                for i in range(N):
                    total_gamma_per_state[i] += np.sum(gamma[i, :])

            # M step to update A and B

            a_new = np.zeros_like(self.A)
            b_new = np.zeros_like(self.B)

            # print("Total Gamma: ", total_gamma)


            for i in range(N): # TODO vectorise this
                denominator = np.sum(gamma[i, :])
                for j in range(N):

                    numerator = total_xi[i, j]
                    a_new[i, j] = numerator / (denominator + 1e-10)

            for i in range(N):  # For each state
                for k in range(M):
                    b_new[i, k] = total_emissions[i, k] / (total_gamma_per_state[i] + 1e-10)

            # Normalise A and B as the normalised a_new

            a_new /= np.sum(a_new, axis=1, keepdims=True)
            b_new /= np.sum(b_new, axis=1, keepdims=True)

            print(a_new, self.A)

            

            if np.linalg.norm(a_new - self.A) < tol and np.linalg.norm(b_new - self.B) < tol:
                print("Converged", np.linalg.norm(a_new - self.A), np.linalg.norm(b_new - self.B))
                break
            else:
                print("Not Converged", np.linalg.norm(a_new - self.A), np.linalg.norm(b_new - self.B))

            self.A = a_new
            self.B = b_new
        # Print the final tolerances


        print("Final A: ", self.A)
        print("Final B: ", self.B)
        return (self.A, self.B) 

        




"""
denominator = total_gamma[i]

                for obs in output_vocab.values():  # For each observation symbol
                    numerator = 0

                    for sentence in sentences:
                        for t, observation in enumerate(sentence):
                            if observation == obs:
                                # print("Observation: ", observation)
                                numerator += gamma[i, t]
                    b_new[i, obs] = numerator / (denominator + 1e-10)


"""