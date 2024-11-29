# Old kmeans

kmeans = KMeans(n_clusters=k, tol=1e-6, max_iter=100)
kmeans.fit(reshaped_embeddings)
labels = kmeans.labels_
labels = labels.reshape(attention_mask.shape[0], attention_mask.shape[1])


# MISC

# first_sentences = embeddings[0].reshape(-1, 768)

        # print(first_sentences.shape)

        # first_sentences = first_sentences[:1000]

        # print(first_sentences.shape)

        # # Apply the attention mask to the first 10 sentences

        # print(attention_mask[0].reshape(-1)[:1000].shape)

        # first_sentences = first_sentences[attention_mask[0].reshape(-1).bool()[:1000]]

        # print(first_sentences.shape)


        # kmeans_labels = mbk.predict(first_sentences.numpy())

        
        # Print the real labels for the first 10 sentences
        # print(len(real_states_sentence[:ones[10]]))

        # print(len(kmeans_labels))


        # homo_score, comp_score, v_score = calculate_v_measure(kmeans_labels[:243], real_states_sentence[:ones[10]][:243])


    # We need embeddings[0][0] and to take the first 10 sentences

    # labels = mbk.predict(reshaped_embeddings[reshape_attention.bool()].numpy())
    
    # Generate labels for the first 10 sentences ONLY


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


Main HMM logic:



        # For each 

        for iteration in tqdm(range(max_iter)):
            
            total_gamma = np.full((N), -np.inf) # Sum over sentences
            total_xi = np.full((N, N), -np.inf)

            total_emissions = np.full((N, M), -np.inf)
            total_gamma_per_state = np.full((N), -np.inf)


            for sentence in sentences:
                
                sentence = np.array(sentence)
                gamma, xi = self.forward_backward(sentence, output_vocab, hidden_state_set)

                total_gamma = np.logaddexp(total_gamma, logsumexp(gamma, axis=1))
                total_xi = np.logaddexp(total_xi, logsumexp(xi, axis=0))


                for t, observation in enumerate(sentence):
                    total_emissions[:, observation] = np.logaddexp(
                            total_emissions[:, observation], gamma[:, t])

                total_gamma_per_state = np.logaddexp(
                    total_gamma_per_state, np.sum(gamma, axis=1))

            


            # M step to update A and B

            a_new = np.zeros_like(self.A)
            b_new = np.zeros_like(self.B)

            # print("Total Gamma: ", total_gamma)


            for i in range(N): # TODO vectorise this
                denominator = total_gamma[i]
                for j in range(N):

                    numerator = total_xi[i, j]
                    a_new[i, j] = xi[i, j] - total_gamma[i]

            for i in range(N):  # For each state
                for k in range(M):
                    b_new[i, k] = total_emissions[i, k] - total_gamma_per_state[i]


            print("A new: ", a_new)
            print("B new: ", b_new)

            raise Exception("Stop here")

            # Normalise A and B as the normalised a_new
            a_new = normalize_log_probs(a_new)
            b_new = normalize_log_probs(b_new)

            print("A new: ", a_new)
            print("B new: ", b_new)

            raise Exception("Stop here")
            # a_new = normalize_log_probs(a_new)

            # print("A new: ", a_new)

            # b_new = normalize_log_probs(b_new)

            # print(self.A == a_new)
            # print(self.B == b_new)


            a_diff = np.abs(logsumexp(a_new) - logsumexp(self.A))
            b_diff = np.abs(logsumexp(b_new) - logsumexp(self.B))

            # print("A diff: ", a_diff , end=" ")
            # print("B diff: ", b_diff)
            # print(self.A - a_new)

            if a_diff < tol and b_diff < tol:
                print("Converged", a_diff)
                
                break
            else:
                print(f"Not Converged  {iteration}", a_diff, b_diff)



            self.A = a_new
            self.B = b_new

                        
            print("A: ", np.round(np.exp(self.A), 2))
            print("B: ", np.round(np.exp(self.B), 2))

        # Print the final tolerances

        print("Final A: ", self.A)
        print("Final B: ", self.B)
        return (self.A, self.B) 



