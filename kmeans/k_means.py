import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from utils import calculate_v_measure

from BERT_encoder import BERTEncoder
from data_processor import ExtractData
import logging

# TODO Analyse the wrong embeddings, and where the mistmatches are coming from
# TODO Generate some training graphs, squash the dimensions
# TODO Try and use munkres
# TODO Try and use the same embeddings for the same sentences

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

    
class KMeans():

    @staticmethod
    def aggregate_predictions(word_ids, predictions):
        """
        This function will aggregate the predictions for a word
        Input:
        - word_ids: The word ids : np.array
        - predictions: The predictions : np.array

        Output:
        - The aggregated predictions : List
        """

        output_labels = []

        unique_word_ids = np.unique(word_ids)
        unique_word_ids = unique_word_ids[~np.isnan(unique_word_ids)]

        for word_id in unique_word_ids:

            token_indices = np.where(word_ids == word_id)[0]
            word_predictions = predictions[token_indices]

            word_predictions = np.bincount(word_predictions).argmax()
            output_labels.append(word_predictions)


        return output_labels
    
    def main(self, fine_grained = True, max_iter = 100):
        """
        This function will run the KMeans algorithm
        Input:
        - fine_grained: The type of tag : boolean
        - max_iter: The maximum number of iterations : int

        Output:
        - The evaluation scores : Tuple[float, float, float]
        """
        
        real_states_sentence, lengths = ExtractData().extract_data(fine_grained)

        num_states = np.unique(np.concatenate(real_states_sentence)).shape[0]

        logger.info(f"Running with {num_states} clusters")

        embeddings, attention_mask, word_mappings = BERTEncoder.load_bert_batch(basic=True)

        embeddings_padded_length = 100 # Is 150 for some embeddings, here 100 for speed

        batch_size = 200000 # This is the maximum number of words we can have in a batch
        mbk = MiniBatchKMeans(n_clusters=num_states, batch_size=batch_size, max_iter=max_iter, tol=1e-20, verbose=0)
    
        if batch_size <= 200000: # In our list of tensors, each has 2000 sentences == 2000 * n = 200000 words in this example
            for l in range(len(embeddings)):
                combined_embeddings = embeddings[l]
                attention = attention_mask[l]

                reshaped_embeddings = combined_embeddings.view(-1, 768)
                reshape_attention = attention.view(-1)

                # mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()][: 1000].numpy()) Used to smaller batches
                # mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()][1000: ].numpy())    

                mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()].numpy())

        else: # This is the case where we have more than 2000 sentences in a batch
            for l in range(0, len(embeddings) - 1, 4):
                combined_embeddings = torch.cat((embeddings[l: l+4]), dim=0)
                attention = torch.cat((attention_mask[l: l+4]), dim=0)

                reshaped_embeddings = combined_embeddings.view(-1, 768)
                reshape_attention = attention.view(-1)

                mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()].numpy())

        overall_predictions = []

        # Shape is [num_sentences, n, 768] where n = max_length_embedding and 768 = embedding size
        for total_embeddings in embeddings: # Total embeddings is of shape [num_sentences, n, 768], num_sentences = 2000
            reshaped_embeddings = total_embeddings.view(-1, 768)

            # Now we need to reshape the predictions back to the original shape of [num_sentences, num_words]
            predictions = mbk.predict(reshaped_embeddings.numpy()) # This is of shape [num_sentences * n]
            predictions = predictions.reshape(-1, embeddings_padded_length)
            overall_predictions.append(predictions)

        # Now we use the word_mappings to get the real states of the words, this is of shape [batch, num_sentences, num_words], we need to flatten this to [num_sentences, num_words]
        word_mappings = torch.cat(word_mappings, dim=0)
        predictions = np.concatenate(overall_predictions, axis=0)

        # Now we want to extract the true labels using this word mapping, we have the function aggregate_predictions to do this, loop over each sentence and aggregate the predictions
        predicted_labels = []
        real_labels = []

        for i in range(len(word_mappings)):
            temp = self.aggregate_predictions(word_mappings[i], predictions[i])

            if len(temp) == lengths[i]:
                predicted_labels.append(temp)
                real_labels.append(real_states_sentence[i])
                pass
            # else:
            #     raise ValueError(f"Lengths do not match {len(temp)} and {lengths[i]} | Probably need to increase the embeddings pad size")

        homo_score, comp_score, v_score = calculate_v_measure(np.concatenate(predicted_labels, axis=0) , np.concatenate(real_labels, axis = 0))
        logger.info(f"Scores of this pass \n Homo {homo_score}, Comps: {comp_score}, V : {v_score}")

        return homo_score, comp_score, v_score

def main():

    iterations = True # Chooses which branch of the coda you want to run

    if iterations:
        Kmeans = KMeans()

        print("Running with 30 iterations")

        print("Running experiments on fine_grained")
        Kmeans.main(fine_grained=True, max_iter=30)
        
        print("Running experiments on fine_grained 100")
        Kmeans.main(fine_grained=True, max_iter=100)

        print("Running on coarse grained")
        Kmeans.main(fine_grained=False, max_iter=30)

        print("Repeating with 100 iterations")
        print("Running on coarse grained 100")
        Kmeans.main(fine_grained=False, max_iter=100)

    else:

        Kmeans = KMeans()
        results_fine = []
        results_coarse = []

        for _ in range(5):
            results_fine.append(Kmeans.main(fine_grained=True, max_iter=30))
            results_coarse.append(Kmeans.main(fine_grained=False, max_iter=30))

        results_coarse = np.array(results_coarse)
        results_fine = np.array(results_fine)

        # Print the variance and mean across each element of the tuple, 
        fine_mean = np.mean(results_fine, axis=0)  # Mean of each metric (homogeneity, completeness, V-measure)
        fine_var = np.var(results_fine, axis=0)    # Variance of each metric

        # Calculate mean and variance for coarse-grained results
        coarse_mean = np.mean(results_coarse, axis=0)
        coarse_var = np.var(results_coarse, axis=0)

        print("Fine-Grained Results:")
        print(f"Mean: Homogeneity={fine_mean[0]:.4f}, Completeness={fine_mean[1]:.4f}, V-Measure={fine_mean[2]:.4f}")
        print(f"Variance: Homogeneity={fine_var[0]:.4f}, Completeness={fine_var[1]:.4f}, V-Measure={fine_var[2]:.4f}")

        print("\nCoarse-Grained Results:")
        print(f"Mean: Homogeneity={coarse_mean[0]:.4f}, Completeness={coarse_mean[1]:.4f}, V-Measure={coarse_mean[2]:.4f}")
        print(f"Variance: Homogeneity={coarse_var[0]:.4f}, Completeness={coarse_var[1]:.4f}, V-Measure={coarse_var[2]:.4f}")




if __name__ == "__main__":
    main()
