import torch
import numpy as np
from conllu import parse_incr
from sklearn.cluster import KMeans, MiniBatchKMeans
from utils import calculate_v_measure

from BERT_encoder import BERTEncoder, Dataset
from data_processor import ExtractData
import logging

# TODO Seperate each class into a new file and add heirarchy
# TODO Add a main function to run the code, and absract it
# TODO Mess around with the batch size and see how it affects the performance
# TODO Analyse the wrong embeddings, and where the mistmatches are coming from
# TODO Generate some training graphs, squash the dimensions
# TODO Error analysis
# TODO Try and use munkres
# TODO Move all the pts into a seperate folder

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

    
class KMeans():

    @staticmethod
    def aggregate_predictions(word_ids, predictions):

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
        
        real_states_sentence, lengths = ExtractData().extract_data(fine_grained)

        num_states = np.unique(np.concatenate(real_states_sentence)).shape[0]

        logger.info(f"Running with {num_states} clusters")

        embeddings, attention_mask, word_mappings = BERTEncoder.load_bert_batch()


        mbk = MiniBatchKMeans(n_clusters=num_states, batch_size=800000, max_iter=max_iter, tol=1e-20, verbose=0)
        # Note batch size is in words, need to redo our calculuations lmao

        # for l in range(len(embeddings)):
        #     combined_embeddings = embeddings[l]
        #     attention = attention_mask[l]


        #     reshaped_embeddings = combined_embeddings.view(-1, 768)
        #     reshape_attention = attention.view(-1)

        #     # The attention mask is used to remove the padding from the embeddings
        #     # logging.debug(reshaped_embeddings[reshape_attention.bool()].shape)

        #     # mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()][: 1000].numpy())
        #     # mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()][1000: ].numpy())    

        #     mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()].numpy())
        
        for l in range(0, len(embeddings) - 1, 4):
            combined_embeddings = torch.cat((embeddings[l: l+4]), dim=0)
            attention = torch.cat((attention_mask[l: l+4]), dim=0)

            reshaped_embeddings = combined_embeddings.view(-1, 768)
            reshape_attention = attention.view(-1)

            # The attention mask is used to remove the padding from the embeddings
            # logging.debug(reshaped_embeddings[reshape_attention.bool()].shape)
            mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()].numpy())

        overall_predictions = []

        # Shape is [num_sentences, 100, 768] where 100 = max_length and 768 = embedding size
        for total_embeddings in embeddings: # Total embeddings is of shape [num_sentences, 100, 768], num_sentences = 2000
            reshaped_embeddings = total_embeddings.view(-1, 768)

            # Now we need to reshape the predictions back to the original shape of [num_sentences, num_words]
            predictions = mbk.predict(reshaped_embeddings.numpy()) # This is of shape [num_sentences * 100]
            predictions = predictions.reshape(-1, 100)
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
            else:
                # This is a problem with inconsistent tokenisation
                # TODO Analyse this case and why is arises
                continue


        # logger.info("Printing the lengths of the predicted and real labels")
        # logger.debug(len(np.concatenate(predicted_labels, axis=0)))
        # logger.debug(len(real_states_sentence))

        homo_score, comp_score, v_score = calculate_v_measure(np.concatenate(predicted_labels, axis=0) , np.concatenate(real_labels, axis = 0))
        logger.info(f"Scores of this pass \n Homo {homo_score}, Comps: {comp_score}, V : {v_score}")
        print(f"Scores of this pass \n Homo {homo_score}, Comps: {comp_score}, V : {v_score}")
        # Save the model: TODO
        return homo_score, comp_score, v_score


if __name__ == "__main__":


    # Kmeans = KMeans()

    # print("Running with 30 iterations")

    # print("Running experiments on fine_grained")
    # Kmeans.main(fine_grained=True, max_iter=30)

    
    # print("Running experiments on fine_grained 100")
    # Kmeans.main(fine_grained=True, max_iter=100)

    # print("Running on coarse grained")
    # Kmeans.main(fine_grained=False, max_iter=30)

    # print("Repeating with 100 iterations")
    # print("Running on coarse grained 100")
    # Kmeans.main(fine_grained=False, max_iter=100)

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



# def create_embeddings():
    
#     file_path = "ptb-train.conllu"
#     word_data, vocab, observations, state_mapping, real_states_sentence, sentences = parse_conllu(file_path) # TODO Make it a seperate function, add testing etc
#     b = BERTEncoder()
#     b.bert_batch(sentences)

