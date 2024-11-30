import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from conllu import parse_incr
from typing import List
from sklearn.cluster import KMeans, MiniBatchKMeans
from utils import calculate_v_measure

import logging
from main import parse_conllu

# TODO Seperate each class into a new file and add heirarchy
# TODO Add a main function to run the code, and absract it
# TODO Mess around with the batch size and see how it affects the performance
# TODO Analyse the wrong embeddings, and where the mistmatches are coming from
# TODO Generate some training graphs, squash the dimensions
# TODO Error analysis
# TODO Try and use munkres
# TODO Move all the pts into a seperate folder

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name) # TODO look at BERT tokenizer
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ExtractData():

    def __init__(self, file_path = "ptb-train.conllu"):
        self.file_path = file_path

    def extract_data(self):
        real_states_sentence, lengths = self.parse_conllu_Kmeans()
        return real_states_sentence, lengths
    
    def parse_conllu_Kmeans(self, fine_grained = True):
        # Ideally want to gather both fine and coarse grained POS tags

        real_states_sentence = []
        lengths = []


        state_mapping = dict()
        state_count = 0

        tags = "upos"
        if fine_grained:
            tags = "xpos"

        with open(self.file_path, "r", encoding="utf-8") as f:
            for sentence in parse_incr(f):
                state_sentence = [] 

                for token in sentence:

                    if token[tags] not in state_mapping:
                        state_mapping[token["xpos"]] = state_count
                        state_count += 1

                    state_sentence.append(state_mapping[token[tags]])

                lengths.append(len(state_sentence))
                real_states_sentence.append(state_sentence)

        return real_states_sentence, lengths


class BERTEncoder():

    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def bert_trial(self):
        """
        This function is used to test the BERT model, tokeniser and torch setup
        """
        sentence = "The cat sat on the mat mat mat"
        inputs = self.tokenizer(sentence, return_tensors="pt")

        inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state[0]

        
        logging.debug("BERT trial")
        logging.debug(last_hidden_states.shape)
        logging.info(last_hidden_states)
        logging.info("BERT trial complete")

    def bert_batch(self, sentences_raw : List[List[str]] ):
        """
        This function is used to generate the embeddings for the sentences in the dataset
        
        Input:
        sentences_raw: List of sentences, where each sentence is a list of words

        Output:
        None

        Effect:
        Saves the embeddings, attention masks and word mappings to the disk
        This is done to save memory and allow for the processing of large datasets
        """


        output_embeddings = []
        output_attention_masks = []
        word_mappings =  []

        dataset = Dataset(sentences_raw, tokenizer)
        DataLoader = torch.utils.data.DataLoader(dataset, batch_size=2000) # TODO Analyse this further

        for sentences in DataLoader:

            input_ids = sentences['input_ids'].to(self.device)
            attention_mask = sentences['attention_mask'].to(self.device)


            with torch.no_grad(): # no grad is used to save memory
                outputs = model(input_ids, attention_mask=attention_mask)


            last_hidden_states = outputs.last_hidden_state
            word_ids = sentences['word_ids'].cpu()

            output_embeddings.append(last_hidden_states.cpu())
            output_attention_masks.append(attention_mask.cpu())
            word_mappings.append(word_ids)

            logging.debug(last_hidden_states.shape)


            # Deallocation of the GPU memory for next batch
            del last_hidden_states
            del attention_mask
            del word_ids

            logging.debug( f"We currently have {len(output_embeddings)}" )


        torch.save(output_embeddings, "output_embeddings.pt")
        torch.save(output_attention_masks, "output_attention_masks.pt")
        torch.save(word_mappings, "word_mappings.pt")
        

    @staticmethod
    def load_bert_batch():
        output_embeddings = torch.load("output_embeddings.pt")
        output_attention_masks = torch.load("output_attention_masks.pt")
        word_mappings = torch.load("word_mappings.pt")

        return output_embeddings, output_attention_masks, word_mappings


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sentences, tokeniser, max_length = 100):
        self.sentences = sentences
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]

        encoding = self.tokeniser(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length, is_split_into_words=True)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'word_ids': np.array([x if x is not None else np.nan for x in encoding.word_ids()])
        }
    
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
    
    def main(self):
        
        real_states_sentence, lengths = ExtractData().extract_data()

        num_states = np.unique(np.concatenate(real_states_sentence)).shape[0]

        logger.info(f"Running with {num_states} clusters")

        embeddings, attention_mask, word_mappings = BERTEncoder.load_bert_batch()

        mbk = MiniBatchKMeans(n_clusters=num_states, batch_size=len(embeddings), max_iter=100)

        for l in range(len(embeddings)):
            combined_embeddings = embeddings[l]
            attention = attention_mask[l]


            reshaped_embeddings = combined_embeddings.view(-1, 768)
            reshape_attention = attention.view(-1)

            # The attention mask is used to remove the padding from the embeddings
            logging.debug(reshaped_embeddings[reshape_attention.bool()].shape)
            mbk.partial_fit(reshaped_embeddings[reshape_attention.bool()].numpy())

        # Shape is [num_sentences, 100, 768] where 100 = max_length and 768 = embedding size
        total_embeddings = torch.cat(embeddings, dim=0)
        reshaped_embeddings = total_embeddings.view(-1, 768)

        # Now we need to reshape the predictions back to the original shape of [num_sentences, num_words]
        predictions = mbk.predict(reshaped_embeddings.numpy())
        predictions = predictions.reshape(total_embeddings.shape[0], total_embeddings.shape[1])

        # Now we use the word_mappings to get the real states of the words, this is of shape [batch, num_sentences, num_words], we need to flatten this to [num_sentences, num_words]
        word_mappings = torch.cat(word_mappings, dim=0)

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


        logger.info("Printing the lengths of the predicted and real labels")
        logger.debug(len(np.concatenate(predicted_labels, axis=0)))
        logger.debug(len(real_states_sentence))

        homo_score, comp_score, v_score = calculate_v_measure(np.concatenate(predicted_labels, axis=0) , np.concatenate(real_labels, axis = 0))
        logger.info(f"Scores of this pass \n Homo {homo_score}, Comps: {comp_score}, V : {v_score}")



if __name__ == "__main__":

    Kmeans = KMeans()
    Kmeans.main()


def create_embeddings():
    
    file_path = "ptb-train.conllu"
    word_data, vocab, observations, state_mapping, real_states_sentence, sentences = parse_conllu(file_path) # TODO Make it a seperate function, add testing etc
    b = BERTEncoder()
    b.bert_batch(sentences[:])


class Tests:

    @staticmethod
    def test_bert_encoder():
        """
        This function is used to test the BERT encoder class
        """
        # TODO add the trial function to this class rather than the other
        bert_encoder = BERTEncoder()

        bert_encoder.bert_trial()
    pass