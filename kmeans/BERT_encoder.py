import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List

import logging


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

        dataset = Dataset(sentences_raw, self.tokenizer)
        DataLoader = torch.utils.data.DataLoader(dataset, batch_size=2000) # TODO Analyse this further

        for sentences in DataLoader:

            input_ids = sentences['input_ids'].to(self.device)
            attention_mask = sentences['attention_mask'].to(self.device)


            with torch.no_grad(): # no grad is used to save memory
                outputs = self.model(input_ids, attention_mask=attention_mask)


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
