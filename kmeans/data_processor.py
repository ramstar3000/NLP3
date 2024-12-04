from conllu import parse_incr


class ExtractData():

    def __init__(self, file_path = "ptb-train.conllu"):
        self.file_path = file_path

    def extract_data(self, fine_grained = True):
        real_states_sentence, lengths = self.parse_conllu_Kmeans(fine_grained)
        return real_states_sentence, lengths
    
    def parse_conllu_Kmeans(self, fine_grained = True):
        """
        Input:
        - file_path: The conllu file
        - fine_grained: Type of tag ; boolean for tags

        Output:
        - real_states_sentence: array
        - lengths: array
        """

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
                        state_mapping[token[tags]] = state_count
                        state_count += 1

                    state_sentence.append(state_mapping[token[tags]])

                lengths.append(len(state_sentence))
                real_states_sentence.append(state_sentence)

        return real_states_sentence, lengths

    def parse_conllu_embeddings(self, file_path):
        """
        Input:
        - file_path: The conllu file

        Output:
        - all_sentences: List
        """
        all_sentences = []

        with open(file_path, "r", encoding="utf-8") as f:
            for sentence in parse_incr(f):
                real_sentence = []

                for token in sentence:
                    real_sentence.append(token["form"])

                all_sentences.append(real_sentence)

        return all_sentences
