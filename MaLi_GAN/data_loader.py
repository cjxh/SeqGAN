import numpy as np
import math

class DataLoader(object):
    def __init__(self, lexicon):
        self.batch_size = 0
        self.token_stream = []
        self.pointer = 0
        self.lexicon = lexicon
        self.SYNTHETIC = True
        self.END_TOKEN = -1
        self.max_length = 0

    '''
    normalizes sentence length to max sentence length
    shuffles indices of data to help randomize minibatches
    '''
    def pre_process_sentences(self):
        temp_token_stream = []
        num_sentences = len(self.token_stream)
        shuffled_idx = np.random.permutation(num_sentences)
        for _, idx in enumerate(shuffled_idx):
            sentence = self.token_stream[idx]
            if len(sentence) < self.max_length:
                for i in range(self.max_length - len(sentence)):
                    sentence.append(self.END_TOKEN)
            temp_token_stream.append(sentence)
        return temp_token_stream

    def load_data(self, data_file):
        self.max_length = 0
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                parsed_line = []
                if self.SYNTHETIC:
                    line = line.strip()
                    line = line.split()
                    parsed_line = [int(num) for num in line]
                else:
                    parsed_line = [self.lexicon[word] for word in line]
                self.token_stream.append(parsed_line)
                if len(parsed_line) > self.max_length:
                    self.max_length = len(parsed_line)
        self.token_stream = self.pre_process_sentences()

    def shuffle_sentences(self):
        num_sentences = len(self.token_stream)
        shuffle_idx = np.random.permutation(np.arange(num_sentences))
        shuffled_data =  self.token_stream[shuffle_idx]
        return shuffled_data

    def mini_batch(self, batch_size):
        num_batch = len(self.token_stream) / batch_size
        shuffled_stream = self.shuffle_sentences()
        shuffled_stream = shuffled_stream[:num_batch * batch_size]
        self.mini_batches = np.split(np.array(shuffled_stream), num_batch, 0)
        return self.mini_batches

    def get_max_length(self):
        return self.max_length
