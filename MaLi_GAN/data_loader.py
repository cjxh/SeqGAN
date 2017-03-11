import numpy as np
import math

class DataLoader(object):
    def __init__(self, batch_size, lexicon):
        self.batch_size = batch_size
        self.token_stream = []
        self.pointer = 0
        self.lexicon = lexicon
        self.SYNTHETIC = True
        self.END_TOKEN = -1

    '''
    normalizes sentence length to max sentence length
    shuffles indices of data to help randomize minibatches
    '''
    def pre_process(self, max_length):
        temp_token_stream = []
        num_sentences = len(self.token_stream)
        shuffled_idx = np.random.permutation(num_sentences)
        for _, idx in enumerate(shuffled_idx):
            sentence = self.token_stream[idx]
            if len(sentence) < max_length:
                for i in range(max_length - len(sentence)):
                    sentence.append(self.END_TOKEN)
            temp_token_stream.append(sentence)
        return temp_token_stream

    def mini_batch(self, data_file):
        max_length = 0
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
                if len(parsed_line) > max_length:
                    max_length = len(parsed_line)
            self.token_stream = self.pre_process(max_length)
            
        self.num_batch = len(self.token_stream) / self.batch_size
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.mini_batches = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.mini_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        if self.pointer == 0:
            return -1
        return ret

    def reset_pointer(self):
        self.pointer = 0
