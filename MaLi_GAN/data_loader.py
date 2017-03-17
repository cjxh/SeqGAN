import numpy as np
import math
from tqdm import tqdm

UNK = 0
START = 1 
END = 2

class DataLoader(object):
    def __init__(self, N, batch_size, is_synthetic, data_file, lexicon = {}):
        self.batch_size = batch_size
        self.num_batch = 0
        self.max_length = N
        self.token_stream = []
        self.pointer = 0
        self.SYNTHETIC = is_synthetic
        self.sentences = []
        self.lexicon = lexicon
        if not is_synthetic:
            self.load_data(data_file)
        else:
            self.load_syn_data(data_file)
        self.mini_batch(self.batch_size)

    '''
    normalizes sentence length to max sentence length
    shuffles indices of data to help randomize minibatches
    '''
    def pre_process_sentences(self):
        temp_token_stream = []
        num_sentences = len(self.token_stream)
        print "Padding sentences in token_stream to " + str(self.max_length)  + "..."
        for sentence in tqdm(self.token_stream):
            if len(sentence) < self.max_length:
                for i in range(self.max_length - len(sentence)):
                    sentence.append(self.END_TOKEN)
            temp_token_stream.append(sentence)
        return temp_token_stream

    def load_syn_data(self, data_file):
        with open(data_file, 'r') as f:
            print "Parsing synthetic sentences in " + str(data_file) + "..."
            for i, line in tqdm(enumerate(f)):
                self.num_batch += 1
                if self.SYNTHETIC:
                    parsed_line = []
                    line = line.strip()
                    line = line.split()
                    parsed_line = [int(num) for num in line]
                    if len(parsed_line) <= self.max_length:
                        self.token_stream.append(parsed_line)
            self.token_stream = self.pre_process_sentences()

    def load_data(self, data_file):
        with open(data_file, 'r') as f:
            print "Parsing sentences in " + str(data_file) + "..."
            parsed_line = []
            for word in tqdm(f):
                if word in self.lexicon.keys():
                    parsed_line.append(self.lexicon[word])
                else:
                    parsed_line.append(UNK)
                if word == '<END>':
                    if len(parsed_line) <= self.max_length:
                        self.token_stream.append(parsed_line)
                    parsed_line = []
        self.token_stream = self.pre_process_sentences()

    def next_batch(self):
        ret = self.mini_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

    def shuffle_sentences(self):
        num_sentences = len(self.token_stream)
        shuffle_idx = np.random.permutation(np.arange(num_sentences))
        shuffled_data = []
        for i in shuffle_idx:
            shuffled_data.append(self.token_stream[i])
        return shuffled_data

    def mini_batch(self, batch_size):
        num_batch = len(self.token_stream) / batch_size
        shuffled_stream = self.shuffle_sentences()
        shuffled_stream = shuffled_stream[:num_batch * batch_size]
        self.mini_batches = np.split(np.array(shuffled_stream), num_batch, 0)
        self.pointer = 0
