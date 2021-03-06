import numpy as np
import math
from tqdm import tqdm

UNK = 0
START = 1 
END = 2

class DataLoader(object):
    def __init__(self, N, batch_size, is_synthetic, data_file, lexicon):
        self.batch_size = batch_size
        self.max_length = N
        self.token_stream = []
        self.mask_sequence_stream = []
        self.pointer = 0
        self.SYNTHETIC = is_synthetic
        self.sentences = []
        self.seq_lens = []
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
        mask_sequence_stream = []
        print "Padding sentences in token_stream to " + str(self.max_length)  + "..."
        for sentence in tqdm(self.token_stream):
            self.seq_lens.append(len(sentence))
            mask_sequence = [True] * len(sentence)
            if len(sentence) < self.max_length:
                for i in range(self.max_length - len(sentence)):
                    sentence.append(END)
                    mask_sequence.append(False)
            mask_sequence_stream.append(mask_sequence)
            temp_token_stream.append(sentence)
        return temp_token_stream, mask_sequence_stream

    def load_syn_data(self, data_file):
        with open(data_file, 'r') as f:
            print "Parsing synthetic sentences in " + str(data_file) + "..."
            for line in tqdm(f):
                print "Doing this loop"
                if self.SYNTHETIC:
                    parsed_line = []
                    line = line.strip()
                    line = line.split()
                    parsed_line = [int(num) for num in line]
                    if len(parsed_line) <= 20:
                        self.token_stream.append(parsed_line)
            self.token_stream, self.mask_sequence_stream = self.pre_process_sentences()

    def load_data(self, data_file):
        with open(data_file, 'r') as f:
            print "Parsing sentences in " + str(data_file) + "..."
            parsed_line = []
            word_ct = 0
            sentence_ct = 0
            self.num_unk = 0
            for word in tqdm(f):
                word_ct += 1
                word = word.strip()
                if word in self.lexicon.keys():
                    parsed_line.append(self.lexicon[word])
                else:
                    self.num_unk += 1
                    parsed_line.append(UNK)
                if word == '<END>':
                    sentence_ct += 1
                    self.token_stream.append(parsed_line)
                    parsed_line = []
        print "percent of unkowns: " + str((self.num_unk * 1.0) / word_ct)
        print "num_sentences: " + str(sentence_ct)
        self.token_stream, self.mask_sequence_stream = self.pre_process_sentences()

    def next_batch(self):
        ret = self.mini_batches[self.pointer]
        ret_mask = self.mini_batches_mask[self.pointer]
        ret_seq_lens = self.mini_batches_seq_lens[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret, ret_mask, ret_seq_lens

    def reset_pointer(self):
        self.pointer = 0

    def shuffle_sentences(self):
        num_sentences = len(self.token_stream)
        shuffle_idx = np.random.permutation(np.arange(num_sentences))
        shuffled_data = []
        shuffled_mask = []
        shuffled_seq_lens = []
        for i in shuffle_idx:
            shuffled_data.append(self.token_stream[i])
            shuffled_mask.append(self.mask_sequence_stream[i])
            shuffled_seq_lens.append(self.seq_lens[i])
        return shuffled_data, shuffled_mask, shuffled_seq_lens

    def mini_batch(self, batch_size):
        self.num_batch = len(self.token_stream) / batch_size
        shuffled_stream, shuffled_mask, shuffled_seq_lens = self.shuffle_sentences()
        shuffled_stream = shuffled_stream[:self.num_batch * batch_size]
        shuffled_mask = shuffled_mask[:self.num_batch * batch_size]
        shuffled_seq_lens = shuffled_seq_lens[:self.num_batch * batch_size]
        self.mini_batches = np.split(np.array(shuffled_stream), self.num_batch, 0)
        self.mini_batches_mask = np.split(np.array(shuffled_mask), self.num_batch, 0)
        self.mini_batches_seq_lens = np.split(np.array(shuffled_seq_lens), self.num_batch, 0)
        self.pointer = 0
