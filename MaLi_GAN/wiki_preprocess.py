import numpy as np
from tqdm import tqdm
from data_loader import DataLoader

lexicon = {}
lexicon['<START>'] = 0
lexicon['<END>'] = 1
lexicon['<UNK>'] = 2
counter = 3
with open('data/word_lexicon.txt', 'r') as f:
    for word in f:
        lexicon[word] = counter
        counter += 1
        

dl = DataLoader(lexicon, 10, 10, False)
dl.load_data('../wikitext-2/wiki.train.tokens')
dl.load_data('../wikitext-2/wiki.test.tokens')
dl.load_data('../wikitext-2/wiki.valid.tokens')

print len(dl.token_stream)

