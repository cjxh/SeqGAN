import numpy as np
from tqdm import tqdm
from collections import defaultdict

UNK = 0
START = 1
END = 2

def preprocess_penn(data_file, max_length, is_eval=False):
    word_frequency = defaultdict(int)
    write_tokens = open('preprocessed_data/' + data_file, 'w')
    with open('penn_treebank/' + data_file, 'r') as f:
        print "Pre-processing sentences in " + data_file + "..."
        sentence = ['<START>']
        for line in tqdm(f):
            line = line.strip()
            tup = line.split()
            if len(tup) == 0: 
                if len(sentence) <= max_length - 1:
                    sentence.append('<END>')
                    for word in sentence: 
                        write_tokens.write(word + '\n')
                sentence = ['<START>']
            else:
                sentence.append(tup[1])
                word_frequency[tup[1]] += 1
    write_tokens.close()
    if not is_eval:
        del word_frequency['<START>']
        if '<UNK>' in word_frequency.keys():
            del word_frequency['<UNK>']
        del word_frequency['<END>']
        all_words = set(sorted(word_frequency, key=word_frequency.get, reverse=True)[:10000])
        return all_words

def create_lexicon(file_name):
    counter = 0
    lexicon = {}
    with open('glove/' + file_name, 'r') as f:
        print "Pre-processing and saving word lexicon in memory..."
        for word in tqdm(f):
            word = word.strip()
            lexicon[word] = counter
            counter += 1
    print len(lexicon.keys())
    return lexicon

def trim_glove_and_lexicon(all_words, lexicon_file, glove_file, emb_size):
    lexicon = create_lexicon(lexicon_file)
    embeddings = np.load('glove/' + glove_file)
    new_lexicon = {'<START>': START, '<UNK>': UNK, '<END>': END}
    new_embeddings = []
    new_embeddings.append([0] * emb_size)
    new_embeddings.append([0] * emb_size)
    new_embeddings.append([0] * emb_size)
    counter = 3
    print "Trimming glove vectors and lexicon..."
    for word in tqdm(all_words):
        if word in lexicon.keys():
            new_lexicon[word] = counter
            idx = lexicon[word]
            new_embeddings.append(embeddings[idx])
    f = open('glove/trimmed_' + lexicon_file, 'w')
    print "Writing top 10k words to lexicon file..."
    for word in tqdm(new_lexicon.keys()):
        f.write(word + '\n')
    f.close()
    print "Saving trimmed glove vectors..."
    np.save('glove/trimmed_' + glove_file, new_embeddings)
    return new_lexicon

def preprocess_wrapper(data_file, eval_file, lexicon_file, glove_file, emb_size, max_length):
    all_words = preprocess_penn(data_file, max_length) 
    preprocess_penn(eval_file, max_length, True) 
    trim_glove_and_lexicon(all_words, lexicon_file, glove_file, emb_size)
    
preprocess_wrapper('train.txt', 'eval.txt', 'word_lexicon.txt', 'glove_vectors.npy', 300, 35)
