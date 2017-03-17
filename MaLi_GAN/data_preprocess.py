import numpy as np
from tqdm import tqdm

UNK = 0
START = 1
END = 2

def preprocess_penn(data_file, lexicon_file, glove_file, emb_size, max_length):
    all_words = set([])
    token_stream = []
    with open('data/' + data_file, 'r') as f:
        print "Pre-processing sentences in the Pen TreeBank..."
        parsed_line = ['<START>']
        for i, word in tqdm(enumerate(f)):
            if word != ' ':
                parsed_line.append(word)
                all_words.add(word)
            elif word == ' ' and (len(parsed_line) <= max_length):
                parsed_line.append('<END>')
                token_stream.append(parsed_line)
                parsed_line = ['<START>']
    new_lexicon = trim_glove(all_words, lexicon_file, glove_file, emb_size)
    f = open('data/preprocessed_' + data_file, 'w')
    for word in tqdm(token_stream):
        f.write(word)
    f.close()
    f = open('data/trimmed_' + lexicon_file, 'w')
    for word in tqdm(new_lexicon.keys()):
        f.write(word)
    return token_stream, new_lexicon

def create_lexicon(file_name):
    counter = 0
    lexicon = {}
    with open('data/' + file_name, 'r') as f:
        print "Pre-processing and saving word lexicon in memory..."
        for word in tqdm(f):
            lexicon[word] = counter
            counter += 1
    print counter
    return lexicon

def trim_glove(all_words, lexicon_file, glove_file, emb_size):
    lexicon = create_lexicon(lexicon_file)
    embeddings = np.load('data/' + glove_file)
    new_lexicon = {'<START>': START, '<UNK>': UNK, '<END>': END}
    new_embeddings = []
    new_embeddings.append([0] * emb_size)
    counter = 3
    for word in tqdm(all_words):
        if word in lexicon.keys():
            new_lexicon[word] = counter
            idx = lexicon[word]
            new_embeddings.append(embeddings[idx])
    np.save('data/trimmed_' + glove_file, new_embeddings)
    return new_lexicon

lexicon = {}

dl = preprocess_penn('train_sentences.conll', 'word_lexicon.txt', 'glove_vectors.npy', 300, 35)

