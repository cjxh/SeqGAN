import numpy as np
from tqdm import tqdm

UNK = 0
START = 1
END = 2

def preprocess_penn(data_file, lexicon_file, glove_file, emb_size, max_length):
    all_words = set([])
    token_stream = []
    write_tokens = open('preprocessed_' + data_file + '.txt', 'w')
    with open(data_file + '.conll', 'r') as f:
        print "Pre-processing sentences in the Pen TreeBank..."
        write_tokens.write('<START>\n')
        for i, word in tqdm(enumerate(f)):
            word = word.strip()
            if word != '':
                write_tokens.write(word + '\n')
                all_words.add(word)
            else:
                write_tokens.write('<END>\n')
                write_tokens.write('<START>\n')
    write_tokens.close()
    '''
    f = open('trimmed_' + lexicon_file, 'w')
    for word in tqdm(new_lexicon.keys()):
        f.write(word)
    return token_stream, new_lexicon
    '''

def create_lexicon(file_name):
    counter = 0
    lexicon = {}
    with open(file_name, 'r') as f:
        print "Pre-processing and saving word lexicon in memory..."
        for word in tqdm(f):
            lexicon[word] = counter
            counter += 1
    print counter
    return lexicon

def trim_glove(all_words, lexicon_file, glove_file, emb_size):
    lexicon = create_lexicon(lexicon_file)
    embeddings = np.load(glove_file)
    new_lexicon = {'<START>': START, '<UNK>': UNK, '<END>': END}
    new_embeddings = []
    new_embeddings.append([0] * emb_size)
    counter = 3
    for word in tqdm(all_words):
        if word in lexicon.keys():
            new_lexicon[word] = counter
            idx = lexicon[word]
            new_embeddings.append(embeddings[idx])
    np.save('trimmed_' + glove_file, new_embeddings)
    return new_lexicon

lexicon = {}

dl = preprocess_penn('train_sentences', 'word_lexicon.txt', 'glove_vectors.npy', 300, 35)

