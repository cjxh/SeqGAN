import tensorflow as tf
import numpy as np
from data_loader import DataLoader as dl
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
import cPickle as pickle

# initialize constants
seqlen = 35
DROPOUT_KEEP_PROB = 0.75
batch_size = 64
embedding_size = 300

lexicon = pickle.load(open('../data/glove/trimmed_word_lexicon.p', 'r'))
print lexicon
vocab_size = len(lexicon.keys()) 

# load real data
positive_file = '../data/preprocessed_data/train.txt'
pos_dl = dl(seqlen, batch_size, False, positive_file, lexicon)

eval_file = '../data/preprocessed_data/eval.txt'
eval_dl = dl(seqlen, batch_size, False, eval_file, lexicon)

#pretrained_embeddings = tf.get_variable('embeddings', initializer=tf.random_normal([vocab_size, embedding_size]))
pretrained_embeddings = tf.get_variable('embeddings', initializer=np.load('../data/glove/trimmed_glove_vectors.npy'))

with tf.variable_scope('generator'):
    gen = Generator(vocab_size, batch_size, embedding_size, 150, seqlen, 0, 10, pretrained_embeddings)
#with tf.variable_scope('discriminator'):
#    dis = Discriminator(seqlen, batch_size, pretrained_embeddings)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#saver.restore(sess, './pretrained')

# pretrain 
#perplexities = cPickle.load(open('pretrain_perplexities.txt'))
perplexities = []
for i in range(500):
    loss = gen.pretrain_one_epoch(sess, pos_dl)

    '''if i % 5 == 0:
    	perp = gen.get_perplexity(sess, pos_dl)
    	print perp
    	perplexities.append(perp)

    	with open('pretrain_perplexities.txt', 'w') as f:
    		pickle.dump(perplexities, f)
    	saver.save(sess, 'pretrained')'''
    print loss

with open('pretrain_perplexities.txt', 'w') as f:
    pickle.dump(perplexities, f)
saver.save(sess, 'pretrained')
