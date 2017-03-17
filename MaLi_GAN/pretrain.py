import tensorflow as tf
import numpy as np
from data_loader import DataLoader as dl
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
import cPickle

# initialize constants
seqlen = 20
DROPOUT_KEEP_PROB = 0.75
batch_size = 512
embedding_size = 300
vocab_size = 5000

# load real data
positive_file = 'save/real_data.txt'
pos_dl = dl(seqlen, batch_size, True, positive_file)

eval_file = 'save/eval_data.txt'
eval_dl = dl(seqlen, batch_size, True, positive_file)

pretrained_embeddings = tf.get_variable('embeddings', initializer=tf.random_normal([vocab_size, embedding_size]))

with tf.variable_scope('generator'):
    gen = Generator(vocab_size, batch_size, embedding_size, 150, seqlen, 0, 10, pretrained_embeddings)
#with tf.variable_scope('discriminator'):
#    dis = Discriminator(seqlen, batch_size, pretrained_embeddings)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#saver.restore(sess, './pretrained')

# pretrain 
perplexities = []#cPickle.load(open('pretrain_perplexities.txt'))
for i in range(500):
    loss = gen.pretrain_one_epoch(sess, pos_dl)
    if i % 5 == 0:
    	perp = gen.get_perplexity(sess, eval_dl)
    	print perp
    	perplexities.append(perp)

    	with open('pretrain_perplexities_eval.txt', 'w') as f:
    		cPickle.dump(perplexities, f)
    	saver.save(sess, 'pretrained_eval')

with open('pretrain_perplexities_eval.txt', 'w') as f:
    cPickle.dump(perplexities, f)
saver.save(sess, 'pretrained_eval')
