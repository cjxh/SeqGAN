import tensorflow as tf
import numpy as np
from data_loader import DataLoader as dl
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
import cPickle as pickle
from target_lstm import TARGET_LSTM

# initialize constants
T = 35
N = T
K = 1
k = 5
DROPOUT_KEEP_PROB = 0.75
batch_size = 64
embedding_size = 300
n_classes = 2

# populate the lexicon of existing words
lexicon = pickle.load(open('../data/glove/trimmed_word_lexicon.p', 'r'))
vocab_size = len(lexicon.keys()) 
reverse_lexicon = {}
for word in lexicon.keys():
    reverse_lexicon[lexicon[word]] = word
    
vocab_size = len(lexicon.keys()) 

# load real data
positive_file = '../data/preprocessed_data/train.txt'
pos_dl = dl(T, batch_size, False, positive_file, lexicon)

eval_file = '../data/preprocessed_data/eval.txt'
eval_dl = dl(T, batch_size, False, eval_file, lexicon)

with tf.variable_scope('generator'):
    pretrained_embeddings = tf.get_variable('embeddings', initializer=tf.random_normal([5000, 300]))
    gen = Generator(vocab_size, batch_size, embedding_size, 150, T, 0, 10, pretrained_embeddings)

with tf.variable_scope('discriminator'):
    pretrained_embeddings = tf.get_variable('embeddings', initializer=tf.random_normal([5000, 300]))
    dis = Discriminator(T, batch_size, n_classes, pretrained_embeddings)

dis_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
dis_optimizer = tf.train.AdamOptimizer(1e-4)
dis_grads_and_vars = dis_optimizer.compute_gradients(dis.loss, dis_params)
dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pretrain 
for i in range(5):
    epoch_loss = gen.pretrain_one_epoch(sess, pos_dl)
    print 'loss: ' + str(epoch_loss)

for i in tqdm(range(k)):
    # minibatches of real training data ... do they mean 1 or all minibatches??
    real_minibatch = pos_dl.next_batch()
    # minibatch, get first N from real_minibatch, generate the rest
    gen_minibatch = gen.generate_from_latch(sess, real_minibatch, N)
    loss = dis.train_one_step(sess, data_loader, real_minibatch, gen_minibatch)
    print 'dis ' + str(loss)

while N - K >= 0:
    N = N - K
    for i in range(k):
        # minibatches of real training data ... do they mean 1 or all minibatches??
        real_minibatch, _, _ = data_loader.next_batch()
        # minibatch, get first N from real_minibatch, generate the rest
        gen_minibatch = gen.generate_from_latch(sess, real_minibatches, N)
        loss = dis.train_one_step(sess, data_loader, real_minibatch, gen_minibatch)
        print 'dis ' + str(loss)    

    # minibatch of real training data
    new_minibatch = data_loader.next_batch()
    xij = gen.generate_xij(sess, new_minibatch, N)
    loss, accuracy, output = gen.train_one_step(sess, dis, xij,  N) / (T-N)
    print 'loss: ' + str(loss)
    print 'accuracy: ' + str(accuracy)
