import tensorflow as tf
import numpy as np
from data_loader import DataLoader as dl
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
import cPickle
from target_lstm import TARGET_LSTM

# initialize constants
T = 20
N = T
K = 1
k = 5
DROPOUT_KEEP_PROB = 0.75
batch_size = 32
embedding_size = 300
n_classes = 2

# populate the lexicon of existing words
lexicon = {}
'''
counter = 0
with open('data/word_lexicon.txt', 'r') as f:
    print "Pre-processing and saving word lexicon in memory..."
    for line in tqdm(f):
        for word in line:
            lexicon[word] = counter
            counter += 1
'''

# load real data
positive_file = 'save/real_data.txt'
data_loader = dl(lexicon, N, batch_size)
print "Loading data from " + positive_file + " into memory..."
positive_data = data_loader.load_data(positive_file)
#pretrained_embeddings = np.load('data/glove_vectors.npy')


pretrained_embeddings = tf.get_variable('embeddings', initializer=tf.random_normal([5000, 300]))

# initialize generator and discriminator
target_params = cPickle.load(open('save/target_params.pkl'))
target_lstm = TARGET_LSTM(5000, 64, 32, 32, 20, 0, target_params)   


with tf.variable_scope('generator'):
    gen = Generator(5000, batch_size, 300, 150, T, 0, 20, pretrained_embeddings)
with tf.variable_scope('discriminator'):
    dis = Discriminator(N, batch_size, n_classes, pretrained_embeddings)


dis_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
dis_optimizer = tf.train.AdamOptimizer(1e-4)
dis_grads_and_vars = dis_optimizer.compute_gradients(dis.loss, dis_params)
dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pretrain 
for i in range(5):
    epoch_loss = gen.pretrain_one_epoch(sess, data_loader)
    print epoch_loss
    tloss = target_lstm.target_loss(sess, gen, 64, data_loader)
    print 'tloss: ' + str(tloss)

for i in tqdm(range(k)):
    # minibatches of real training data ... do they mean 1 or all minibatches??
    real_minibatch = data_loader.next_batch()
    # minibatch, get first N from real_minibatch, generate the rest
    gen_minibatch = gen.generate_from_latch(sess, real_minibatch, N)
    loss = dis.train_one_step(sess, data_loader, real_minibatch, gen_minibatch)
    print 'dis ' + str(loss)

while N - K >= 0:
    N = N - K
    for i in range(k):
        # minibatches of real training data ... do they mean 1 or all minibatches??
        real_minibatch = data_loader.next_batch()
        # minibatch, get first N from real_minibatch, generate the rest
        gen_minibatch = gen.generate_from_latch(sess, real_minibatches, N)
        loss = dis.train_one_step(sess, data_loader, real_minibatch, gen_minibatch)
        print 'dis ' + str(loss)    

    # minibatch of real training data
    new_minibatch = data_loader.next_batch()
    xij = gen.generate_xij(sess, new_minibatch, N)
    print gen.train_one_step(sess, dis, xij,  N) / (T-N)
    tloss = target_lstm.target_loss(sess, gen, 64, data_loader)
    print 'tloss: ' + str(tloss)
