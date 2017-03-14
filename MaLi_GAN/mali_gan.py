import tensorflow as tf
import numpy as np
from data_loader import DataLoader as dl
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
# import evaluation

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
def target_loss(sess, target_lstm, generator, batch_size):
    for it in xrange(data_loader.num_batch):
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: generator.generate(sess, batch_size)})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

# def generate_samples_to_file(sess, generator, generated_num, output_file):
#     #  Generated Samples
#     generated_samples = generator.generate(sess, generated_num)

#     with open(output_file, 'w') as fout:
#         for poem in generated_samples:
#             buffer = ' '.join([str(x) for x in poem]) + '\n'
#             # buffer = u''.join([words[x] for x in poem]).encode('utf-8') + '\n'
#             fout.write(buffer)

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
    tloss = target_loss(sess, target_lstm, gen, 64)
    print 'tloss: ' + str(tloss)


for i in tqdm(range(k)):
    # minibatches of real training data ... do they mean 1 or all minibatches??
    real_minibatches = data_loader.next_batch()
    # minibatch, get first N from real_minibatch, generate the rest
    gen_minibatches = gen.generate_from_latch(sess, real_minibatches, N)
    dis_x_train = np.concatenate((real_minibatches, gen_minibatches), axis=0)
    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))
    dis_y_real = np.concatenate((real, fake), axis=1)
    dis_y_fake = np.concatenate((fake, real), axis=1)
    dis_y_train = np.concatenate((dis_y_real, dis_y_fake), axis=0)

    shuffle_idx = np.random.permutation(np.arange(2 * batch_size))
    shuffled_x =  dis_x_train[shuffle_idx]
    shuffled_y =  dis_y_train[shuffle_idx]
    
    feed = {
        dis.input_x: shuffled_x,
        dis.input_y: shuffled_y,
        dis.dropout_keep_prob: DROPOUT_KEEP_PROB
    }
    _, loss = sess.run([dis_train_op,dis.loss] , feed)
    print 'dis ' + str(loss)

while N - K >= 0:
    N = N - K
    for i in range(k):
        # minibatches of real training data ... do they mean 1 or all minibatches??
        real_minibatches = data_loader.next_batch()
        # minibatch, get first N from real_minibatch, generate the rest
        gen_minibatches = gen.generate_from_latch(sess, real_minibatches, N)
        dis_x_train = np.concatenate((real_minibatches, gen_minibatches), axis=0)
        real = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        dis_y_real = np.concatenate((real, fake), axis=1)
        dis_y_fake = np.concatenate((fake, real), axis=1)
        dis_y_train = np.concatenate((dis_y_real, dis_y_fake), axis=0)

        shuffle_idx = np.random.permutation(np.arange(2 * batch_size))
        shuffled_x =  dis_x_train[shuffle_idx]
        shuffled_y =  dis_y_train[shuffle_idx]
    
        feed = {
            dis.input_x: shuffled_x,
            dis.input_y: shuffled_y,
            dis.dropout_keep_prob: DROPOUT_KEEP_PROB
        }
        _, loss = sess.run([dis_train_op, dis.loss], feed)
        print 'dis ' + str(loss)    

    # minibatch of real training data
    new_minibatch = data_loader.next_batch()
    xij = gen.generate_xij(sess, new_minibatch, N)
    print gen.train_one_step(sess, dis, xij,  N) / (T-N)
