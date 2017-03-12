import tensorflow as tf
import numpy as np
import data_loader as dl
from tqdm import tqdm
# import generator
import Discriminator
# import evaluation

def train_discriminator():

# initialize constants
T = 35
N = T
K = 5
k = 5
DROPOUT_KEEP_PROB = 0.75
batch_size = 32
embedding_size = 300
n_classes = 2

# populate the lexicon of existing words
lexicon = {}
counter = 0
with open('word_lexicon.txt', 'r') as f:
    print "Pre-processing and saving word lexicon in memory..."
    for line in tqdm(f):
        for word in line:
            lexicon[word] = counter
            counter += 1

# load real data
positive_file = 'save/real_data.txt'
data_loader = dl(lexicon, N)
print "Loading data from " + positive_file + " into memory..."
positive_data = data_loader.load_data(positive_file)
pretrained_embeddings = np.load('data/glove_vectors.npy')

# initialize generator and discriminator
gen = Generator()
with tf.variable_scope('discriminator'):
    dis = Discriminator(N, batch_size, embedding_size, n_classes, pretrained_embeddings)
dis_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
dis_global_step = tf.Variable(0, name="global_step", trainable=False)
dis_optimizer = tf.train.AdamOptimizer(1e-4)
dis_grads_and_vars = dis_optimizer.compute_gradients(dis.loss, dis_params, aggregation_method=2)
dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)

sess = tf.Session()
sess.run(tf.global_variable_initializer())

# pretrain 
# generator.pretrain()
# discriminator.pretrain()
for i in range(k):
    # minibatches of real training data ... do they mean 1 or all minibatches??
    real_minibatches = data_loader.mini_batch(batch_size)
    # minibatch, get first N from real_minibatch, generate the rest
    gen_minibatch = generator.generate_from_latch(sess, real_minibatches, N)
    dis_batches = zip(dis_x_train, dis_y_train)
    for batch in dis_batches:
        x_batch, y_batch = zip(*batch)
        feed = {
            discriminator.input_x: x_batch,
            discriminator.input_y: y_batch,
            discriminator.dropout_keep_prob: DROPOUT_KEEP_PROB
        }
        _, step = sess.run([dis_train_op, dis_global_step], feed)

while N >= 0:
    N = N - K
    for i in range(k):
        # minibatches of real training data ... do they mean 1 or all minibatches??
        real_minibatches = data_loader.mini_batch(batch_size)
        # minibatch, get first N from real_minibatch, generate the rest
        gen_minibatch = generator.generate_from_latch(sess, real_minibatches, N)
        dis_batches = zip(dis_x_train, dis_y_train)
        for batch in dis_batches:
            x_batch, y_batch = zip(*batch)
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: DROPOUT_KEEP_PROB
            }
            _, step = sess.run([dis_train_op, dis_global_step], feed)
    
    # minibatch of real training data
    new_minibatch = real_data_loader.mini_batch(positive_file)
    x_ij = generator.generate_from_latch(new_minibatch, N)
    generator.update_params()
