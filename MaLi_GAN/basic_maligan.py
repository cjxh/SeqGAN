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
k = 100
DROPOUT_KEEP_PROB = 0.75
batch_size = 32
embedding_size = 300
n_classes = 2

# populate the lexicon of existing words
lexicon = {}

# load real data
positive_file = 'save/real_data.txt'
data_loader = dl(lexicon, N, batch_size)
print "Loading data from " + positive_file + " into memory..."
positive_data = data_loader.load_data(positive_file)

pretrained_embeddings = tf.get_variable('embeddings', initializer=tf.random_normal([5000, 300]))

# initialize generator and discriminator
target_params = cPickle.load(open('save/target_params.pkl'))
target_lstm = TARGET_LSTM(5000, batch_size, 32, 32, 20, 0, target_params)   

with tf.variable_scope('generator'):
    gen = Generator(5000, batch_size, 300, 150, T, 0, 20, pretrained_embeddings)
with tf.variable_scope('discriminator'):
    dis = Discriminator(N, batch_size, n_classes, pretrained_embeddings)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pretrain 
for i in range(3):
    epoch_loss = gen.pretrain_one_epoch(sess, data_loader)
    print epoch_loss
    tloss = target_lstm.target_loss(sess, gen, batch_size, data_loader)
    print 'tloss: ' + str(tloss)

accuracies=[]
for i in range(k):
    real_minibatch = data_loader.next_batch()
    gen_minibatch = gen.generate(sess, batch_size)
    loss, accuracy, output = dis.train_one_step(sess, real_minibatch, gen_minibatch)
    #print 'dis loss: ' + str(loss) + ', dis accuracy' + str(accuracy)
    accuracies.append(accuracy)
print np.mean(accuracies)

for _ in range(10):
    accuracies=[]
    for i in range(k):
        real_minibatch = data_loader.next_batch()
        gen_minibatch = gen.generate(sess, batch_size)
        loss, accuracy, output = dis.train_one_step(sess, real_minibatch, gen_minibatch)
        #print 'dis loss: ' + str(loss) + ', dis accuracy' + str(accuracy)
        accuracies.append(accuracy)
    print np.mean(accuracies)

    gen_minibatch = gen.generate(sess, batch_size)
    gen.train_one_step(sess, dis, gen_minibatch)

    tloss = target_lstm.target_loss(sess, gen, batch_size, data_loader)
    print 'tloss: ' + str(tloss)
