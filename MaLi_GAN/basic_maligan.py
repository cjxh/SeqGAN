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
data_loader = dl(N, batch_size, True, positive_file, lexicon)
print "Loading data from " + positive_file + " into memory..."
positive_data = data_loader.mini_batch(batch_size)

pretrained_embeddings = tf.get_variable('embeddings', initializer=tf.random_normal([5000, 300]))

# initialize generator and discriminator
target_params = cPickle.load(open('save/target_params.pkl'))
target_lstm = TARGET_LSTM(5000, batch_size, 32, 32, 20, 0, target_params)   

with tf.variable_scope('generator'):
    gen = Generator(5000, batch_size, 300, 150, T, 0, 20, pretrained_embeddings)
with tf.variable_scope('discriminator'):
    dis = Discriminator(N, batch_size, n_classes, pretrained_embeddings)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pretrain 
'''losses = []
for i in range(20):
    epoch_loss = gen.pretrain_one_epoch(sess, data_loader)
    print epoch_loss
    losses.append(epoch_loss)
    #tloss = target_lstm.target_loss(sess, gen, batch_size, data_loader)
    #print 'tloss: ' + str(tloss)

with open('pretrain_losses.txt', 'w') as f:
    cPickle.dump(losses, f)
saver.save(sess, 'pretrained')'''
gensaver = tf.train.Saver([param for param in tf.trainable_variables() if 'generator' in param.name])
#gensaver.restore(sess, './pretrained')
saver.restore(sess, './trained')

'''for _ in range(10):
    accuracies=[]
    for i in range(k):
        real_minibatch = data_loader.next_batch()
        gen_minibatch = gen.generate(sess, batch_size)
        loss, accuracy, output = dis.train_one_step(sess, real_minibatch, gen_minibatch)
        #print 'dis loss: ' + str(loss) + ', dis accuracy' + str(accuracy)
        accuracies.append(accuracy)
    print np.mean(accuracies)'''

tlosses = []
tloss = target_lstm.target_loss(sess, gen, batch_size, data_loader)
tlosses.append(tloss)
print 'tloss: ' + str(tloss)

acc = []
for _ in range(10):
    print 'discriminator...'
    for _ in range(100):
        accuracies=[]
        for i in range(k):
            real_minibatch = data_loader.next_batch()
            gen_minibatch = gen.generate(sess, batch_size)
            loss, accuracy, output = dis.train_one_step(sess, real_minibatch, gen_minibatch)
            #print 'dis loss: ' + str(loss) + ', dis accuracy' + str(accuracy)
            accuracies.append(accuracy)
            #print accuracy
        mean_acc =  np.mean(accuracies)
        print mean_acc
        acc.append(mean_acc)
        if mean_acc > 0.8:
            break

    print 'generator....'
    for _ in range(10):
        losses=[]
        for i in range(5):
            gen_minibatch = gen.generate(sess, 128)
            loss, partial = gen.train_one_step(sess, dis, gen_minibatch)
            losses.append(loss)
        print np.mean(losses)

    tloss = target_lstm.target_loss(sess, gen, batch_size, data_loader)
    print 'tloss: ' + str(tloss)
    tlosses.append(tloss)
    with open('accuracies3.txt', 'w') as f:
        cPickle.dump(acc, f)
    with open('eval_losses3.txt', 'w') as f:
        cPickle.dump(tlosses, f)

with open('accuracies3.txt', 'w') as f:
    cPickle.dump(acc, f)

with open('eval_losses3.txt', 'w') as f:
    cPickle.dump(tlosses, f)

saver.save(sess, 'trained')
