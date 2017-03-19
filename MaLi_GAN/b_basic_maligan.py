import tensorflow as tf
import numpy as np
from data_loader import DataLoader as dl
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
import cPickle
from target_lstm import TARGET_LSTM
import time, os
TIME = time.strftime('%Y%m%d-%H%M%S')

# initialize constants
T = 20
N = T
K = 1
k = 10
DROPOUT_KEEP_PROB = 0.75
batch_size = 32
embedding_size = 300
n_classes = 2

# load real data
positive_file = 'save/real_data.txt'
pos_dl = dl(N, batch_size, True, positive_file)

eval_file = 'save/eval_data.txt'
eval_dl = dl(N, batch_size, True, eval_file)

with tf.variable_scope('embeddings'):
    pretrained_embeddings = tf.get_variable('embeddings', initializer=tf.random_normal([5000, 300]))

# initialize generator and discriminator
with tf.variable_scope('generator'):
    gen = Generator(5000, batch_size, 300, 150, T, 0, 20, pretrained_embeddings)
with tf.variable_scope('discriminator'):
    dis = Discriminator(N, batch_size, pretrained_embeddings)

saver = tf.train.Saver()
gensaver = tf.train.Saver([param for param in tf.trainable_variables() if 'generator' in param.name or 'embeddings' in param.name] )
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('./'+TIME):
    os.makedirs('./'+TIME)

#gensaver.restore(sess, './test1/pretrained_eval')
# pretrain
pretrained = False
if pretrained:
    oldtime = '20170318-004737'
    perps = cPickle.load(open('./'+oldtime+'/pretrain_perplexities.txt'))
    saver.restore(sess, './'+oldtime+'/pretrained')
else:
    perps = []
    for i in range(500):
        gen.pretrain_one_epoch(sess, pos_dl)
        perp = gen.get_perplexity(sess, eval_dl)
        print perp
        perps.append(perp)

        if i % 5 == 0:
            with open(TIME + '/pretrain_perplexities.txt', 'w') as f:
                cPickle.dump(perps, f)
            saver.save(sess, './'+TIME + '/pretrained')

    with open(TIME + '/pretrain_perplexities.txt', 'w') as f:
        cPickle.dump(perps, f)
    saver.save(sess, './'+TIME+'/pretrained')


#gensaver = tf.train.Saver([param for param in tf.trainable_variables() if 'generator' in param.name])
#gensaver.restore(sess, './pretrained')
#saver.restore(sess, './trained')

dpretrained = False
if dpretrained:
    acc = []
    saver.restore(sess, './20170318-011742/pretrained_disc')
else:
    acc = []
    for _ in range(100):
        accuracies=[]
        for i in range(100):
            real_minibatch, _ = pos_dl.next_batch()
            gen_minibatch = gen.generate(sess, batch_size)
            loss, accuracy, output = dis.train_one_step(sess, real_minibatch, gen_minibatch)
            accuracies.append(accuracy)
        mean_acc =  np.mean(accuracies)
        print mean_acc
        acc.append(mean_acc)
        if mean_acc > 0.8:
            break

    with open(TIME + '/pretrain_accuracies.txt', 'w') as f:
        cPickle.dump(acc, f)
    saver.save(sess, './'+TIME+'/pretrained_disc')

perp = gen.get_perplexity(sess, eval_dl)
perps.append(perp)
print 'perp: ' + str(perp)

for _ in range(10000):
    print 'discriminator...'
    for i in range(k):
        real_minibatch, _ = pos_dl.next_batch()
        gen_minibatch = gen.generate(sess, batch_size)
        loss, accuracy, output = dis.train_one_step(sess, real_minibatch, gen_minibatch)
        acc.append(accuracy)

    print 'generator...'
    for i in range(5):
        gen_minibatch = gen.generate(sess, 32)
        loss, partial = gen.train_one_step(sess, dis, gen_minibatch)

        gen_minibatch = gen.generate(sess, 32)
        real_minibatch, _ = pos_dl.next_batch()
        accuracy = dis.get_accuracy(sess, real_minibatch, gen_minibatch)
        acc.append(accuracy)
        print 'acc: ' + str(accuracy)

    perp = gen.get_perplexity(sess, eval_dl)
    print 'perp: ' + str(perp)
    perps.append(perp)
    with open(TIME + '/accuracies.txt', 'w') as f:
        cPickle.dump(acc, f)
    with open(TIME + '/eval_perps.txt', 'w') as f:
        cPickle.dump(perps, f)
    saver.save(sess, './'+TIME +'/trained')

with open(TIME + '/accuracies.txt', 'w') as f:
    cPickle.dump(acc, f)

with open(TIME + '/eval_perps.txt', 'w') as f:
    cPickle.dump(perps, f)

saver.save(sess, './'+TIME +'/trained')
