import tensorflow as tf
import numpy as np
from data_loader import DataLoader as dl
from tqdm import tqdm
from generator import Generator
import nltk
from discriminator import Discriminator
import cPickle
from target_lstm import TARGET_LSTM
import time, os
TIME = time.strftime('%Y%m%d-%H%M%S')

# initialize constants
T = 35
N = T
K = 1
k = 100
DROPOUT_KEEP_PROB = 0.75
batch_size = 64
embedding_size = 300
n_classes = 2

# populate the lexicon of existing words
lexicon = cPickle.load(open('../data/glove/trimmed_word_lexicon.p', 'r'))
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

saver = tf.train.Saver()
gensaver = tf.train.Saver([param for param in tf.trainable_variables() if 'generator' in param.name or 'embeddings' in param.name] )
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('./'+TIME):
    os.makedirs('./'+TIME)

#gensaver.restore(sess, './test1/pretrained_eval')
# pretrain
print 'Starting generator pretraining...'
pretrained = True
if pretrained:
    oldtime = '20170321-045805' #'20170318-004737'
    perps = cPickle.load(open('./'+oldtime+'/pretrain_perplexities.txt'))
    saver.restore(sess, './'+oldtime+'/pretrained')
else:
    perps = []
    for i in range(100):
        gen.pretrain_one_epoch(sess, pos_dl)
        perp = gen.get_perplexity(sess, eval_dl)
        print "perplexity: " + str(perp)
        perps.append(perp)

        if i % 5 == 0:
            with open(TIME + '/pretrain_perplexities.txt', 'w') as f:
                cPickle.dump(perps, f)
            saver.save(sess, './'+TIME + '/pretrained')

    with open(TIME + '/pretrain_perplexities.txt', 'w') as f:
        cPickle.dump(perps, f)
    saver.save(sess, './'+TIME+'/pretrained')

print 'Starting discriminator pretraining...'
dpretrained = True
if dpretrained:
    acc = []
    saver.restore(sess, './20170321-045805/pretrained_disc')#'./20170318-011742/pretrained_disc')
else:
    acc = []
    for _ in range(100):
        accuracies=[]
        for i in range(100):
            real_minibatch, _, _ = pos_dl.next_batch()
            gen_minibatch = gen.generate(sess, batch_size)
            loss, accuracy, output = dis.train_one_step(sess, real_minibatch, gen_minibatch)
            accuracies.append(accuracy)
        mean_acc =  np.mean(accuracies)
        print 'accuracy: ' + str(mean_acc)
        acc.append(mean_acc)
        if mean_acc > 0.97:
            break

    with open(TIME + '/pretrain_accuracies.txt', 'w') as f:
        cPickle.dump(acc, f)
    saver.save(sess, './'+TIME+'/pretrained_disc')

perp = gen.get_perplexity(sess, eval_dl)
perps.append(perp)
print 'perp: ' + str(perp)

print 'Starting GAN training...'
for _ in range(10000):
    accuracy = 0.0
    print 'discriminator...'
    for i in range(k):
        real_minibatch, _, _ = pos_dl.next_batch()
        gen_minibatch = gen.generate(sess, batch_size)
        
        loss, accuracy, output = dis.train_one_step(sess, real_minibatch, gen_minibatch)
        acc.append(accuracy)

    print 'generator...'
    for i in range(5):
        gen_minibatch = gen.generate(sess, 35)
        loss, partial = gen.train_one_step(sess, dis, gen_minibatch)

        gen_minibatch = gen.generate(sess, 35)
        real_minibatch, _, _ = pos_dl.next_batch()
        accuracy = dis.get_accuracy(sess, real_minibatch, gen_minibatch)
        acc.append(accuracy)
        print 'acc: ' + str(accuracy)

    sentences = gen.generate(sess, 5)
    english = []
    for sentence in sentences:
        references, _, _= pos_dl.next_batch()
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(references, list(sentence), (1.0 / 3, 1.0 / 3, 1.0 / 3))
        print "BLEUscore: " + str(BLEUscore)
        temp = []
        for idx in sentence:
            temp.append(reverse_lexicon[idx])
        temp_sent = ' '.join(temp)
        print temp_sent
        english.append(temp_sent)
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
