import model_real
import numpy as np
import tensorflow as tf
import random
#from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from data_loader import DataLoader as dl
from target_lstm import TARGET_LSTM
import cPickle, yaml
import time, os
from config import GenConfig
TIME = time.strftime('%Y%m%d-%H%M%S')

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
CONFIG = GenConfig()
CONFIG.PRE_EPOCH_NUM = 50

##########################################################################################
positive_file = '../data/preprocessed_data/train.txt'
eval_file = '../data/preprocessed_data/eval.txt'

generated_num = 10000

class PoemGen(model_real.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, CONFIG.BATCH_SIZE, 300, CONFIG.HIDDEN_DIM, CONFIG.SEQ_LENGTH, \
        CONFIG.START_TOKEN)


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch, mask = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch, mask)
        supervised_g_losses.append(g_loss)

    print '>>>> generator train loss:', np.mean(supervised_g_losses)
    loss = np.mean(supervised_g_losses)
    return loss, np.exp(loss)

def main():
    #random.seed(CONFIG.SEED)
    #np.random.seed(CONFIG.SEED)

    #assert CONFIG.START_TOKEN == 0
    lexicon = cPickle.load(open('../data/glove/trimmed_word_lexicon.p', 'r'))
    vocab_size = len(lexicon.keys())
    seqlen = 35
    batch_size = 64 

    #gen_data_loader = DataLoader(CONFIG.BATCH_SIZE)
    gen_data_loader = dl(seqlen, batch_size, False, positive_file, lexicon)
    #likelihood_data_loader = DataLoader(CONFIG.BATCH_SIZE)
    eval_dl = dl(seqlen, batch_size, False, eval_file, lexicon)

    generator = get_trainable_model(vocab_size)

    #target_params = cPickle.load(open('save/target_params.pkl'))
    #target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    #  pre-train generator
    print 'Start pre-training...'
    
    for epoch in xrange(CONFIG.PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        

    sess.close()


if __name__ == '__main__':
    main()
