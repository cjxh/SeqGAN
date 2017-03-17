import numpy as np
import tensorflow as tf
import random
import time
from target_lstm import TARGET_LSTM
import cPickle, yaml
import time, os
TIME = time.strftime('%Y%m%d-%H%M%S')

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
##########################################################################################

TOTAL_BATCH = 10

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
#########################################################################################

positive_file = 'save/real_data.txt'
negative_file = 'target_generate/generator_sample.txt'
eval_file = 'target_generate/eval_file.txt'

generated_num = 10000

##############################################################################################




def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    #  Generated Samples
    generated_samples = []
    start = time.time()
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    end = time.time()
    # print 'Sample generation time:', (end - start)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            # buffer = u''.join([words[x] for x in poem]).encode('utf-8') + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def significance_test(sess, target_lstm, data_loader, output_file):
    loss = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.out_loss, {target_lstm.x: batch})
        loss.extend(list(g_loss))
    with open(output_file, 'w')as fout:
        for item in loss:
            buffer = str(item) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(88)
    np.random.seed(88)

    #assert GCONFIG.START_TOKEN == 0
    vocab_size = 5000

    target_params = cPickle.load(open('save/target_params.pkl'))
    target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)
    



    ### START SESSION ###
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    generate_samples(sess, target_lstm, 64, 10000, positive_file)
    #gen_data_loader.create_batches(positive_file)


if __name__ == '__main__':
    main()
