import model
import numpy as np
import tensorflow as tf
import random
from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from target_lstm import TARGET_LSTM
import cPickle, yaml
import time
from config import GenConfig
TIME = time.strftime('%Y%m%d-%H%M%S')

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
CONFIG = GenConfig()
CONFIG.PRE_EPOCH_NUM = 50
with open('save/mle-config-'+TIME+'.yaml', 'w') as f:
    yaml.dump(CONFIG, f)

##########################################################################################
positive_file = 'save/real_data.txt'
negative_file = 'target_generate/generator_sample.txt'
eval_file = 'target_generate/eval_file.txt'

generated_num = 10000

class PoemGen(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, CONFIG.BATCH_SIZE, CONFIG.EMB_DIM, CONFIG.HIDDEN_DIM, CONFIG.SEQ_LENGTH, \
        CONFIG.START_TOKEN)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    #  Generated Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


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
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    print '>>>> generator train loss:', np.mean(supervised_g_losses)
    return np.mean(supervised_g_losses)


def main():
    random.seed(CONFIG.SEED)
    np.random.seed(CONFIG.SEED)

    assert CONFIG.START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(CONFIG.BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(CONFIG.BATCH_SIZE)
    vocab_size = 5000

    generator = get_trainable_model(vocab_size)

    target_params = cPickle.load(open('save/target_params.pkl'))
    target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    generate_samples(sess, target_lstm, 64, 10000, positive_file)
    gen_data_loader.create_batches(positive_file)

    #  pre-train generator
    print 'Start pre-training...'
    losses = np.zeros((CONFIG.PRE_EPOCH_NUM / 5 + 1, 2))
    for epoch in xrange(CONFIG.PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)

        # Every 5 epochs, save current loss to np array
        if epoch % 5 == 0:
            generate_samples(sess, generator, CONFIG.BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            losses[epoch / 5] = [epoch, test_loss]

        # Every 50 epochs, save loss and session to disk
        if epoch % 100 == 0:
            #saver.save(sess, 'save/mle-sess-'+TIME+'.ckpt')
            with open('save/mle-loss-' + TIME + '.pkl', 'w') as f:
                cPickle.dump(losses, f, -1)

    generate_samples(sess, generator, CONFIG.BATCH_SIZE, generated_num, eval_file)
    likelihood_data_loader.create_batches(eval_file)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    losses[CONFIG.PRE_EPOCH_NUM / 5] = [CONFIG.PRE_EPOCH_NUM, test_loss]

    # Save final loss and session to disk
    saver.save(sess, 'save/mle-sess-'+TIME+'.ckpt')
    with open('save/mle-loss-' + TIME + '.pkl', 'w') as f:
        cPickle.dump(losses, f, -1)

    sess.close()


if __name__ == '__main__':
    main()
