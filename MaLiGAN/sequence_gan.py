import model
import numpy as np
import tensorflow as tf
import random
import time
from gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from dis_dataloader import Dis_dataloader
from text_classifier import TextCNN
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import cPickle, yaml
import time, os
from config import GenConfig, DisConfig
TIME = time.strftime('%Y%m%d-%H%M%S')

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
GCONFIG = GenConfig()
##########################################################################################

TOTAL_BATCH = 10

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
DCONFIG = DisConfig()
#########################################################################################

positive_file = 'save/real_data.txt'
negative_file = 'target_generate/generator_sample.txt'
eval_file = 'target_generate/eval_file.txt'

generated_num = 10000

##############################################################################################


class PoemGen(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, GCONFIG.BATCH_SIZE, GCONFIG.EMB_DIM, GCONFIG.HIDDEN_DIM, \
        GCONFIG.SEQ_LENGTH, GCONFIG.START_TOKEN)


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
    random.seed(GCONFIG.SEED)
    np.random.seed(GCONFIG.SEED)

    assert GCONFIG.START_TOKEN == 0

    ### LOAD DATA ###
    gen_data_loader = Gen_Data_loader(GCONFIG.BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(GCONFIG.BATCH_SIZE)
    vocab_size = 5000
    dis_data_loader = Dis_dataloader()
    #################



    ### CREATE MODELS (VARIABLES) ###
    best_score = 1000
    generator = get_trainable_model(vocab_size)
    target_params = cPickle.load(open('save/target_params.pkl'))
    target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)
    gen_params = [param for param in tf.trainable_variables() if 'generator' in param.name]

    with tf.variable_scope('discriminator'):
        cnn = TextCNN(
            sequence_length=20,
            num_classes=2,
            vocab_size=vocab_size,
            embedding_size=DCONFIG.DIS_EMBEDDING_DIM,
            filter_sizes=DCONFIG.DIS_FILTER_SIZES,
            num_filters=DCONFIG.DIS_NUM_FILTERS,
            l2_reg_lambda=DCONFIG.DIS_12_REG_LAMBDA)

    cnn_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
    # Define Discriminator Training procedure
    dis_global_step = tf.Variable(0, name="global_step", trainable=False)
    dis_optimizer = tf.train.AdamOptimizer(1e-4)
    dis_grads_and_vars = dis_optimizer.compute_gradients(cnn.loss, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)
    #######################################
    gensaver = tf.train.Saver(gen_params)
    discsaver = tf.train.Saver(cnn_params)
    ######################



    ### START SESSION ###
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    generate_samples(sess, target_lstm, 64, 10000, positive_file)
    gen_data_loader.create_batches(positive_file)

    '''
    # if no checkpoint file
    log = open('log/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(GCONFIG.PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, GCONFIG.BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            buffer = str(epoch) + ' ' + str(test_loss) + '\n'
            log.write(buffer)

    generate_samples(sess, generator, GCONFIG.BATCH_SIZE, generated_num, eval_file)
    likelihood_data_loader.create_batches(eval_file)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    buffer = 'After pre-training:' + ' ' + str(test_loss) + '\n'
    log.write(buffer)

    generate_samples(sess, generator, GCONFIG.BATCH_SIZE, generated_num, eval_file)
    likelihood_data_loader.create_batches(eval_file)
    significance_test(sess, target_lstm, likelihood_data_loader, 'significance/supervise.txt')

    print 'Start training discriminator...'
    for i in range(DCONFIG.DIS_ALTER_EPOCH):
	print "Starting epoch " + str(i)
        generate_samples(sess, generator, GCONFIG.BATCH_SIZE, generated_num, negative_file)

        #  train discriminator
        dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
        dis_batches = dis_data_loader.batch_iter(
            zip(dis_x_train, dis_y_train), DCONFIG.DIS_BATCH_SIZE, DCONFIG.DIS_NUM_EPOCHS
        )

        for batch in dis_batches:
            try:
                x_batch, y_batch = zip(*batch)
                feed = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: DCONFIG.DIS_DROPOUT_KEEP_PROB
                }
                _, step = sess.run([dis_train_op, dis_global_step], feed)
            except ValueError:
                pass

    discsaver.save(sess, 'save/disc-sess-'+TIME+'.ckpt')
    '''

    print '#########################################################################'
    print 'Restoring old generator/discriminator training sessions...'

    gensaver.restore(sess, './save/20170305-232733-seqgan/seqgan-gen-sess-20170305-232733.ckpt')
    discsaver.restore(sess, './save/20170305-232733-seqgan/seqgan-disc-sess-20170305-232733.ckpt')
    losses = cPickle.load(open('./save/20170305-232733-seqgan/seqgan-loss-20170305-232733.pkl'))
    num_prev_points, last_batch_num = len(losses), losses[-1, 0]
    losses = np.concatenate((losses, np.zeros((TOTAL_BATCH, 2))), axis=0)

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Reinforcement Training Generator...'

    if not os.path.exists('./save/'+TIME+'-seqgan'):
        os.makedirs('./save/'+TIME+'-seqgan')
    with open('./save/'+TIME+'-seqgan/seqgan-config-'+TIME+'.yaml', 'w') as f:
        yaml.dump(GCONFIG, f)
        yaml.dump(DCONFIG, f)
        yaml.dump(TOTAL_BATCH, f)

    for total_batch in range(TOTAL_BATCH):
        
        for it in range(GCONFIG.TRAIN_ITER):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, cnn)
            feed = {generator.x: samples, generator.rewards: rewards}
            _, g_loss = sess.run([generator.g_updates, generator.g_loss], feed_dict=feed)

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, GCONFIG.BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            losses[num_prev_points + total_batch] = [last_batch_num + total_batch, test_loss]

            if test_loss < best_score:
                best_score = test_loss
                print 'best score: ', test_loss
                significance_test(sess, target_lstm, likelihood_data_loader, 'significance/seqgan.txt')

        rollout.update_params()

        # generate for discriminator
        print 'Start training discriminator'
        for _ in range(5):
            generate_samples(sess, generator, GCONFIG.BATCH_SIZE, generated_num, negative_file)

            dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
            dis_batches = dis_data_loader.batch_iter(zip(dis_x_train, dis_y_train), DCONFIG.DIS_BATCH_SIZE, 3)

            for batch in dis_batches:
                try:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: DCONFIG.DIS_DROPOUT_KEEP_PROB
                    }
                    _, step = sess.run([dis_train_op, dis_global_step], feed)
                except ValueError:
                    pass

        ### Save session and loss to disk ###
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            gensaver.save(sess, 'save/'+TIME+'-seqgan/seqgan-gen-sess-' + TIME + '.ckpt')
            discsaver.save(sess, 'save/'+TIME+'-seqgan/seqgan-disc-sess-' + TIME + '.ckpt')
            #with open('save/seqgan-loss-' + TIME + '.pkl', 'w') as f:
    	    with open('./save/'+TIME+'-seqgan/seqgan-loss-' + TIME + '.pkl', 'w') as f:
                cPickle.dump(losses, f, -1)

    sess.close()


if __name__ == '__main__':
    main()
