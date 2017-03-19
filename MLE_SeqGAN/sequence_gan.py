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
import cPickle, tqdm
import time, os
TIME = time.strftime('%Y%m%d-%H%M%S')

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 32
HIDDEN_DIM = 32
SEQ_LENGTH = 20
START_TOKEN = 0

PRE_EPOCH_NUM = 300
TRAIN_ITER = 1  # generator
SEED = 88
BATCH_SIZE = 64
##########################################################################################

TOTAL_BATCH = 800

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2

# Training parameters
dis_batch_size = 64
dis_num_epochs = 3
dis_alter_epoch = 50

positive_file = 'save/real_data.txt'
negative_file = 'target_generate/generator_sample.txt'
eval_file = 'target_generate/eval_file.txt'
target_file = 'target_generate/target_file.txt'

generated_num = 10000


##############################################################################################


class PoemGen(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return PoemGen(num_emb, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

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


def target_loss(sess, generator, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(generator.pretrain_loss, {generator.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Likelihood_data_loader(BATCH_SIZE)
    target_data_loader = Likelihood_data_loader(BATCH_SIZE)
    vocab_size = 5000
    dis_data_loader = Dis_dataloader()

    best_score = 10000
    generator = get_trainable_model(vocab_size)
    target_params = cPickle.load(open('save/target_params.pkl'))
    target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)

    with tf.variable_scope('discriminator'):
        cnn = TextCNN(
            sequence_length=20,
            num_classes=2,
            vocab_size=vocab_size,
            embedding_size=dis_embedding_dim,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            l2_reg_lambda=dis_l2_reg_lambda)

    cnn_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
    # Define Discriminator Training procedure
    dis_global_step = tf.Variable(0, name="global_step", trainable=False)
    dis_optimizer = tf.train.AdamOptimizer(1e-4)
    dis_grads_and_vars = dis_optimizer.compute_gradients(cnn.loss, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    #generate_samples(sess, target_lstm, 64, 10000, positive_file)
    gen_data_loader.create_batches(positive_file)

    #generate_samples(sess, target_lstm, 64, 7000, eval_file)
    likelihood_data_loader.create_batches(eval_file)



    if not os.path.exists('./data/'+TIME):
        os.makedirs('./data/'+TIME)

    #  pre-train generator
    pretrained = False
    if pretrained:
        oldtime = '20170319-012131'
        perps = cPickle.load(open('./data/'+oldtime+'/pretrain_perps_30.txt'))
        saver.restore(sess, './data/'+oldtime+'/pretrained_30')
    else:
        print 'Start pre-training...'
        perps=[]
        oraclelosses=[]
        for epoch in xrange(PRE_EPOCH_NUM):
            print 'pre-train epoch:', epoch
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            if epoch % 5 == 0:
                test_perp = np.exp(target_loss(sess, generator, eval_file))
                perps.append(test_perp)

                generate_samples(sess, generator, BATCH_SIZE, generated_num, target_file)
                target_data_loader.create_batches(target_file)
                test_loss = target_loss(sess, target_lstm, target_data_loader)
                oraclelosses.append(test_loss)
                print 'pre-train epoch ', epoch, 'test_perp ', test_perp, 'test loss', test_loss
            if epoch == 150:
                with open('./data/'+TIME + '/pretrain_perps_150.txt', 'w') as f:
                    cPickle.dump(perps, f)
                with open('./data/'+TIME + '/pretrain_losses_150.txt', 'w') as f:
                    cPickle.dump(oraclelosses, f)
                saver.save(sess, './data/'+TIME + '/pretrained_150')

        test_perp = np.exp(target_loss(sess, generator, eval_file))
        perps.append(test_perp)

        generate_samples(sess, generator, BATCH_SIZE, generated_num, target_file)
        target_data_loader.create_batches(target_file)
        test_loss = target_loss(sess, target_lstm, target_data_loader)
        oraclelosses.append(test_loss)
        with open('./data/'+TIME + '/pretrain_perps_300.txt', 'w') as f:
            cPickle.dump(perps, f)
        with open('./data/'+TIME + '/pretrain_losses_300.txt', 'w') as f:
            cPickle.dump(oraclelosses, f)

    quit()

    dpretrained = False
    if dpretrained:
        oldtime = '20170319-015507'
        saver.restore(sess, './data/'+oldtime+'/dpretrained')
    else:
        print 'Start training discriminator...'
        accuracies=[]
        for _ in tqdm(range(dis_alter_epoch)):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)

            #  train discriminator
            dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
            dis_batches = dis_data_loader.batch_iter(
                zip(dis_x_train, dis_y_train), dis_batch_size, dis_num_epochs
            )

            batch_accs=[]
            for batch in dis_batches:
                try:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, step, acc = sess.run([dis_train_op, dis_global_step, cnn.accuracy], feed)
                    batch_accs.append(acc)
                except ValueError:
                    pass

            accuracies.append(np.mean(batch_accs))
            print 'acc: ', np.mean(batch_accs)

        with open('./data/'+TIME + '/pretrain_accuracies.txt', 'w') as f:
            cPickle.dump(accuracies, f)
        saver.save(sess, './data/'+TIME+'/dpretrained')

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Reinforcement Training Generator...'

    for total_batch in range(TOTAL_BATCH):
        for it in range(TRAIN_ITER):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, cnn)
            feed = {generator.x: samples, generator.rewards: rewards}
            _, g_loss = sess.run([generator.g_updates, generator.g_loss], feed_dict=feed)

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            test_perp = np.exp(target_loss(sess, generator, likelihood_data_loader))
            perps.append(test_perp)

            generate_samples(sess, generator, BATCH_SIZE, generated_num, target_file)
            target_data_loader.create_batches(target_file)
            test_loss = target_loss(sess, target_lstm, target_data_loader)
            oraclelosses.append(test_loss)
            print 'total_batch: ', total_batch, 'test_perp: ', test_perp, 'test loss' test_loss
            
        rollout.update_params()

        # generate for discriminator
        print 'Start training discriminator'
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)

            dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_file, negative_file)
            dis_batches = dis_data_loader.batch_iter(zip(dis_x_train, dis_y_train), dis_batch_size, 3)

            batch_accs = []
            for batch in dis_batches:
                try:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, step, accuracy = sess.run([dis_train_op, dis_global_step, cnn.accuracy], feed)
                    batch_accs.append(accuracy)
                except ValueError:
                    pass
            accuracies.append(np.mean(batch_accs))
            print 'acc: ', np.mean(batch_accs)

        if total_batch % 5 == 0:
            with open('./data/'+TIME+'/perps.txt', 'w') as f:
                cPickle.dump(perps, f)
            with open('./data/'+TIME+'/losses.txt', 'w') as f:
                cPickle.dump(test_loss, f)
            with open('./data/'+TIME + '/pretrain_accuracies.txt', 'w') as f:
                cPickle.dump(accuracies, f)

if __name__ == '__main__':
    main()
