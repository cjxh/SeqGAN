import tensorflow as tf
import numpy as np
from discriminator import Discriminator as dis

N = 10
batch_size = 3
n_classes = 2
k = 5
DROPOUT_KEEP_PROB = 0.75
pretrained_embeddings = np.random.rand(10, 5)
with tf.variable_scope('discriminator'):
    dis = dis(N, batch_size, n_classes, pretrained_embeddings)
dis_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
dis_optimizer = tf.train.AdamOptimizer(1e-4)
dis_grads_and_vars = dis_optimizer.compute_gradients(dis.loss, dis_params)
dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(k):
    real_minibatches = np.random.randint(0, high=9, size=(batch_size, N))
    gen_minibatches = np.random.randint(0, high=9, size=(batch_size, N))
    dis_x_train = np.concatenate((real_minibatches, gen_minibatches), axis=0)
    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))
    dis_y_train = np.concatenate((real, fake), axis=0)

    shuffle_idx = np.random.permutation(np.arange(2 * batch_size))
    shuffled_x =  dis_x_train[shuffle_idx]
    shuffled_y =  dis_y_train[shuffle_idx]
    
    feed = {
        dis.input_x: shuffled_x,
        dis.input_y: shuffled_y,
        dis.dropout_keep_prob: DROPOUT_KEEP_PROB
    }
    _ = sess.run([dis_train_op], feed)
