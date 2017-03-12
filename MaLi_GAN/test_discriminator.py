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
dis_global_step = tf.Variable(0, name="global_step", trainable=False)
dis_optimizer = tf.train.AdamOptimizer(1e-4)
dis_grads_and_vars = dis_optimizer.compute_gradients(dis.loss, dis_params)
dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(k):
    # minibatches of real training data ... do they mean 1 or all minibatches??
    real_minibatches = np.random.randint(0, high=9, size=(5 * batch_size, N))
    # minibatch, get first N from real_minibatch, generate the rest
    gen_minibatches = np.random.randint(0, high=9, size=(5 * batch_size, N))
    dis_batches = zip(real_minibatches, gen_minibatches)
    for i in range(0, 5):
        batch = dis_batches[i*batch_size:i*batch_size + batch_size]
        x_batch, y_batch = zip(*batch)
        feed = {
            dis.input_x: x_batch,
            dis.input_y: y_batch,
            dis.dropout_keep_prob: DROPOUT_KEEP_PROB
        }
        _, step = sess.run([dis_train_op, dis_global_step], feed)
