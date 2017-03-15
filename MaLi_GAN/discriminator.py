import tensorflow as tf
import numpy as np

class Discriminator(object):
    def __init__(self, sequence_length, batch_size, n_classes, pretrained_embeddings):
        print "Initializing discriminator..."
        self.batch_size = 2 * batch_size
        print 'batch_size: ' + str(self.batch_size)
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.n_hidden = 10
        self.g_embeddings = pretrained_embeddings

        self.add_placeholders()
        self.x = self.add_embedding()
        self.preds = self.build()
        self.loss = self.add_loss_op()

    ### client functions ###
    def get_predictions(self, sess, x):
        feed = {self.input_x: x}
        return sess.run(self.outputs, feed)
        
    def train_one_step(self, sess, data_loader, x_real, x_fake):
        real_minibatches = data_loader.next_batch()
        dis_x_train = np.concatenate((x_real, x_fake), axis=0)

        y_real = [[0, 1] for _ in real_examples]
        y_fake = [[1, 0] for _ in real_examples]
        dis_y_train = np.concatenate((y_real, y_fake), axis=0)

        shuffle_idx = np.random.permutation(np.arange(2 * batch_size))
        shuffled_x =  dis_x_train[shuffle_idx]
        shuffled_y =  dis_y_train[shuffle_idx]
    
        feed = {
            dis.input_x: shuffled_x,
            dis.input_y: shuffled_y,
            dis.dropout_keep_prob: DROPOUT_KEEP_PROB
        }
        _, loss = sess.run([dis_train_op, dis.loss], feed)
        return loss
    ########################

    def add_placeholders(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_length])
        self.input_y = tf.placeholder(tf.float32, shape=[None, 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def add_embedding(self):
        embeddings = tf.nn.embedding_lookup(self.g_embeddings, self.input_x)
        return tf.cast(embeddings, tf.float32)
    
    def build(self):
        # Define weights
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        self.weights = tf.get_variable('weights', initializer=tf.random_normal([2*self.n_hidden, self.n_classes]))
        self.biases = tf.get_variable('biases', initializer=tf.random_normal([self.n_classes]))
        
        self.cell_fw = tf.contrib.rnn.GRUCell(self.n_hidden)
        self.cell_bw = tf.contrib.rnn.GRUCell(self.n_hidden)

        with tf.variable_scope('preds'):
            seqlen = tf.fill(tf.expand_dims(tf.shape(self.input_x)[0], 0), tf.constant(self.sequence_length, dtype=tf.int32))
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.x, dtype=tf.float32, sequence_length=seqlen)
            print self.sequence_length * np.ones(self.batch_size)
            output_state = tf.concat(output_states, 1)
        
        preds = tf.matmul(output_state, self.weights) + self.biases
        self.outputs = tf.slice(tf.nn.softmax(preds), [0,1],[-1,-1])
        return preds

    def add_loss_op(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.preds))
        # add regularization?
        return loss
