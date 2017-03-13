import tensorflow as tf
import numpy as np

class Discriminator(object):
    def __init__(self, sequence_length, batch_size, n_classes, pretrained_embeddings):
        print "Initializing discriminator..."
        self.batch_size = 2 *batch_size
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.n_hidden = 10
        self.g_embeddings = pretrained_embeddings

        self.add_placeholders()
        self.x = self.add_embedding()
        self.preds = self.build()
        self.loss = self.add_loss_op()

    def add_placeholders(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_length])
        self.input_y = tf.placeholder(tf.float32, shape=[None, 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def add_embedding(self):
        embeddings = tf.Variable(self.g_embeddings)
        embeddings = tf.nn.embedding_lookup(embeddings, self.input_x)
        return tf.cast(embeddings, tf.float32)
    
    def build(self):
        # Define weights
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        self.weights = tf.Variable(tf.random_normal([2*self.n_hidden, self.n_classes]))
        self.biases = tf.Variable(tf.random_normal([self.n_classes]))
        
        self.cell_fw = tf.contrib.rnn.GRUCell(self.n_hidden)
        self.cell_bw = tf.contrib.rnn.GRUCell(self.n_hidden)

        with tf.variable_scope('preds'):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.x, dtype=tf.float32, sequence_length=(self.sequence_length * np.ones(self.batch_size)))
            output_state = tf.concat(output_states, 1)
        
        preds = tf.matmul(output_state, self.weights) + self.biases
	self.outputs = tf.slice(tf.nn.softmax(preds), [0,1],[-1,-1])
        return preds

    def get_predictions(self, sess, x_ij):
	feed = {self.input_x: x_ij}
	return sess.run(self.outputs, feed)

    def add_loss_op(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.preds))
        # add regularization?
        return loss
