import tensorflow as tf
import numpy as np

class Discriminator(object):
    def __init__(self, sequence_length, batch_size, n_classes, pretrained_embeddings):
        print "Initializing discriminator..."
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.n_hidden = 1
        self.pretrained_embeddings = pretrained_embeddings

        self.add_placeholders()
        self.x, self.y = self.add_embeddings()
        self.real_preds = self.build(True)
        self.fake_preds = self.build(False)
        self.loss = self.add_loss_op()

    def add_placeholders(self):
        self.input_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.input_y = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def add_embedding(self):
        embeddings = tf.Variable(self.pretrained_embeddings)
        real_embeddings = tf.nn.embedding_lookup(embeddings, self.input_x)
        fake_embeddings = tf.nn.embedding_lookup(embeddings, self.input_y)
        return real_embeddings, fake_embeddings
    
    def build(self, isReal):
        # Define weights
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        self.weights = tf.Variable(tf.random_normal([2*self.n_hidden, self.n_classes]))
        self.biases = tf.Variable(tf.random_normal([self.n_classes]))
        self.theta = [self.weights, self.biases]
        
        self.cell_fw = tf.contrib.rnn.BasicGRUCell(self.hidden_size)
        self.cell_bw = tf.contrib.rnn.BasicGRUCell(self.hidden_size)

        if isReal:
            outputs, _, _ = tf.nn.bidirectional_rnn(self.cell_fw, self.cell_bw, self.x, dtype=tf.float32)
        else:
            outputs, _, _ = tf.nn.bidirectional_rnn(self.cell_fw, self.cell_bw, self.y, dtype=tf.float32)

        return tf.matmul(outputs[-1], self.weights) + self.biases

    def add_loss_op(self):
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_preds, tf.ones_like(self.real_preds)))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_preds, tf.zeros_like(self.fake_preds)))
        # add regularization?
        loss = real_loss + fake_loss
        return loss
