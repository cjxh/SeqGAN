import tensorflow as tf

class Generator(object):
    def __init__(self, sequence_length, hidden_size, vocab_size, batch_size):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.add_placeholders()
        self.build()
        
    def add_placeholders():
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.latch_num = tf.placeholder(tf.int32)

    def build(self):
        self.U = tf.get_variable('U_output', shape=[self..hidden_size, self.vocab_size], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = tf.get_variable('b2_output', tf.zeros(self.vocab_size))
        h_t_1 = tf.zeros((self.batch, self.hidden_size))
        x_t = tf.embedding_lookup(self.embeddings, self.start_token)
        self.cell = tf.contrib.rnn.BasicGRUCell(self.hidden_size)

        with tf.variable_scope("RNN"):
            for time_step in range(self.sequence_length):
                if time_step != 0:
                    tf.get_variable_scope().reuse_variables()

                o_t, h_t = cell(tf.embedding_lookup(self.embeddings, x_t), h_t_1)
                y_t = tf.matmul(o_t, U) + b2

                # update values for next time step
                if time_step < self.latch_num:
                    x_t = self.x.read(time_step)
                else:
                    x_t = tf.cast(tf.reshape(tf.multinomial(y_t, 1), [self.batch_size]), tf.int32)
                h_t_1 = h_t
    
    def add_training_op(self, loss):
        pass

    def generate_from_latch(self, sess, input_x, N):
        feed = {self.x: input_x, self.latch_num: N}
        outputs = sess.run([], feed)
        return outputs[0]
        
    def update_params(self):
        pass
