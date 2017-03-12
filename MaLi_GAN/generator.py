import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Generator(object):
    def __init__(self, sequence_length, hidden_size, vocab_size, batch_size, emb_size, start_token):
        self.hidden_dim = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_emb = vocab_size
        self.emb_dim = emb_size
        # self.add_placeholders()
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.g_embeddings = tf.get_variable('embedding', initializer=tf.random_normal([self.num_emb, self.emb_dim], stddev=.1))
        
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

        #####################################################################################################
        # placeholder definition
        self.add_placeholders()

        # processed for batch
        self.build_rnn()

        
    def add_placeholders(self):
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)

    def build_rnn(self):
        with tf.device("/cpu:0"):
            inputs = tf.split(1, self.sequence_length, tf.nn.embedding_lookup(self.g_embeddings, self.x))
            self.processed_x = tf.pack(
                [tf.squeeze(input_, [1]) for input_ in inputs])  # seq_length x batch_size x emb_dim

        ta_emb_x = tensor_array_ops.TensorArray( 
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unpack(self.processed_x)

        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unpack(tf.transpose(self.x, perm=[1, 0]))
        #####################################################################################################

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.pack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x

        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        self.gen_x = self.gen_x.pack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

    def add_training_op(self, loss):
        pass

    def generate_from_latch(self, sess, input_x, N):
        feed = {self.x: input_x, self.given_num: N}
        outputs = sess.run([self.gen_x], feed)
        return outputs[0]

    def generate(self, sess):
        feed = {self.x: None, self.latch_num: 0}
        outputs = sess.run([self.gex_x], feed)
        return outputs[0]
        
    def update_params(self):
        pass

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.get_variable('Wi', shape=[self.emb_dim, self.hidden_dim], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.Ui = tf.get_variable('Ui', shape=[self.hidden_dim, self.hidden_dim], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.bi = tf.get_variable('bi', shape=[self.hidden_dim])

        self.Wf = tf.get_variable('Wf', shape=[self.emb_dim, self.hidden_dim], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.Uf = tf.get_variable('Uf', shape=[self.hidden_dim, self.hidden_dim], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.bf = tf.get_variable('bf', shape=[self.hidden_dim])

        self.Wog = tf.get_variable('Wog', shape=[self.emb_dim, self.hidden_dim], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.Uog = tf.get_variable('Uog', shape=[self.hidden_dim, self.hidden_dim], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.bog = tf.get_variable('bog', shape=[self.hidden_dim])

        self.Wc = tf.get_variable('Wc', shape=[self.emb_dim, self.hidden_dim], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.Uc = tf.get_variable('Uc', shape=[self.hidden_dim, self.hidden_dim], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.bc = tf.get_variable('bc', shape=[self.hidden_dim])


        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unpack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.pack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        self.Wo = tf.get_variable('Wo', shape=[self.hidden_dim, self.num_emb], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.bo = tf.get_variable('bo', initializer=self.init_matrix([self.num_emb]))

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unpack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit
