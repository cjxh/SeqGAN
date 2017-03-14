import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tqdm import tqdm

class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token, m, pretrained_embeddings, 
                 learning_rate=0.01, reward_gamma=0.95):
        self.hidden_dim = 10
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.learning_rate = learning_rate
        self.m = m
        self.baseline = 0

        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        #self.g_embeddings = pretrained_embeddings #tf.get_variable('g_embeddings', initializer=self.init_matrix([self.num_emb, self.emb_dim]))
        self.g_embeddings = pretrained_embeddings
        
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

        self.add_placeholders()

        # build the basic graphs
        self.preprocess_x()
        self.gen_x = self.build_latch_rnn()
        self.g_predictions = self.build_pretrain()
        self.build_xij()

        # pretraining functions
        self.add_pretrain_loss()
        self.pretrain_op = self.add_train_op(self.pretrain_loss)

        # maligan functions
        self.add_train_loss()
        self.train_op = self.add_train_op(self.train_loss)
        
    ###### Client functions ###################################################################
    def pretrain_one_step(self, sess, input_x):
        #### Call this to pretrain #######
        feed = {self.x: input_x}
        _, loss = sess.run([self.pretrain_op, self.pretrain_loss], feed_dict=feed)
        return loss

    def pretrain_one_epoch(self, sess, data_loader):
        supervised_g_losses = []
        data_loader.reset_pointer()

        for it in tqdm(xrange(data_loader.num_batch)):
            batch = data_loader.next_batch()
            g_loss = self.pretrain_one_step(sess, batch)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)

    def generate_from_latch(self, sess, input_x, N):
        feed = {self.x: input_x, self.given_num: N}
        outputs = sess.run([self.gen_x], feed)
        return outputs[0] # batch_size x seqlen

    def generate(self, sess):
        dummy = np.zeros((self.batch_size, self.sequence_length))
        feed = {self.x: dummy, self.given_num: 0}
        outputs = sess.run([self.gen_x], feed)
        return outputs[0] # batch size x seqlen

    def generate_xij(self, sess, input_x, N):
        feed = {self.x: input_x, self.given_num: N}
        outputs = sess.run([self.xij_calc, self.predsijs_calc], feed)
        return outputs[0], outputs[1] # xij, predsijs

    def train_one_step(self, sess, dis, xij, predsijs, N):
        rewards = self.RD(dis.get_predictions(sess, xij))
        rewards = np.reshape(rewards, (self.batch_size, self.m))
        denom = np.sum(rewards, axis=1)
        denom = denom.reshape((np.shape(denom)[0], 1))
        norm_rewards = np.divide(rewards, denom) #- self.baseline
        rewards = np.reshape(norm_rewards, (self.batch_size * self.m))
        feed = {self.xij: xij, self.predsijs: predsijs, self.rewards: rewards, self.given_num: N}
        return sess.run([self.train_op], feed)

    ############################################################################################


    ########## graph building ########################################
    def RD(self, reward):
        return reward / (1-reward)

    def add_train_loss(self):
        contrib = tf.reduce_sum(tf.one_hot(tf.to_int32(self.xij), self.num_emb, 1.0, 0.0) * \
            tf.log(tf.clip_by_value(self.predsijs, 1e-20, 1.0)), 2)
        masked = tf.slice(contrib, [0, self.given_num], [-1, -1])
        self.train_loss = tf.reduce_sum(tf.reduce_sum(masked, 1) * self.rewards)

    def build_xij(self):
        xij = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=self.m, dynamic_size=False, infer_shape=True, clear_after_read=False)
        predsijs = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.m, dynamic_size=False, infer_shape=True, clear_after_read=False)

        def body_(i, xij, predsijs):
            xij = xij.write(i, self.build_latch_rnn())
            predsijs = predsijs.write(i, self.build_pretrain())
            return i + 1, xij, predsijs

        _, xij, predsijs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2: i < self.m,
            body=body_, 
            loop_vars=(tf.constant(0, dtype=tf.int32), xij, predsijs))

        xij = xij.stack() # m x batch_size x seq len
        xij = tf.transpose(xij, perm=[1, 0, 2]) # batch_size x m x seqlen
        self.xij_calc = tf.reshape(xij, [self.batch_size * self.m, self.sequence_length])

        predsijs = predsijs.stack()
        predsijs = tf.transpose(predsijs, perm=[1, 0, 2, 3])
        self.predsijs_calc = tf.reshape(predsijs, [self.batch_size * self.m, self.sequence_length, self.num_emb])

    def add_placeholders(self):
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)
        self.rewards = tf.placeholder(tf.float32, shape=[self.m * self.batch_size])
        self.xij = tf.placeholder(tf.int32, shape=[self.batch_size * self.m, self.sequence_length])
        self.predsijs = tf.placeholder(tf.float32, shape=[self.batch_size * self.m, self.sequence_length, self.num_emb])

    def add_train_op(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads_and_vars = zip(*grads_and_vars)
        gradients = grads_and_vars[0]
        variables = grads_and_vars[1]
        gradients, global_norm = tf.clip_by_global_norm(gradients, 5.0)
        return optimizer.apply_gradients(zip(gradients, variables))

    def add_pretrain_loss(self):
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

    def preprocess_x(self):
        with tf.device("/cpu:0"):
            print self.sequence_length
            inputs = tf.split(tf.nn.embedding_lookup(self.g_embeddings, self.x), self.sequence_length, 1)
            self.processed_x = tf.stack(
                [tf.squeeze(input_, [1]) for input_ in inputs])  # seq_length x batch_size x emb_dim

        ta_emb_x = tensor_array_ops.TensorArray( 
            dtype=tf.float32, size=self.sequence_length, clear_after_read=False)
        self.ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, clear_after_read=False)
        self.ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))

    def build_pretrain(self):
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
            x_tp1 = self.ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))

        return tf.transpose(
            g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

    def build_latch_rnn(self):
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True,
                                             clear_after_read=False)

        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = self.ta_emb_x.read(i)
            gen_x = gen_x.write(i, self.ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x

        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, gen_x))

        gen_x = gen_x.stack()  # seq_length x batch_size
        return tf.transpose(gen_x, perm=[1, 0])  # batch_size x seq_length

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    ####################################################################################



    ############### Defining the RNN cell #############################################
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
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

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

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        self.Wo = tf.get_variable('Wo', shape=[self.hidden_dim, self.num_emb], \
            initializer=tf.contrib.layers.xavier_initializer())
        self.bo = tf.get_variable('bo', initializer=self.init_matrix([self.num_emb]))

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit
    ###############################################################################################
