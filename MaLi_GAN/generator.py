import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tqdm import tqdm

class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token, m, pretrained_embeddings, 
                 learning_rate=0.01, reward_gamma=0.95):
        self.hidden_dim = 200
        self.sequence_length = sequence_length
        #self.batch_size = batch_size
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.learning_rate = learning_rate
        self.m = m
        self.baseline = 0

        self.g_embeddings = tf.cast(pretrained_embeddings, tf.float32)
        
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

        self.add_placeholders()
        self.batch_size = tf.shape(self.x)[0]
        self.start_token = tf.fill(tf.expand_dims(self.batch_size, 0), tf.constant(start_token, dtype=tf.int32))

        # build the basic graphs
        self.preprocess_x()
        self.gen_x = self.build_latch_rnn()
        self.g_predictions = self.build_pretrain()
        #self.build_xij()

        # pretraining functions
        self.pretrain_loss = self.add_pretrain_loss()
        self.pretrain_op = self.add_train_op(self.pretrain_loss, .1)

        # maligan functions
        self.train_loss = self.add_train_loss()
        self.train_op = self.add_train_op(self.train_loss, .001)
        
    ###### Client functions ###################################################################
    def pretrain_one_step(self, sess, input_x, input_mask):
        #### Call this to pretrain #######
        feed = {self.x: input_x, self.mask : input_mask}
        _, loss = sess.run([self.pretrain_op, self.pretrain_loss], feed_dict=feed)
        return loss

    def pretrain_one_epoch(self, sess, data_loader):
        supervised_g_losses = []
        data_loader.reset_pointer()

        for it in xrange(data_loader.num_batch):
            batch, mask_batch = data_loader.next_batch()
            g_loss = self.pretrain_one_step(sess, batch, mask_batch)
            supervised_g_losses.append(g_loss)

        loss = np.mean(supervised_g_losses)
        return np.exp(loss)

    def get_perplexity(self, sess, data_loader):
        supervised_g_losses = []
        data_loader.reset_pointer()

        for it in xrange(data_loader.num_batch):
            batch, mask_batch = data_loader.next_batch()
            feed = {self.x: batch, self.mask : mask_batch}
            g_loss = sess.run(self.pretrain_loss, feed_dict=feed)
            supervised_g_losses.append(g_loss)

        loss = np.mean(supervised_g_losses)
        return np.exp(loss)

    def generate_from_latch(self, sess, input_x, N):
        feed = {self.x: input_x, self.given_num: N}
        outputs = sess.run([self.gen_x], feed)
        return outputs[0] # batch_size x seqlen

    def generate(self, sess, num_to_gen):
        dummy = np.zeros((num_to_gen, self.sequence_length))
        feed = {self.x: dummy, self.given_num: 0}
        outputs = sess.run([self.gen_x], feed)
        return outputs[0] # batch size x seqlen

    def generate_xij(self, sess, input_x, N):
        input_x = np.repeat(input_x, self.m, axis=0)
        feed = {self.x: input_x, self.given_num: N}
        outputs = sess.run([self.gen_x], feed)
        return outputs[0]

    def train_one_step(self, sess, dis, xij):
        rewards = self.RD(dis.get_predictions(sess, xij))
        #rewards = np.reshape(rewards, (-1, self.m))
        denom = np.sum(rewards)
        #print np.mean(rewards)
        #denom = denom.reshape((np.shape(denom)[0], 1))
        baseline = 1.0/(rewards.shape[0])
        rewards = np.divide(rewards, denom) - baseline
        #rewards = np.reshape(norm_rewards, (-1))
        feed = {self.x: xij, self.rewards: rewards}
        outputs = sess.run([self.train_op, self.train_loss, self.partial], feed)
        return outputs[1], outputs[2]

    ############################################################################################


    ########## graph building ########################################
    def RD(self, reward):
        return reward / (1-reward)

    def add_train_loss(self):
        contrib = tf.reduce_sum(tf.one_hot(tf.to_int32(self.x), self.num_emb, 1.0, 0.0) * \
            tf.log(tf.clip_by_value(self.g_predictions, 1e-20, 1.0)), 2)
        contrib = contrib * self.rewards
        self.partial = contrib
        return -tf.reduce_sum(contrib)

        # masked = tf.slice(contrib, [0, self.given_num], [-1, -1])
        # self.train_loss = -tf.reduce_sum(tf.reduce_sum(masked, 1) * self.rewards)

    def add_placeholders(self):
        self.x = tf.placeholder(tf.int32, shape=[None, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)
        self.mask = tf.placeholder(tf.bool, shape=[None, self.sequence_length])
        self.rewards = tf.placeholder(tf.float32, shape=[None, 1])

    def add_train_op(self, loss, lr):
        optimizer = tf.train.AdamOptimizer(lr)
        return optimizer.minimize(loss)
        '''grads_and_vars = optimizer.compute_gradients(loss)
        grads_and_vars = zip(*grads_and_vars)
        gradients = grads_and_vars[0]
        variables = grads_and_vars[1]
        gradients, global_norm = tf.clip_by_global_norm(gradients, 5.0)
        return optimizer.apply_gradients(zip(gradients, variables))'''

    def add_pretrain_loss(self):
        print self.g_predictions
        return -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(tf.boolean_mask(tensor=self.x, mask=self.mask), [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(tf.boolean_mask(tensor=self.g_predictions, mask=self.mask), [-1, self.num_emb]), 1e-20, 1.0)
            )
        ) / tf.reduce_sum(tf.cast(self.mask, tf.float32))#(self.sequence_length * tf.to_float(self.batch_size))

    def preprocess_x(self):
        with tf.device("/cpu:0"):
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
