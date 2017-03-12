from generator import Generator 
from lstm import LSTM
import tensorflow as tf 
import numpy as np

with tf.variable_scope('generator'):
	generator = Generator(5, 6, 4, 3, 7, 0)

# generator = LSTM(4, 3, 7, 6, 5, 0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

data = np.ones((3, 5)) * 2

result = generator.generate_from_latch(sess, data, 2)

print result

# self.U = tf.get_variable('U_output', shape=[self.hidden_size, self.vocab_size], \
#             initializer=tf.contrib.layers.xavier_initializer())
#         self.b2 = tf.get_variable('b2_output', initializer=tf.zeros(self.vocab_size))
#         h_t_1 = tf.zeros((self.batch_size, self.hidden_size))
#         x_t = [self.start_token] * self.batch_size
#         self.cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
#         self.gen_x = tensor_array_ops.TensorArray(self.sequence_length)

#         for time_step in range(self.sequence_length):
#             if time_step != 0:
#                 tf.get_variable_scope().reuse_variables()

#             o_t, h_t = self.cell(tf.nn.embedding_lookup(self.embeddings, x_t), h_t_1)
#             y_t = tf.nn.softmax(tf.matmul(o_t, self.U) + self.b2)

#             # update values for next time step
#             next_token = tf.cond(time_step < self.latch_num, lambda: self.x[:, time_step:time_step+1], \
#                 lambda: tf.cast(tf.multinomial(y_t, 1), tf.int32))

#             print tf.squeeze(next_token).get_shape()
#             self.gen_x.write(time_step, next_token)
#             x_t = tf.embedding_lookup(self.nn.embeddings, next_token)
#             h_t_1 = h_t