

positive_file = 'save/real_data.txt'

target_params = cPickle.load(open('save/target_params.pkl'))

target_lst = TARGET_LSTM(5000, 64, 32, 32, 20, 0, target_params)
sess = tf.Session()

sess.run(tf.global_variables_initialization())


