import tensorflow as tf
import numpy as np
import data_loader as dl
from tqdm import tqdm
# import generator
# import discriminator
# import evaluation

# initialize constants
T = 35
N = T
K = 5

# populate the lexicon of existing words
lexicon = {}
counter = 0
with open('word_lexicon.txt', 'r') as f:
    print "Pre-processing and saving word lexicon in memory..."
    for line in tqdm(f):
        for word in line:
            lexicon[word] = counter
            counter += 1

# load real data
positive_file = 'save/real_data.txt'
data_loader = dl(lexicon, N)
print "Loading data from " + positive_file + " into memory..."
positive_data = data_loader.load_data(positive_file)

# initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

sess = tf.Session()
sess.run(tf.global_variable_initializer())

# pretrain 
# generator.pretrain()
# discriminator.pretrain()

while N >= 0:
    N = N - K
    for i in range(k):
        # minibatch of real training data
        real_minibatches = data_loader.mini_batch()
        # minibatch, get first N from real_minibatch, generate the rest
        gen_minibatch = generator.generate_from_latch(sess, real_minibatches, N)
        discriminator.update_params()
    
    # minibatch of real training data
    new_minibatch = real_data_loader.mini_batch(positive_file)
    x_ij = generator.generate_from_latch(new_minibatch, N)
    generator.update_params()
        
