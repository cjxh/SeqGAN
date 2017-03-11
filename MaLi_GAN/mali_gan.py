import tensorflow as tf
import numpy as np
import data_loader as dl
# import generator
# import discriminator
# import evaluation

# populate the lexicon of existing words
lexicon = {}
counter = 0
with open(lexicon_file, 'r') as f:
    for line in f:
        for word in line:
            lexicon[word] = counter
            counter += 1

# load real data
positive_file = 'save/real_data.txt'
data_loader = dl(lexicon)
positive_data = data_loader.load_data(positive_file)

# initialize constants
T = data_loader.get_max_length()
N = T
K = 1

# initialize generator and discriminator
generator = generator()
discriminator = discriminator()

# pretrain 
# generator.pretrain()
# discriminator.pretrain()

while N >= 0:
    N = N - K
    for i in range(k):
        # minibatch of real training data
        real_minibatches = data_loader.mini_batch()
        # minibatch, get first N from real_minibatch, generate the rest
        gen_minibatch = generator.generate(real_minibatches, N)
        discriminator.update_params()
    
    # minibatch of real training data
    new_minibatch = real_data_loader.mini_batch(positive_file)
    x_ij = generator.generate(new_minibatch, N)
    generator.update_params()
        
