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
real_data_loader = dl()
gen_data_loader = dl()

# initialize constants
T = len(data[0])
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
        real_minibatch = real_data_loader.mini_batch(positive_file)
        # minibatch, get first N from real_minibatch, generate the rest
        gen_minibatch = gen_data_loader.mini_batch(negative_file) 
        discriminator.update_params()
    
    new_minibatch = # minibatch of real training data
    x_ij = generator.generate(new_minibatch, N)
    generator.update_params()
        
