import tensorflow as tf
import numpy as np
# import generator
# import discriminator
# import data_preprocess
# import evaluation
 
generator = generator()
discriminator = discriminator()

data = load_data()
T = len(data[0])

N = T
K = 1

# pretrain 
# generator.pretrain()
# discriminator.pretrain()

while N >= 0:
    N = N - K
    for i in range(k):
        real_minibatch = # minibatch of real training data
        gen_minibatch = # minibatch, get first N from real_minibatch, generate the rest
        discriminator.update_params()
    
    new_minibatch = # minibatch of real training data
    x_ij = generator.generate(new_minibatch, N)
    generator.update_params()
        
