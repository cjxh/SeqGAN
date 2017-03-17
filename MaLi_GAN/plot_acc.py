import matplotlib.pyplot as plt  
import cPickle 
import numpy as np

acc = cPickle.load(open('./pretrain_losses.txt'))

plt.plot(range(len(acc)), acc)

plt.show()