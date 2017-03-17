import matplotlib.pyplot as plt  
import cPickle 
import numpy as np

acc = cPickle.load(open('./pretrain_perplexities_eval.txt'))
acc2 = cPickle.load(open('./eval_perps.txt'))
acc = np.concatenate((acc, acc2))

plt.plot(range(len(acc)), acc)

plt.show()
