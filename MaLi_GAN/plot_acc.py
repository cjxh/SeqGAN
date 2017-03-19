import matplotlib.pyplot as plt  
import cPickle 
import numpy as np

# acc = cPickle.load(open('./pretrain_perplexities_eval.txt'))
# acc2 = cPickle.load(open('./20170317-234848/eval_perps.txt'))
# acc = np.concatenate((acc, acc2))

acc = cPickle.load(open('./20170318-011742/eval_perps.txt'))

plt.plot(range(len(acc)), acc)

plt.show()
