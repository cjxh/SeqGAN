import matplotlib.pyplot as plt  
import cPickle 
import numpy as np

acc2 = cPickle.load(open('./accuracies.txt'))
acc3 = cPickle.load(open('./accuracies3.txt'))
print acc2, acc3
acc = np.concatenate((acc2, acc3))

plt.plot(range(len(acc)), acc)

plt.show()