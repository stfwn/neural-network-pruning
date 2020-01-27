import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 100)
Y = 100 * 0.8**(X//5)

#plt.xkcd()
fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(10,7))
ax.plot(Y)
ax.set_ylabel('Percentage of Weights Remaining')
ax.set_xlabel('Epoch')
plt.savefig('pruning-progression.png')
