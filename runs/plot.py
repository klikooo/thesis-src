import matplotlib.pyplot as plt
import pickle
import numpy as np

title = 'Spread network'
with open('x_{}.r'.format(title), 'rb') as f:
    ranks_x = pickle.load(f)
with open('y_{}.r'.format(title), 'rb') as f:
    ranks_y = pickle.load(f)

runs = len(ranks_x)
print(runs)

plt.title('Performance of {}'.format(title))
plt.xlabel('number of traces')
plt.ylabel('rank')
plt.grid(True)
for i in range(runs):
    plt.plot(ranks_x[i], ranks_y[i])
plt.figure()

# Mean figure
rank_avg_y = np.mean(ranks_y, axis=0)

plt.title('Performance of {}'.format(title))
plt.xlabel('number of traces')
plt.ylabel('rank')
plt.grid(True)
plt.plot(ranks_x[0], rank_avg_y, label='mean')
plt.legend()
plt.show()
plt.figure()
