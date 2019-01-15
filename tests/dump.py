import pickle
import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
z = np.array([x, y])

with open('test.txt', 'wb') as fp:
    pickle.dump(z, fp)

with open('test.txt', 'rb') as fp:
    print(pickle.load(fp))
