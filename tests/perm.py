import numpy as np


batch_size = 10
n_features = 100
kernel_size = 5
r = np.random.rand(batch_size, 1, n_features)

# Create a random permutation for each kernel size
p = []
for i in range(int(n_features/kernel_size)):
    p1 = np.random.permutation(kernel_size) + kernel_size * i
    p = np.concatenate((p, p1))

print(r[:, :, p.astype(int)])
