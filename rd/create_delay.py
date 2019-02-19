import numpy as np


def create_delay(a, b, n):
    delays = np.zeros(n)
    m = np.random.randint(0, a-b)
    div = int(n/2)
    for i in range(div):
        delays[i] = m + np.random.randint(0, b)
    for i in range(div):
        index = i+div
        delays[index] = a - m - np.random.randint(0, b)
    return delays, m


n = 100
d, m = create_delay(18, 3, 10)
half = d[0:int(n/2)]
print('Mean: {} guess mean:{}, guess std {}'.format(m, np.mean(half) - int(3/2), np.std(half)))


all = np.arange(0, 255)

print(all)
y = np.random.randint(0, 255)
# y_value = np.sum(random_feature)
# print(random_feature)
print(y)

print('')




