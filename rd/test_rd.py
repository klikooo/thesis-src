from models.ConvNetKernel import ConvNetKernel
from models.DenseNet import DenseNet
from test import accuracy
from train import train
from util import load_csv
import numpy as np


train_size = 2000
attack_size = 3000
x = load_csv('testX.csv', size=train_size)
y = load_csv('testY.csv', size=train_size)
print(np.shape(x))
print(y)

num_features = 1250
out_features = 10

network = ConvNetKernel(num_features, out_features)
train(x, y, train_size, network,
      epochs=80,
      batch_size=100,
      lr=0.0001)


x_test = load_csv('testX.csv', start=train_size, size=attack_size)
y_test = load_csv('testY.csv', start=train_size, size=attack_size, dtype=np.long)
accuracy(network, x_test, y_test)