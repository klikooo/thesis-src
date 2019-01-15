import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from ascad import HW

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(x_profiling, y_profiling, train_size, network, epochs=700, batch_size=1000, lr=0.00001, use_hw=True):
    # Cut to the correct training size
    x_profiling = x_profiling[0:train_size]
    y_profiling = y_profiling[0:train_size]

    # Convert values to hamming weight if asked for
    if use_hw:
        y_profiling = np.array([HW[val] for val in y_profiling])

    print(network)

    optimizer = torch.optim.RMSprop(network.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # Perform training
    for epoch in range(epochs):
        total_batches = int(train_size / batch_size)
        # total_batches = 1000
        # Loop over all batches
        running_loss = 0.0
        for i in range(total_batches):
            # TODO: doing this total_batches times is rather useless, it is better to do it before training
            batch_x = Variable(
                torch.from_numpy(x_profiling[i * batch_size: (i + 1) * batch_size].astype(np.float32))).to(
                device)
            batch_y = Variable(torch.from_numpy(y_profiling[i * batch_size: (i + 1) * batch_size].astype(np.long))).to(
                device)
            optimizer.zero_grad()
            net_out = network(batch_x)

            # TODO: klopt dit?
            loss = criterion(net_out, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch {}, loss {}".format(epoch, running_loss / total_batches))
    return network

