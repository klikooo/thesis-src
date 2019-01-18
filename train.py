import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from DataAscad import DataAscad
from optimizers.Nadam import Nadam
from util import HW

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train2(x_profiling, y_profiling, train_size, network, epochs=700, batch_size=1000, lr=0.00001, use_hw=True):
    # Cut to the correct training size
    x_profiling = x_profiling[0:train_size]
    y_profiling = y_profiling[0:train_size]

    # Convert values to hamming weight if asked for
    if use_hw:
        y_profiling = np.array([HW[val] for val in y_profiling])

    print(network)

    optimizer = torch.optim.RMSprop(network.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = Nadam(network.parameters(), lr=lr)
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
                torch.from_numpy(x_profiling[i * batch_size: (i + 1) * batch_size].astype(np.float32)),
                requires_grad=False).to(device)
            batch_y = Variable(torch.from_numpy(y_profiling[i * batch_size: (i + 1) * batch_size].astype(np.long)),
                               requires_grad=False).to(
                device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # TODO: klopt dit?
            net_out = network(batch_x)
            loss = criterion(net_out, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch {}, loss {}".format(epoch, running_loss / total_batches))
    return network


def train(x_profiling, y_profiling, train_size, network, epochs=700, batch_size=1000, lr=0.00001, use_hw=True):
    # Cut to the correct training size
    x_profiling = x_profiling[0:train_size]
    y_profiling = y_profiling[0:train_size]

    # Convert values to hamming weight if asked for
    if use_hw:
        y_profiling = np.array([HW[val] for val in y_profiling])

    train_data_set = DataAscad(x_profiling, y_profiling, train_size)

    print(network)

    # Optimizer
    # optimizer = torch.optim.RMSprop(network.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = Nadam(network.parameters(), lr=lr)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Perform training
    for epoch in range(epochs):
        # Load the data and shuffle it each epoch
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        train_iter = iter(train_loader)
        total_batches = int(train_size / batch_size)
        # total_batches = 1000
        # Loop over all batches
        running_loss = 0.0
        for i in range(total_batches):
            batch_x, batch_y = train_iter.next()

            # zero the parameter gradients
            optimizer.zero_grad()

            # TODO: klopt dit?
            net_out = network(batch_x)
            loss = criterion(net_out, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch {}, loss {}".format(epoch, running_loss / total_batches))
    return network

