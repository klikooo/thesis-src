import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from DataLoaders.DataAscad import DataAscad
from DataLoaders.DataDK import DataDK
from util import HW, device, save_model


def train(x_profiling, y_profiling, train_size, network, epochs=80, batch_size=1000, lr=0.00001,
          checkpoints=None, save_path=None):
    # Cut to the correct training size
    x_profiling = x_profiling[0:train_size]
    y_profiling = y_profiling[0:train_size]

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

        # Save checkpoints
        if checkpoints is not None and epoch in checkpoints:
            save_model(network, '{}.{}.pt'.format(save_path, epoch))

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

            # Calculate the batch and do a backward pass
            net_out = network(batch_x)
            loss = criterion(net_out, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch {}, loss {}".format(epoch, running_loss / total_batches))
    return network


def train_dk(x_profiling, y_profiling, plain, train_size, network, epochs=700, batch_size=1000, lr=0.00001,
            checkpoints=None, save_path=None):
    # Cut to the correct training size
    x_profiling = x_profiling[0:train_size]
    y_profiling = y_profiling[0:train_size]
    plain = plain[0:train_size]

    train_data_set = DataDK(x_profiling, y_profiling, plain, train_size)

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

        # Save checkpoints
        if checkpoints is not None and epoch in checkpoints:
            save_model(network, '{}.{}.pt'.format(save_path, epoch))

        # Load the data and shuffle it each epoch
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        train_iter = iter(train_loader)
        total_batches = int(train_size / batch_size)
        # total_batches = 1000
        # Loop over all batches
        running_loss = 0.0
        for i in range(total_batches):
            batch_x, batch_y, plains = train_iter.next()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Calculate the batch and do a backward pass
            net_out = network(batch_x, plains)
            loss = criterion(net_out, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch {}, loss {}".format(epoch, running_loss / total_batches))
    return network

