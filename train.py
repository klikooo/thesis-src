import torch
import math
from torch.utils.data import DataLoader

from DataLoaders.DataAscad import DataAscad
from DataLoaders.DataDK import DataDK
from util import HW, device, save_model
import util_optimizer
import util_scheduler


def train(x_profiling, y_profiling, train_size,
          x_validation, y_validation, validation_size,
          network, loss_function, epochs=80, batch_size=1000, lr=0.00001,
          checkpoints=None, save_path=None, l2_penalty=0.0, optimizer="Adam",
          scheduler=None, scheduler_args=None):
    # Cut to the correct training size

    train_data_set = DataAscad(x_profiling, y_profiling, train_size)
    validation_data_set = DataAscad(x_validation, y_validation, validation_size)

    print(network)

    # Optimizer
    # optimizer = torch.optim.RMSprop(network.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = Nadam(network.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=l2_penalty)
    optimizer_args = {
        "lr": lr,
        "l2": l2_penalty
    }
    print(f"Using optimizer {optimizer}")
    optimizer_func = util_optimizer.get_optimizer(optimizer)
    optimizer = optimizer_func(network.parameters(), optimizer_args)

    # Loss function
    # criterion = nn.CrossEntropyLoss().to(device)
    loss_function = loss_function.to(device)

    # Losses and accuracy for saving
    vali_losses = []
    train_losses = []
    vali_acc = []
    train_acc = []

    schedule_func = lambda: None
    # Scheduler
    if scheduler is not None:
        scheduler = util_scheduler.get_scheduluer(scheduler)(optimizer, scheduler_args)
        schedule_func = lambda: scheduler.step()


    # Perform training
    for epoch in range(epochs):

        # Save checkpoints
        if checkpoints is not None and epoch in checkpoints:
            save_model(network, '{}.{}.pt'.format(save_path, epoch))

        # Load the data and shuffle it each epoch
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        train_iter = iter(train_loader)
        total_batches = int(train_size / batch_size)

        # Loop over all batches
        train_running_loss = 0.0
        train_correct = 0
        for i in range(total_batches):
            batch_x, batch_y = train_iter.next()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Calculate the batch and do a backward pass
            net_out = network(batch_x)
            loss = loss_function(net_out, batch_y)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

            _, pred = net_out.max(1)
            z = pred == batch_y
            train_correct += z.sum().item()

            schedule_func()

        # Do a check on the validation
        validation_iter = iter(DataLoader(validation_data_set, batch_size=batch_size))
        validation_loss = 0.0
        validation_batches = int(validation_size / batch_size)
        validation_correct = 0
        with torch.no_grad():
            for i in range(validation_batches):
                batch_x, batch_y = validation_iter.next()

                net_out = network(batch_x)
                loss = loss_function(net_out, batch_y)
                validation_loss += loss.item()

                _, pred = net_out.max(1)
                z = pred == batch_y
                validation_correct += z.sum().item()


        train_loss = train_running_loss / total_batches
        vali_loss = validation_loss / validation_batches
        print("Epoch {}, train loss {}, train acc {}%, validation loss {}, vali acc {}%".format(
            epoch,
            train_loss, train_correct / train_size * 100.0,
            vali_loss, validation_correct / validation_size * 100.0))

        # Append the results of the epoch
        vali_losses.append(vali_loss)
        train_losses.append(train_loss)
        vali_acc.append(validation_correct / validation_size)
        train_acc.append(train_correct / train_size)
    return network, (train_losses, vali_losses, train_acc, vali_acc)


def train_dk2(x_profiling, y_profiling, p_profiling, train_size,
              x_validation, y_validation, p_validation, validation_size,
              network, loss_function, epochs=80, batch_size=1000, lr=0.00001,
              checkpoints=None, save_path=None, l2_penalty=0.0):
    # Cut to the correct training size
    x_profiling = x_profiling[0:train_size]
    y_profiling = y_profiling[0:train_size]
    p_profiling = p_profiling[0:train_size]

    train_data_set = DataDK(x_profiling, y_profiling, p_profiling, train_size)
    validation_data_set = DataDK(x_validation, y_validation, p_validation, validation_size)

    print(network)

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=l2_penalty)

    # Loss function
    # criterion = nn.CrossEntropyLoss().to(device)
    loss_function = loss_function.to(device)

    # Losses and accuracy for saving
    vali_losses = []
    train_losses = []
    vali_acc = []
    train_acc = []

    # Perform training
    for epoch in range(epochs):

        # Save checkpoints
        if checkpoints is not None and epoch in checkpoints:
            save_model(network, '{}.{}.pt'.format(save_path, epoch))

        # Load the data and shuffle it each epoch
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        train_iter = iter(train_loader)
        total_batches = int(train_size / batch_size)

        # Loop over all batches
        train_running_loss = 0.0
        train_correct = 0
        for i in range(total_batches):
            batch_x, batch_y, plaintexts = train_iter.next()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Calculate the batch and do a backward pass
            net_out = network(batch_x, plaintexts)
            loss = loss_function(net_out, batch_y)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

            _, pred = net_out.max(1)
            z = pred == batch_y
            train_correct += z.sum().item()

        # Do a check on the validation
        validation_iter = iter(DataLoader(validation_data_set, batch_size=batch_size))
        validation_loss = 0.0
        validation_batches = int(validation_size / batch_size)
        validation_correct = 0
        with torch.no_grad():
            for i in range(validation_batches):
                batch_x, batch_y, plaintexts = validation_iter.next()

                net_out = network(batch_x, plaintexts)
                loss = loss_function(net_out, batch_y)
                validation_loss += loss.item()

                _, pred = net_out.max(1)
                z = pred == batch_y
                validation_correct += z.sum().item()

        train_loss = train_running_loss / total_batches
        vali_loss = validation_loss / validation_batches
        print("Epoch {}, train loss {}, train acc {}%, validation loss {}, vali acc {}%".format(
            epoch,
            train_loss, train_correct / train_size * 100.0,
            vali_loss, validation_correct / validation_size * 100.0))

        # Append the results of the epoch
        vali_losses.append(vali_loss)
        train_losses.append(train_loss)
        vali_acc.append(validation_correct / validation_size)
        train_acc.append(train_correct / train_size)
    return network, (train_losses, vali_losses, train_acc, vali_acc)
