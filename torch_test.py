import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ascad import load_ascad, HW, SBOX_INV, SBOX, test_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, input_shape=700, out_shape=256):
        super(Net, self).__init__()
        n_hidden = 200
        self.fc1 = nn.Linear(input_shape, n_hidden).to(device)
        self.fc2 = nn.Linear(n_hidden, n_hidden).to(device)
        self.fc3 = nn.Linear(n_hidden, n_hidden).to(device)
        self.fc4 = nn.Linear(n_hidden, n_hidden).to(device)
        self.fc5 = nn.Linear(n_hidden, n_hidden).to(device)
        self.fc6 = nn.Linear(n_hidden, out_shape).to(device)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x)).to(device)
        x = F.relu(self.fc2(x)).to(device)
        x = F.relu(self.fc3(x)).to(device)
        x = F.relu(self.fc4(x)).to(device)
        x = F.relu(self.fc5(x)).to(device)

        x = self.fc6(x).to(device)
        # return F.softmax(x, dim=1).to(device)
        return x


def seq(in_, num_classes):
    n_hidden = 200
    return nn.Sequential(
        nn.Linear(in_, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, num_classes),
    ).to(device)


traces_file = '/media/rico/Data/TU/thesis/data/ASCAD.h5'
(x_profiling, y_profiling), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(traces_file,
                                                                                                     load_metadata=True)
use_hw = True
n_classes = 9 if use_hw else 256

if use_hw:
    y_profiling = np.array([HW[val] for val in y_profiling])
    y_attack = np.array([HW[val] for val in y_attack])

# net = seq(700, n_classes)
net = Net(out_shape=n_classes)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss().to(device)

epochs = 700
batch_size = 1000

for epoch in range(epochs):
    total_batches = int(50000 / batch_size)
    # Loop over all batches
    running_loss = 0.0
    for i in range(total_batches):
        # TODO: doing this total_batches times is rather useless, it is better to do it before training
        batch_x = Variable(torch.from_numpy(x_profiling[i * batch_size: (i + 1) * batch_size].astype(np.float32))).to(
            device)
        batch_y = Variable(torch.from_numpy(y_profiling[i * batch_size: (i + 1) * batch_size].astype(np.long))).to(
            device)
        # batch_x, batch_y = (Variable(x_profiling[i * batch_size: (i + 1) * batch_size]),
        #                     Variable(y_profiling[i * batch_size: (i + 1) * batch_size]))

        optimizer.zero_grad()
        net_out = net(batch_x)

        # TODO: klopt dit?
        loss = criterion(net_out, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 50 == 0:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

    print("Epoch {}, loss {}".format(epoch, running_loss/total_batches))


def accuracy(predictions, y_test):
    _, pred = predictions.max(1)
    # print(pred)
    # for i in range(20):
    #     print('i:: {} Prediction: {} Real {}, C: {}'.format(i,
    #                                                         pred[i],
    #                                                         y_test[i],
    #                                                         'JA' if pred[i] == y_test[i] else ''))
    z = pred == torch.from_numpy(y_test).to(device)
    num_correct = z.sum().item()
    print('Correct: {}'.format(num_correct))
    print('Accuracy: {}'.format(num_correct/10000.0))


with torch.no_grad():
    data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
    print(data.cpu().size())
    predictions = F.softmax(net(data).to(device), dim=-1).to(device)
    d = predictions[0].cpu().numpy()
    print(np.sum(d))

    accuracy(predictions, y_attack)

    test_model(predictions.cpu().numpy(), metadata_attack, 2, use_hw=use_hw, title='Torch {}'.format(
        'HW' if use_hw else 'ID'))
