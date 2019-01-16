import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import HW
from ascad import load_ascad, test_model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DenseSpreadNet(nn.Module):
    def __init__(self, spread_factor, input_shape, out_shape):
        super(DenseSpreadNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 100 * spread_factor).to(device)
        self.fc2 = nn.Linear(100 * spread_factor, 100 * spread_factor).to(device)
        self.spread_factor = spread_factor
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.fc3 = nn.Linear(100 * spread_factor, out_shape).to(device)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x)).to(device)
        x = F.relu(self.fc2(x)).to(device)

        x = self.fc3(x).to(device)
        # return F.softmax(x, dim=1).to(device)
        return x

    def name(self):
        return "DenseSpreadNet"

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'sf': self.spread_factor,
            'out_shape': self.out_shape,
            'input_shape': self.input_shape
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = DenseSpreadNet(checkpoint['sf'], input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def accuracy(predictions, y_test):
    _, pred = predictions.max(1)
    z = pred == torch.from_numpy(y_test).to(device)
    num_correct = z.sum().item()
    print('Correct: {}'.format(num_correct))
    print('Accuracy: {}'.format(num_correct/10000.0))


if __name__ == "__main__":
    traces_file = '/media/rico/Data/TU/thesis/data/ASCAD.h5'
    (x_profiling, y_profiling), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(traces_file,
                                                                                                         load_metadata=True)
    use_hw = True
    n_classes = 9 if use_hw else 256

    if use_hw:
        y_profiling = np.array([HW[val] for val in y_profiling])
        y_attack = np.array([HW[val] for val in y_attack])

    # net = seq(700, n_classes)
    net = DenseSpreadNet(out_shape=n_classes)
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
            batch_x = Variable(
                torch.from_numpy(x_profiling[i * batch_size: (i + 1) * batch_size].astype(np.float32))).to(
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

        print("Epoch {}, loss {}".format(epoch, running_loss / total_batches))

    with torch.no_grad():
        data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
        print(data.cpu().size())
        predictions = F.softmax(net(data).to(device), dim=-1).to(device)
        d = predictions[0].cpu().numpy()
        print(np.sum(d))

        accuracy(predictions, y_attack)

        test_model(predictions.cpu().numpy(), metadata_attack, 2, use_hw=use_hw, title='Torch {}'.format(
            'HW' if use_hw else 'ID'))
