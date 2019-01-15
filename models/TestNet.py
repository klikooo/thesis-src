import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TestNet(nn.Module):
    def __init__(self, pr_shape, sbox_shape, n_classes=9):
        super(TestNet, self).__init__()
        self.pr_shape = pr_shape
        self.sbox_shape = sbox_shape
        # self.tensor_min = Variable(torch.from_numpy(np_max), requires_grad=False).to(device)
        # self.tensor_max = Variable(torch.from_numpy(np_min), requires_grad=False).to(device)

        pr_n_hidden = 250
        pr_out_shape = n_classes
        self.pr_fc1 = nn.Linear(pr_shape, pr_n_hidden).to(device)
        self.pr_fc2 = nn.Linear(pr_n_hidden, pr_n_hidden).to(device)
        self.pr_fc_end = nn.Linear(pr_n_hidden, pr_out_shape).to(device)

        sbox_n_hidden = 250
        sbox_out_shape = n_classes
        self.sbox_fc1 = nn.Linear(sbox_out_shape + pr_out_shape, sbox_n_hidden).to(device)
        self.sbox_fc2 = nn.Linear(sbox_n_hidden, sbox_n_hidden).to(device)
        self.sbox_fc_end = nn.Linear(sbox_n_hidden, sbox_out_shape).to(device)

        self.sbox_r_fc1 = nn.Linear(sbox_shape, sbox_n_hidden).to(device)
        self.sbox_r_fc2 = nn.Linear(sbox_n_hidden, sbox_n_hidden).to(device)
        self.sbox_r_end = nn.Linear(sbox_n_hidden, sbox_out_shape).to(device)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.pr_fc1.weight)
        torch.nn.init.xavier_uniform_(self.pr_fc2.weight)
        torch.nn.init.xavier_uniform_(self.pr_fc_end.weight)
        torch.nn.init.xavier_uniform_(self.sbox_fc1.weight)
        torch.nn.init.xavier_uniform_(self.sbox_fc2.weight)
        torch.nn.init.xavier_uniform_(self.sbox_fc_end.weight)
        torch.nn.init.xavier_uniform_(self.sbox_r_fc1.weight)
        torch.nn.init.xavier_uniform_(self.sbox_r_fc2.weight)
        torch.nn.init.xavier_uniform_(self.sbox_r_end.weight)

    def name(self):
        return "TestNet"

    def forward(self, x):
        # x = F.relu(self.fc1(x)).to(device)
        pr = x[:, :self.pr_shape]
        sbox_r = x[:, self.pr_shape:self.pr_shape+self.sbox_shape]

        # print('pr shape var: {}'.format(self.pr_shape))
        # print('size x {}'.format(x.size()))
        # print(pr.size())
        # print('sbox size {}'.format(sbox.size()))
        pr = F.relu(self.pr_fc1(pr).to(device))
        pr = F.relu(self.pr_fc2(pr).to(device))
        pr = F.relu(self.pr_fc_end(pr).to(device))

        sbox_r = F.relu(self.sbox_r_fc1(sbox_r).to(device))
        sbox_r = F.relu(self.sbox_r_fc2(sbox_r).to(device))
        sbox_r_end = F.relu(self.sbox_r_end(sbox_r).to(device))

        sbox_pr = torch.cat((pr, sbox_r_end), 1)
        sbox_pr = F.relu(self.sbox_fc1(sbox_pr).to(device))
        sbox_pr = F.relu(self.sbox_fc2(sbox_pr).to(device))
        sbox_pr = self.sbox_fc_end(sbox_pr).to(device)

        return sbox_pr

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape
        }, path)

    # @staticmethod
    # def load_spread(file):
        # checkpoint = torch.load(file)

        # model = SpreadNet(checkpoint['sf'], input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model.tensor_max = checkpoint['max']
        # model.tensor_min = checkpoint['min']
        # return model
