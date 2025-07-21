import torch
import torch.nn as nn
import torch.nn.functional as F

class ResetGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResetGate, self).__init__()

        self.linear_x = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_old):
        return torch.sigmoid(self.linear_x(x) + self.linear_h(h_old))


class UpdateGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UpdateGate, self).__init__()

        self.linear_x = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_old):
        return torch.sigmoid(self.linear_x(x) + self.linear_h(h_old))


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNNEncoder, self).__init__()

        self.W = nn.Linear(500, )