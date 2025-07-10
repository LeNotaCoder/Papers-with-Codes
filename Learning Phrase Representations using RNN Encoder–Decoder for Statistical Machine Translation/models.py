import torch
import torch.nn as nn

class ResetGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResetGate, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, x, h_old):
        W_x = torch.matmul(self.W, x)
        U_h = torch.matmul(self.U, h_old)
        return torch.sigmoid(W_x + U_h)


class UpdateGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UpdateGate, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, x, h_old):
        W_x = torch.matmul(self.W, x)
        U_h = torch.matmul(self.U, h_old)
        return torch.sigmoid(W_x + U_h)


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.reset = ResetGate(input_size, hidden_size)
        self.update = UpdateGate(input_size, hidden_size)
        self.W = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, x, h_old):
        z_t = self.update(x, h_old)
        r_t = self.reset(x, h_old)

        r_h = r_t * h_old
        h_tilde = torch.tanh(torch.matmul(self.W, x) + torch.matmul(self.U, r_h))

        h_t = z_t * h_old + (1 - z_t) * h_tilde

        return h_t


class RNNDecoder(nn.Module):
    def __init__(self, c, input_size, hidden_size, output_size):
        super(RNNDecoder, self).__init__()
        self.reset = ResetGate(input_size, hidden_size)
        self.update = UpdateGate(input_size, hidden_size)
        self.W = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.W_o = nn.Parameter(torch.randn(output_size, hidden_size))
        self.b_o = nn.Parameter(torch.randn(output_size))

        self.c = c  # context vector 

    def hidden(self, h_old):
        c = self.c 

        z_t = self.update(c, h_old)
        r_t = self.reset(c, h_old)

        r_h = r_t * h_old
        h_tilde = torch.tanh(torch.matmul(self.W, c) + torch.matmul(self.U, r_h))

        h_t = z_t * h_old + (1 - z_t) * h_tilde

        return h_t

    def forward(self, h_t):

        o_t = torch.matmul(self.W_o, h_t) + self.b_o
        y_t = torch.softmax(o_t, dim=0)
        
        return y_t