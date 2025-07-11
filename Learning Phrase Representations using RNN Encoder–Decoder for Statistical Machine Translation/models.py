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

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.reset = ResetGate(embedding_dim, hidden_size)
        self.update = UpdateGate(embedding_dim, hidden_size)
        self.linear_x = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input_seq):

        h_t = torch.zeros(self.linear_h.out_features).to(input_seq.device)
        for token_idx in input_seq:
            x = self.embedding(token_idx)
            z_t = self.update(x, h_t)
            r_t = self.reset(x, h_t)
            r_h = r_t * h_t
            h_tilde = torch.tanh(self.linear_x(x) + self.linear_h(r_h))
            h_t = z_t * h_t + (1 - z_t) * h_tilde
        return h_t  # context vector



class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNNDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.reset = ResetGate(embedding_dim, hidden_size)
        self.update = UpdateGate(embedding_dim, hidden_size)
        self.linear_x = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_idx, h_old):
        
        x = self.embedding(token_idx)
        z_t = self.update(x, h_old)
        r_t = self.reset(x, h_old)
        r_h = r_t * h_old
        h_tilde = torch.tanh(self.linear_x(x) + self.linear_h(r_h))
        h_t = z_t * h_old + (1 - z_t) * h_tilde
        output = self.output_layer(h_t)
        return output, h_t
