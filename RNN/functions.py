import torch
from main import rnn, criterion, optimizer

def train(word_tensor, label_tensor):

    hidden = rnn.init_hidden()

    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)
    
    loss = criterion(output, label_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

def get_index(output):
    return torch.argmax(output)