import torch 
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from torch.optim import SGD
import random


from data import ALL_LETTRS, NUM_LETTERS, MyDataset
from data import load_raw_data, word_to_tensor
from models import RNN

NUM_CATEGORIES = len(os.listdir("./data/data/"))
NUM_HIDDEN = 128
NUM_EPOCHS = 1
BATCH_SIZE = 64

data, labels = load_raw_data()

tensor_data = []
for word in data:
    tensor = word_to_tensor(word)
    tensor_data.append(tensor)

def labels_to_tensors(index):
    tensor = torch.zeros(1, NUM_CATEGORIES, dtype=torch.long)
    tensor[0][index] = 1
    return tensor


labels = [labels_to_tensors(label) for label in labels]

NUM_DATAPOINTS = len(labels)

X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=0.2, random_state=42)

train_dataset = MyDataset(X_train, y_train, BATCH_SIZE)
test_dataset = MyDataset(X_test, y_test, BATCH_SIZE)

rnn = RNN(NUM_LETTERS, NUM_HIDDEN, NUM_CATEGORIES)

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = SGD(rnn.parameters(), lr=learning_rate)

current_loss = 0.0
correct = 0
all_losses = []

def train(word_tensor, label_tensor):

    hidden = rnn.init_hidden()

    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)  
    
    loss = criterion(output[0], label_tensor[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

def get_index(output):
    return torch.argmax(output).item()

plot_steps = 1000
n_iters = 100000
train_size = int(0.8 * (NUM_DATAPOINTS - 1)) - 1

for i in range(n_iters):
    index = random.randint(0, train_size)
    line_tensor, category_tensor = train_dataset.getitem(index)
    
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss 
    
    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0


