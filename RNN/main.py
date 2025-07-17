import torch 
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from optim import SGD

from data import ALL_LETTRS, NUM_LETTERS
from data import load_raw_data, word_to_tensor, MyDataset
from models import RNN

NUM_CATEGORIES = len(os.listdir("./data/data/"))
NUM_HIDDEN = 128
NUM_EPOCHS = 2
BATCH_SIZE = 64

data, labels = load_raw_data()

tensor_data = []
for word in data:
    tensor = word_to_tensor(word)
    tensor_data.append(tensor)

X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=0.2, random_state=42)

train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

train_loader = Dataloader(train_dataset, batch_size=BATCH_SIZE)
test_loader = Dataloader(test_dataset, batch_size=BATCH_SIZE)

rnn = RNN(NUM_LETTERS, NUM_HIDDEN, NUM_CATEGORIES)

#one step

input_tensor = word_to_tensor('J')[0]
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = SGD(rnn.parameters(), lr=learning_rate)

def train(word_tensor, label_tensor):

    hidden = rnn.init_hidden()

    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)
    
    loss = criterion(output, label_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0.0
all_losses = []

for epoch in range(NUM_EPOCHS):





