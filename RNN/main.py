import torch 
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from torch.optim import SGD


from data import ALL_LETTRS, NUM_LETTERS
from data import load_raw_data, word_to_tensor
from models import RNN

NUM_CATEGORIES = len(os.listdir("./data/data/"))
NUM_HIDDEN = 128
NUM_EPOCHS = 2000
BATCH_SIZE = 64

data, labels = load_raw_data()
print("hello")

tensor_data = []
for word in data:
    tensor = word_to_tensor(word)
    tensor_data.append(tensor)

def labels_to_tensors(index):
    tensor = torch.zeros(1, NUM_CATEGORIES, dtype=torch.long)
    tensor[0][index] = 1
    return tensor


labels = [labels_to_tensors(label) for label in labels]

X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=0.2, random_state=42)

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

for epoch in range(NUM_EPOCHS):
    for k in range(len(y_train)):
        output, loss = train(X_train[k], y_train[k])
        
        current_loss += loss
        correct += 1 if get_index(output) == get_index(y_train[k]) else 0
    
    all_losses.append(current_loss / BATCH_SIZE)
    print(f"Accuracy: {correct / BATCH_SIZE}, Loss: {current_loss / BATCH_SIZE}")
    correct = 0
    current_loss = 0.0

