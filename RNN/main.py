import torch 
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from torch.optim import SGD
import random
import matplotlib.pyplot as plt


from data import ALL_LETTRS, NUM_LETTERS, MyDataset
from data import load_raw_data, word_to_tensor
from models import RNN

NUM_CATEGORIES = len(os.listdir("./data/data/"))
NUM_HIDDEN = 128
NUM_EPOCHS = 1
BATCH_SIZE = 64

#data loading
data, labels = load_raw_data()

tensor_data = []
for word in data:
    tensor = word_to_tensor(word)
    tensor_data.append(tensor)

def labels_to_tensors(index):
    return torch.tensor(index, dtype=torch.long)


labels = [labels_to_tensors(label) for label in labels]

NUM_DATAPOINTS = len(labels)

X_train, X_test, y_train, y_test = train_test_split(tensor_data, labels, test_size=0.2, random_state=42)

train_dataset = MyDataset(X_train, y_train, BATCH_SIZE)
test_dataset = MyDataset(X_test, y_test, BATCH_SIZE)

#helper functions
def train(word_tensor, label_tensor):

    hidden = rnn.init_hidden()

    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)  
    
    loss = criterion(output[0], label_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

def get_index(output):
    return torch.argmax(output).item()

rnn = RNN(NUM_LETTERS, NUM_HIDDEN, NUM_CATEGORIES)

#training the RNN

criterion = nn.CrossEntropyLoss()
learning_rate = 0.005
optimizer = SGD(rnn.parameters(), lr=learning_rate)

current_loss = 0.0
correct = 0
all_losses = []

plot_steps = 1000
n_iters = 10000
train_size = int(0.8 * (NUM_DATAPOINTS - 1)) - 1
test_size = int(0.2 * (NUM_DATAPOINTS - 1)) - 1

for i in range(n_iters):
    index = random.randint(0, train_size)
    line_tensor, category_tensor = train_dataset.getitem(index)
    
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss 
    correct += 1 if get_index(output) == category_tensor else 0
    
    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        print(f"Num Epoch: {int(i / 999)}, Accuracy: {correct / plot_steps}, Loss: {all_losses[-1]}")
        current_loss = 0.0
        correct = 0

#checking validation accuracy
val_correct = 0
for i in range(test_size):
    line_tensor, category_tensor = train_dataset.getitem(i)
    
    output, loss = train(line_tensor, category_tensor)
    val_correct += 1 if get_index(output) == category_tensor else 0

print(f"Validation Accuracy: {val_correct / test_size}")

#Plotting the losses

x = range(1, NUM_EPOCHS + 1)
plt.plot(x, all_losses, marker='o', color='#4B0082', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Line Plot of Array')
plt.grid(True)
plt.show()

