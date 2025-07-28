import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader

from models import StepwiseFrameLSTM, FinalModel 
from data import dataset
from config import device

loader = DataLoader(dataset, batch_size=1, shuffle=True)

lstm = StepwiseFrameLSTM(3 * 224 * 224, 1024, (3, 224, 224)).to(device)
final_model = FinalModel(lstm).to(device)

optimizer = optim.Adam(final_model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 5
loss_history, acc_history = [], []

for epoch in range(num_epochs):
    total_loss, correct, total = 0.0, 0, 0

    i = 0

    for batch in loader:
        print(i)
        i += 1
        if batch is None:
            continue

        inputs, T, qa_pairs, true_label = batch
        result = [item[0] for item in qa_pairs]
        inputs, true_label = inputs.to(device), true_label.to(device)

        optimizer.zero_grad()
        outputs = final_model(inputs, T, result)
        loss = loss_fn(outputs, true_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == true_label).sum().item()
        total += true_label.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    loss_history.append(avg_loss)
    acc_history.append(accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), loss_history, label='Loss')
plt.plot(range(1, num_epochs+1), acc_history, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.savefig('training_metrics.png')
plt.close()
