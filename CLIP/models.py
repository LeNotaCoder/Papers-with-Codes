import torch
import torch.nn as nn

from data import context_vector_to_image
from config import processor, clip_model  # ✅ import from config

class StepwiseFrameLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_shape):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.output_shape = output_shape
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, frames, T):
        device = frames.device
        T, C, H, W = frames.shape
        frames = frames.view(T, -1) 

        h_t = torch.zeros(1, self.lstm_cell.hidden_size, device=device)
        c_t = torch.zeros(1, self.lstm_cell.hidden_size, device=device)

        for t in range(T):
            x_t = frames[t].unsqueeze(0).to(device)
            h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))

        context = self.fc(h_t)
        context = context.view(1, *self.output_shape)
        return context

class FinalModel(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    def forward(self, frames, T, final):
        frames = frames.squeeze(0)  # remove batch dim -> [T, 3, 224, 224]
        context_vector = self.lstm(frames, T)
        image = context_vector_to_image(context_vector)
        inputs = processor(text=final, images=image, return_tensors="pt", padding=True).to(frames.device)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs
