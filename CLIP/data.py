import torch
import cv2
import numpy as np
import os
import pandas as pd
import json
from torch.utils.data import Dataset

from vid import processor, device

def video_frames(video_path):
  cap = cv2.VideoCapture(video_path)
  frames = []
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break

      frame = np.array(frame)
      frames.append(frame)

  cap.release()

  return frames


def context_vector_to_image(context_vector):
    context_vector = context_vector.clone()

    for k in range(3):
        channel = context_vector[0][k]
        min_val = torch.min(channel)
        max_val = torch.max(channel)

        for i in range(224):
            for j in range(224):
                context_vector[0][k][i][j] = ((channel[i][j] - min_val) / (max_val - min_val)) * 255

    return context_vector.byte()

class MyDataset(Dataset):
    def __init__(self, vids, labels, path):
        self.path = path
        self.vids = vids
        self.labels = labels

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        question_and_answer_parirs = self.labels[vid[:-4]]['question']
        true_label = self.labels[vid[:-4]]['answer_id']
        raw_frames = video_frames(os.path.join(self.path, vid))

        if len(raw_frames) == 0:
            return None

        inputs = processor(images=raw_frames, texts=question_and_answer_parirs, return_tensors="pt", padding=True)['pixel_values']
        inputs = inputs.to(device)

        T = inputs.shape[0]

        return inputs, T, true_label


with open('/home/cs23b1055/all_train.json', 'r') as f:
    all_train = json.load(f)

labels = pd.DataFrame(all_train)
final_labels = dict()
for video_id in labels.keys():
   for i in range(len(labels[video_id][5])):
         if labels[video_id][5][i]['id'] is not None:
            context = labels[video_id][5][i]['question']
            options = labels[video_id][5][i]['options']

            final_question = [context + ", " + str(opt) for opt in options]

            final_labels['video_id'] = {'question': final_question, 'answer_id': labels[video_id][5][i]['answer_id']}







vids = []
correct = 0
total = 0

path = '/home/cs23b1055/videos'

for dirs, paths, files in os.walk(path):
  for file in files:
    vids.append(file)


dataset = MyDataset(vids, final_labels, path)
