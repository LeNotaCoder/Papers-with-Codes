import jax
import numpy as np
from videoprism import models as vp

import cv2
import numpy as np
import os

def video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(np.array(frame))
    cap.release()
    return frames

vids = []
path = '/home/cs23b1055/videos'
for _, _, files in os.walk(path):
    for file in files:
        if file[:-4] in final_labels.keys():
            if file.endswith('.mp4'):
                vids.append(file)

# Choose model name
model_name = "videoprism_public_v1_large"  # VideoPrism-L (ViT‑L variant)
flax_model = vp.get_model(model_name)
state = vp.load_pretrained_weights(model_name)

# Prepare dummy input: shape (batch, time, height, width, 3)
dummy = video_frames(os.path.join(path, vids[0]))
dummy = np.array(dummy, dtype=np.float32)
dummy = np.expand_dims(dummy, axis=0)  # Add batch dimension
dummy = np.transpose(dummy, (0, 4, 1, 2, 3))  # Change to (batch, channels, time, height, width)

# Run inference
video_embeds, _ = flax_model.apply(state, dummy, train=False)
print("Output shape:", video_embeds.shape)
print("Output video embedding:", video_embeds)