import jax
import jax.numpy as jnp
from videoprism import models as vp
import numpy as np

import torch
import os
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
import torch.nn.functional as F


from models import IntermidiateModel
from functions import read_video_frames


path = "/home/cs23b1055/videos2"

model_name = 'videoprism_public_v1_base'
flax_model = vp.get_model(model_name)
state = vp.load_pretrained_weights(model_name)

@jax.jit
def forward_fn(inputs):
    return flax_model.apply(state, inputs, train=False)





clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import json
import pandas as pd

records = []
with open('/home/cs23b1055/all_train.json', 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

labels = pd.DataFrame(records)


correct = 0
total = 0

for _, _, files in os.walk(path):
    
    for file in files:

        video_inputs = read_video_frames(path + f"/{file}")
        if video_inputs is None:
            print("Frames loaded:", path)
        

        video_np = video_inputs.permute(0,2,3,1).numpy() 
        video_np = np.expand_dims(video_np, axis=0)  
        video_jax = jnp.array(video_np)
        
        outputs, _ = forward_fn(video_jax) 
        print("Encoder output shape:", outputs.shape)
        
        outputs_torch = torch.tensor(np.array(outputs))
        
        model = ToImage()
        image = model(outputs_torch)
        
        image = show_tensor_as_image(image)
        
        vid = file[:-4]
        print(vid)
        for i in range(len(labels[vid][0]['mc_question'])):
            question = labels[vid][0]['mc_question'][i]['question']
            options = labels[vid][0]['mc_question'][i]['options']
            label = labels[vid][0]['mc_question'][i]['answer_id']
            text = []
            for option in options:
                text.append(question + ", " + option)
            
            inputs = clip_processor(images=image, text=text, return_tensors="pt", padding=True)
    
            output = clip_model(**inputs)
            logits_per_image = output.logits_per_image
            probs = F.softmax(logits_per_image, dim=-1)
            if (probs.argmax(dim=1).item() == label):
                correct += 1
            
            total += 1


total *= 1 

print(correct / total)
