import torch
from transformers import CLIPModel, CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
