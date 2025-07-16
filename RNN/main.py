import torch 
import torch.nn as nn
import os


from data import ALL_LETTRS, NUM_LETTERS
from data import load_raw_data, word_to_tensor
from models import RNN

NUM_CATEGORIES = len(os.listdir("./data/data/"))
print(NUM_CATEGORIES)
#data, labels = load_raw_data()

