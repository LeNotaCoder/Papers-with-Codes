import torch
import string
import os
import pandas as pd

ALL_LETTRS = string.ascii_letters + ".,;'"
NUM_LETTERS = len(ALL_LETTRS)

def letter_index(letter):
    return ALL_LETTRS.find(letter)

def word_to_tensor(word):
    word_size = len(word)
    one_hot = torch.zeros(word_size, 1, NUM_LETTERS)
    for i in range(word_size):
        one_hot[i][0][letter_index(word[i])] = 1
    
    return one_hot


def load_raw_data():
    path = "./data/data/"
    dirs = os.listdir(path)

    dataset = []
    labels = []

    for i, dir in enumerate(dirs):
        data = pd.read_csv(path + dir, header=None, on_bad_lines='skip')
        dataset += list(data[0])
        labels += [i] * len(data)

    return dataset, labels

class MyDataset():
    def __init__(self, X, y, batch_size):

        self.data = X
        self.labels = y
        self.batch_size = batch_size
    
    def getitem(self, index):
        return self.data[index], self.labels[index]

    



