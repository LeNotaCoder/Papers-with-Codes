import re
from torch.nn.functional import pad

def tokenize(sentence):
    return re.findall(r"\w+|[^\w\s]", sentence.lower())


def get_tokens(sentences):
    words = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        words += tokens
    
    return words

def english_numericalize(vocab, sentence):
  tokens = re.findall(r"\w+|[^\w\s]", sentence.lower())
  return [vocab[token] for token in tokens]

def french_numericalize(vocab, sentence):
  tokens = re.findall(r"\w+|[^\w\s]", sentence.lower())
  return [vocab[token] for token in tokens]


class Pad_or_Trunc():
    def __init__(self, max, vocab):
        self.max_length = max
        self.pad_value = vocab['<pad?>']
        self.vocab_size = len(vocab)
        self.embedding_dim = 50
    
    def pad_or_trunc(self, sequence):
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            return pad(sequence, (0, self.max_length - len(sequence)), value=self.pad_value)