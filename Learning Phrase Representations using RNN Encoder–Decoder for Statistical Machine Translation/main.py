import sys
"""path = "/Users/yadyneshsonale/github/Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation/"
sys.path.append(path)  """
from models import RNNEncoder, RNNDecoder
from functions import tokenize, get_tokens, english_numericalize, french_numericalize, Pad_or_Trunc

import torch 
import pandas as pd
import numpy as np
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


data_path = "/home/cs23b1055/text/data.csv"

data = pd.read_csv(data_path)

english_sentences = data['English words/sentences']
english_sentences = list(english_sentences)

french_sentences = data['French words/sentences']
french_sentences = list(french_sentences)

english_words = get_tokens(english_sentences)
english_dict = Counter(english_words)

french_words = get_tokens(french_sentences)
french_dict = Counter(french_words)

specials = ['<unk>', '<pad>', '<sos>', '<eos>']

english_vocab = build_vocab_from_iterator([english_words], specials=specials)
french_vocab = build_vocab_from_iterator([french_words], specials=specials)
english_vocab.set_default_index(english_vocab['<unk>'])
french_vocab.set_default_index(french_vocab['<unk>'])

print([(token, english_vocab[token]) for token in tokenize("hello my black friend")])


X = []
y = []

n = int(len(data) / 100)
print(n)
for i in range(n):
    english = data['English words/sentences'][i]
    french = data['French words/sentences'][i]

    eng_num = english_numericalize(english_vocab, english)
    fre_num = french_numericalize(french_vocab, french)

    X.append(torch.tensor(eng_num))
    y.append(torch.tensor(fre_num))

eng_pad_or_trunc = Pad_or_Trunc(12, english_vocab)
fre_pad_or_trunc = Pad_or_Trunc(12, french_vocab)

X_final = torch.stack([eng_pad_or_trunc.pad_or_trunc(seq) for seq in X])
y_final = torch.stack([fre_pad_or_trunc.pad_or_trunc(seq) for seq in y])

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, train_size=0.8, random_state=42)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 64

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)



input_size = len(english_vocab)
output_size = len(french_vocab)

print(input_size, output_size)

num_epochs = 10
max_length = 12
embedding_dim = 500
hidden_size = 1000

encoder = RNNEncoder(input_size, embedding_dim, hidden_size).to(device)
decoder = RNNDecoder(output_size, embedding_dim, hidden_size).to(device)


criterion = nn.CrossEntropyLoss(ignore_index=french_vocab['<pad>'])
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)


for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    running_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for eng_batch, fre_batch in train_loader:
        eng_batch, fre_batch = eng_batch.to(device), fre_batch.to(device)
        optimizer.zero_grad()
        batch_loss = 0.0

        for eng, fre in zip(eng_batch, fre_batch):
            context = encoder(eng)
            outputs = decoder(fre, context)


            loss = criterion(outputs, fre)
            batch_loss += loss.item()

            preds = outputs.argmax(dim=1)
            mask = fre != french_vocab['<pad>']
            total_correct += (preds == fre).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

        batch_loss = torch.tensor(batch_loss, requires_grad=True)
        batch_loss.backward()
        optimizer.step()
        running_loss += batch_loss.item()

    accuracy = 100 * total_correct / total_tokens
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Train Accuracy: {accuracy:.2f}%")


"""

torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'english_vocab': english_vocab.get_stoi(), 
    'french_vocab': french_vocab.get_itos(),  
}, 'rnn_translation_model.pth')



def translate(sentence, encoder, decoder, english_vocab, french_vocab, max_length=12):
    encoder.eval()
    decoder.eval()

    # Tokenize and numericalize the input sentence
    tokens = tokenize(sentence.lower())
    token_ids = [english_vocab[token] if token in english_vocab else english_vocab['<unk>'] for token in tokens]
    input_tensor = torch.tensor(token_ids).to(device)

    # Encode the input sequence to get the context vector
    context = encoder(input_tensor)
    h_t = torch.tanh(decoder.V_prime(context))  # decoder initial hidden state

    # Start decoding from <sos>
    sos_idx = french_vocab['<sos>']
    token_idx = sos_idx

    output_tokens = []

    for _ in range(max_length):
        # Embed the current input token
        token_tensor = torch.tensor([token_idx]).to(device)
        x = decoder.embedding(token_tensor).squeeze(0)

        # Compute gates and hidden state
        z_t = torch.sigmoid(decoder.Wz(x) + decoder.Uz(h_t) + decoder.Cz(context))
        r_t = torch.sigmoid(decoder.Wr(x) + decoder.Ur(h_t) + decoder.Cr(context))
        h_tilde = torch.tanh(decoder.W(x) + decoder.U(r_t * h_t) + decoder.C(context))
        h_t = z_t * h_t + (1 - z_t) * h_tilde

        # Maxout output layer
        s_t_prime = decoder.Oh(h_t) + decoder.Oy(x) + decoder.Oc(context)
        s_t = torch.max(s_t_prime.view(-1, 2), dim=1).values

        # Final projection: vocab logits
        g_t = decoder.Gl(s_t)
        token_idx = g_t.argmax(dim=0).item()
        word = french_vocab.get_itos()[token_idx]

        if word in ['<eos>', '<pad>']:
            break

        output_tokens.append(word)

    return ' '.join(output_tokens)





# Rebuild vocab objects
checkpoint = torch.load('rnn_translation_model.pth', map_location=device)
english_vocab = {k: v for k, v in checkpoint['english_vocab'].items()}
french_vocab_list = checkpoint['french_vocab']
french_vocab = {i: tok for i, tok in enumerate(french_vocab_list)}
french_vocab_reverse = {tok: i for i, tok in enumerate(french_vocab_list)}

# Rebuild models
embedding_dim = 500
hidden_size = 1000

encoder = RNNEncoder(len(english_vocab), embedding_dim, hidden_size).to(device)
decoder = RNNDecoder(len(french_vocab), embedding_dim, hidden_size).to(device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Translate sentence
sentence = "hello my black friend"
output = translate(sentence, encoder, decoder, english_vocab, french_vocab)
print(f"English: {sentence}")
print(f"French: {output}")
"""