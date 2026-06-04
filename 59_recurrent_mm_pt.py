# %% SETUP

# Import the libraries

import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

# Seed

torch.manual_seed(1)


# %% DATA

# Download the dataset

data_dir = Path('datasets/gutenberg')
data_file = data_dir / 'gutenberg.txt'
download_url = 'https://www.gutenberg.org/files/1268/1268-0.txt'

data_dir.mkdir(parents=True, exist_ok=True)

if not data_file.exists():

    urlretrieve(download_url, data_file)

# Import the dataset

with open(data_file, 'r', encoding='utf8') as fp:

    text = fp.read()

# Remove portfions from beginning and end

start_idx = text.find('THE MYSTERIOUS ISLAND')
end_idx = text.find('End of the Project Gutenberg')
text = text[start_idx:end_idx]
print('Total Length:', len(text))


# %% TOKENS

# Extract the set of unique characters

char_set = set(text)

print('Unique Characters:', len(char_set))

# Dictionary to map characters to integers and array to map integers to characters

char_set_sorted = sorted(char_set)
char2int = {ch:i for i, ch in enumerate(char_set_sorted)}
char_array = np.array(char_set_sorted)

text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
print(text[:15], '==> Encoding ==>', text_encoded[:15])
print(text_encoded[:15], '==> Reverse ==>',''.join(char_array[text_encoded[:15]]))

# Divide the encoded text into chunks of 41 characters

seq_length = 40
chunk_size = seq_length + 1
text_chunks = [text_encoded[i:i+chunk_size] for i in range(len(text_encoded)-chunk_size+1)]

# Define the sequences of (input, target)

class TextDataset(Dataset):

    def __init__(self, text_chunks):

        self.text_chunks = text_chunks

    # Tells the DataLoader how many training samples exist

    def __len__(self):

        return len(self.text_chunks)

    # Returns one (input, target) pair

    def __getitem__(self, idx):

        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()

seq_dataset = TextDataset(torch.tensor(text_chunks)) 

# Visualise some example sequences

for i, (seq, target) in enumerate(seq_dataset):

    print(' Input (x): ',
          repr(''.join(char_array[seq])))
    print('Target (y): ',
          repr(''.join(char_array[target])))
    print()

    if i == 1:

        break

# Create a dataset

batch_size=64
train_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# %% MODEL

# Design the recurrent neural nework

class RNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
    
        super().__init__()
        
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=None)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

        self.rnn_hidden_size = rnn_hidden_size

    def forward(self, x, hidden, cell):

        x = self.emb(x).unsqueeze(1)
        x, (hidden, cell) = self.rnn(x, (hidden, cell))
        x = self.fc(x).reshape(x.size(0), -1)
        
        return x, hidden, cell

    def init_hidden(self, batch_size):
        
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)

        return hidden, cell

# Initialize a recurrent neural network object

vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512

model = RNN(vocab_size, embed_dim, rnn_hidden_size)

# Loss function

loss_fun = nn.CrossEntropyLoss()


# %% TRAINING

# Parameters

num_epochs = 10000

# Learn from the data

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(num_epochs):

    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(train_dl))
    loss = 0
    
    for c in range(seq_length):
    
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fun(pred, target_batch[:, c])
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    avg_loss = loss.item() / seq_length
    
    if epoch % 500 == 0:
    
        print(f'Epoch {epoch} loss: {avg_loss:.4f}') 


# %% TESTING

# Function that receives a string and generates a new one

def sample(model, starting_str, len_generated_text=500, scale_factor=1.0):

    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))
    generated_str = starting_str

    model.eval()
    hidden, cell = model.init_hidden(1)

    for c in range(len(starting_str)-1):

        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)
    
    last_char = encoded_input[:, -1]
    
    for i in range(len_generated_text):

        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])
        
    return generated_str

# Generate some new text

print(sample(model, starting_str='The island')) 
