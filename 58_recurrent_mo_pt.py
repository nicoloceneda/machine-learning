# %% SETUP

# Import the libraries

import re
import warnings
from collections import Counter, OrderedDict

import torchtext
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

torchtext.disable_torchtext_deprecation_warning()
warnings.filterwarnings('ignore', category=UserWarning, module=r'torchdata\.datapipes')

from torchtext.datasets import IMDB
from torchtext.vocab import vocab
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

# Seed

torch.manual_seed(1)


# %% DATA

# Import the dataset

df_train = IMDB(split='train')
df_test = IMDB(split='test')

# Separate the data into train, validation and test subsets

df_train, df_valid = random_split(list(df_train), [20_000, 5_000])
df_test = list(df_test)


# %% TOKENS

# Function to clean the documents and apply the tokenizer

def clean_tokenizer(text):

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = text.split()

    return tokenized

# Collect the unique tokens

token_counts = Counter()

for label, text in df_train:

    token_list = clean_tokenizer(text)
    token_counts.update(token_list)

print(f'Vocab-size: {len(token_counts)}')

# Encoding each unique token into an integer

token_counts_sorted_by_freq = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
token_counts_sorted_by_freq = OrderedDict(token_counts_sorted_by_freq)
vocabulary = vocab(token_counts_sorted_by_freq)
vocabulary.insert_token("<pad>", 0)
vocabulary.insert_token("<unk>", 1)
vocabulary.set_default_index(1)

# Functions to remap labels and transform each text into the corresponding integer

label_pipeline = lambda x: 1.0 if x == 2 else 0.0
text_pipeline = lambda x: [vocabulary[token] for token in clean_tokenizer(x)]

# Wrap the label remapping and text encoding functions

def collate_batch(batch):

    label_list, text_list, lengths = [], [], []

    for label, text in batch:

        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)

    return padded_text_list, label_list, lengths

# Create a dataset

batch_size = 32
train_dl = DataLoader(df_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(df_valid, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_dl  = DataLoader(df_test,  batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


# %% MODEL

# Design the recurrent neural nework

class Model(nn.Module):

    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):

        super().__init__()

        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, 1)

    def forward(self, text, lengths):

        x = self.emb(text)      # x: (B,T,E)
        x = pack_padded_sequence(x, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        x, (h, c) = self.rnn(x) # x: h(1), ..., h(T), (B,T,H) | h: h(T), (L,B,H) | c: c(T), (L,B,H)
        x = h[-1, :, :]         # (B,H)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)

        return x

# Initialize a recurrent neural network object

vocab_size = len(vocabulary)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64

model = Model(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size) 

# Loss function

loss_fun = nn.BCELoss()

# Function to train the model

def train(dataloader):
    
    model.train()
    total_acc, total_loss = 0, 0
    
    for text_batch, label_batch, lengths in dataloader:
        
        pred = model(text_batch, lengths)[:, 0] # text_batch: (B,T); (B, 1) -> (B,)
        loss = loss_fun(pred, label_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * label_batch.size(0)
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        
    return total_loss / len(dataloader.dataset), total_acc / len(dataloader.dataset)

# Function to evaluate the model

def evaluate(dataloader):
    
    model.eval()
    total_acc, total_loss = 0, 0

    with torch.no_grad():
        
        for text_batch, label_batch, lengths in dataloader:
            
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fun(pred, label_batch)
            total_loss += loss.item() * label_batch.size(0)
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            
    return total_loss / len(dataloader.dataset), total_acc / len(dataloader.dataset)


# %% TRAINING

# Parameters

learning_rate = 0.001
num_epochs = 10

# Learn from the data

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    loss_train, acc_train = train(train_dl)
    loss_valid, acc_valid = evaluate(valid_dl)
    print(f'Epoch {epoch} accuracy: {acc_train:.4f} val_accuracy: {acc_valid:.4f}') 


# %% TESTING

# Evaluate the model on the test set

_, acc_test = evaluate(test_dl)
print(f'test_accuracy: {acc_test:.4f}') 