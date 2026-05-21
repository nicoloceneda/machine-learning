# %% SETUP
 
# Import the libraries

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader

# Use GPU if available

print('='*20)
print("Torch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print('='*20)

device = torch.device("cpu")
torch.set_default_device(device)


# %% DATA

# Import the dataset

transform = transforms.Compose([transforms.ToTensor()])
df_train = torchvision.datasets.MNIST(root='datasets', train=True, transform=transform, download=True)
df_test = torchvision.datasets.MNIST(root='datasets', train=False, transform=transform, download=True)

# Separate the train data into train and validation subsets

df_valid = Subset(df_train, torch.arange(10_000))
df_train = Subset(df_train, torch.arange(10_000, len(df_train)))

# Create a dataset

dlg = torch.Generator(device=device)
dlg.manual_seed(2)
train_dl = DataLoader(df_train, batch_size=64, shuffle=False, generator=dlg)
valid_dl = DataLoader(df_valid, batch_size=64, shuffle=False, generator=dlg)


# %% MODEL

# Design the convolutional neural network

x = torch.ones((64, 1, 28, 28))
print(f'{"x":>7}: {x.shape}')

model = nn.Sequential()

model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
model.add_module('relu1', nn.ReLU())
print(f'{"conv1":>7}: {model(x).shape}')

model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
print(f'{"pool1":>7}: {model(x).shape}')

model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
model.add_module('relu2', nn.ReLU())
print(f'{"conv2":>7}: {model(x).shape}')

model.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
print(f'{"pool2":>7}: {model(x).shape}')

model.add_module('flatten', nn.Flatten())
print(f'{"flatten":>7}: {model(x).shape}')

model.add_module('fc1', nn.Linear(3136, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))
print(f'{"fc1":>7}: {model(x).shape}')

model.add_module('fc2', nn.Linear(1024, 10))
print(f'{"fc2":>7}: {model(x).shape}')

# Loss function

loss_fun = nn.CrossEntropyLoss()


# %% TRAINING

# Parameters

learning_rate = 0.001
num_epochs = 20

# Learn from the data

loss_hist_train = torch.zeros(num_epochs)
acc_hist_train = torch.zeros(num_epochs)
loss_hist_valid = torch.zeros(num_epochs)
acc_hist_valid = torch.zeros(num_epochs)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    model.train()
    for x_batch, y_batch in train_dl:

        pred = model(x_batch)
        loss = loss_fun(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist_train[epoch] += loss.item() * y_batch.shape[0]
        correct = (torch.argmax(pred, dim=1) == y_batch).float()
        acc_hist_train[epoch] += correct.sum()
    
    loss_hist_train[epoch] /= len(train_dl.dataset)
    acc_hist_train[epoch] /= len(train_dl.dataset)

    model.eval()
    with torch.no_grad():
        
        for x_batch, y_batch in valid_dl:
            pred = model(x_batch)
            loss = loss_fun(pred, y_batch)
            loss_hist_valid[epoch] += loss.item() * y_batch.shape[0]
            correct = (torch.argmax(pred, dim=1) == y_batch).float()
            acc_hist_valid[epoch] += correct.sum()

    loss_hist_valid[epoch] /= len(valid_dl.dataset)
    acc_hist_valid[epoch] /= len(valid_dl.dataset)

    print(f'Epoch {epoch+1} accuracy: '
          f'{acc_hist_train[epoch]:.4f} val_accuracy: '
          f'{acc_hist_valid[epoch]:.4f}')

# Plot the training and validation loss and accuracy

x_arr = np.arange(len(loss_hist_train)) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, loss_hist_train, '-o', label='Train loss')
ax.plot(x_arr, loss_hist_valid, '--<', label='Validation loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, acc_hist_train, '-o', label='Train acc.')
ax.plot(x_arr, acc_hist_valid, '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)


# %% TESTING

# Evaluate the model on the test set

pred = model(df_test.data.unsqueeze(1) / 255.)
is_correct = (torch.argmax(pred, dim=1) == df_test.targets).float()
print(f'Test accuracy: {is_correct.mean():.4f}')

# Show handwritten inputs and predicted labels

fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    img = df_test[i][0][0, :, :]
    pred = model(img.unsqueeze(0).unsqueeze(1))
    y_pred = torch.argmax(pred)
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, y_pred.item(), size=15, color='blue', horizontalalignment='center', 
            verticalalignment='center', transform=ax.transAxes) 


# %% GENERAL

# Show plots

plt.show()