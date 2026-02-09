# %% SETUP

# Import the libraries

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Use GPU if available

print('='*20)
print("Torch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print('='*20)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_device(device)

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)

# Seed

np.random.seed(1)
torch.manual_seed(1)


# %% DATA

# Simulate the dataset

X = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(X))
y[X[:, 0] * X[:, 1] < 0] = 0

# Separate the data into train and validation subsets

n_train = 100
X_train = torch.tensor(X[:n_train, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.float32)
X_valid = torch.tensor(X[n_train:, :], dtype=torch.float32)
y_valid = torch.tensor(y[n_train:], dtype=torch.float32)

# Plot the data in a scatter plot

fig = plt.figure(figsize=(6, 6))
plt.plot(X[y==0, 0], X[y==0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(X[y==1, 0], X[y==1, 1], 'o', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$', size=15)
plt.ylabel(r'$x_2$', size=15)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Data.png'))

# Create a dataset

dlg = torch.Generator(device=device)
dlg.manual_seed(1)
batch_size = 2
train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, generator=dlg) 

# %% MODEL

# Design the multilayer perceptron

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Loss function

loss_fun = nn.BCELoss()


# %% TRAINING

# Parameters

learning_rate = 0.015
num_epochs = 200

# Learn from the data

loss_hist_train = torch.zeros(num_epochs)
acc_hist_train = torch.zeros(num_epochs)
loss_hist_valid = torch.zeros(num_epochs)
acc_hist_valid = torch.zeros(num_epochs)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):

    for x_batch, y_batch in train_dl:

        pred = model(x_batch)[:, 0] # (batch, 1) -> (batch,)
        loss = loss_fun(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist_train[epoch] += loss.item() * y_batch.shape[0]
        correct = ((pred >= 0.5).float() == y_batch).float()
        acc_hist_train[epoch] += correct.sum()

    loss_hist_train[epoch] /= X_train.shape[0]
    acc_hist_train[epoch] /= X_train.shape[0]

    pred = model(X_valid)[:, 0]
    loss = loss_fun(pred, y_valid)
    loss_hist_valid[epoch] = loss.item()
    correct = ((pred >= 0.5).float() == y_valid).float()
    acc_hist_valid[epoch] += correct.mean()

# Plot the training and validation loss and accuracy

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(loss_hist_train.detach().cpu().numpy(), lw=3, zorder=1, label='Train')
ax[0].plot(loss_hist_valid.detach().cpu().numpy(), lw=3, zorder=1, label='Valid')
ax[0].set_title('Training and valid loss', size=15)
ax[0].set_xlabel('Epoch', size=15)
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[0].grid(zorder=0)
ax[0].legend(fontsize=12)
ax[1].plot(acc_hist_train.detach().cpu().numpy(), lw=3, label='Train')
ax[1].plot(acc_hist_valid.detach().cpu().numpy(), lw=3, label='Valid')
ax[1].set_title('Training and valid accuracy', size=15, zorder=1)
ax[1].set_xlabel('Epoch', size=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)
ax[1].grid(zorder=0)
ax[1].legend(fontsize=12)
plt.savefig(os.path.join(save_dir, 'Train_valid_loss_acc_boundary.png'))


# %% GENERAL

# Show plots

plt.show()
