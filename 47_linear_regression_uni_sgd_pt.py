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


# %% DATA

# Simulate the dataset

X_train = torch.arange(10, dtype=torch.float32).reshape(10, 1)
y_train = torch.tensor([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6,7.4, 8.0, 9.0], dtype=torch.float32)

# Apply the standardization to scale the features

X_train_std = (X_train - torch.mean(X_train)) / torch.std(X_train)

# Create a dataset

dlg = torch.Generator(device=device)
dlg.manual_seed(1)
train_ds = TensorDataset(X_train_std, y_train)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, generator=dlg)

# Plot the data in a scatter plot

plt.figure()
plt.scatter(X_train.detach().cpu().numpy(), 
            y_train.detach().cpu().numpy(), 
            c='blue', marker='o', edgecolor='black', s=50, zorder=2)
plt.title('Scatter plot of the dependent and independent variable')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Data.png'))


# %% MODEL

# Design the linear regression

input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss function

loss_fun = nn.MSELoss(reduction='mean')


# %% TRAINING

# Parameters

learning_rate = 0.001
num_epochs = 200
log_epochs = 10

# Learn from the data

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    for x_batch, y_batch in train_dl:

        pred = model(x_batch)[:, 0]
        loss = loss_fun(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % log_epochs == 0:

        print(f'Epoch {epoch} Loss {loss.item():.4f}')

print(f'Final parameters: {model.weight.item()}, {model.bias.item()}')

# Plot the best fit line

X_fit = torch.linspace(start=0, end=9, steps=100, dtype=torch.float32).reshape(-1, 1)
X_fit_std = (X_fit - torch.mean(X_train)) / torch.std(X_train)
y_fit_pred = model(X_fit_std)

plt.figure()
plt.scatter(X_train_std.detach().cpu().numpy(), y_train.detach().cpu().numpy(), color='blue', marker='o', s=50, edgecolor='black', zorder=3)
plt.plot(X_fit_std.detach().cpu().numpy(), y_fit_pred.detach().cpu().numpy(), color='black', lw=2, zorder=2)
plt.title('Scatter plot of the dependent and independent variable')
plt.xlabel('x', size=15)
plt.ylabel('y', size=15)
plt.tick_params(axis='both', which='major', labelsize=15) 
plt.grid(zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Best_fit_line.png'))


# %% GENERAL

# Show plots

plt.show()