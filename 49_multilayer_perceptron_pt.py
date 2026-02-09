# %% SETUP

# Import the libraries

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Import the dataset

data = load_iris()

# Extract the class labels

y = data.target

# Extract the features

X = data.data

# Separate the data into train and test subsets with the same proportions of class labels as the input dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3, random_state=1, stratify=y)

# Apply the standardization to scale the features

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

# Convert numpy arrays to torch arrays

y_train = torch.tensor(y_train, dtype=torch.int32, device=device)
y_test = torch.tensor(y_test, dtype=torch.int32, device=device)
X_train_std = torch.tensor(X_train_std, dtype=torch.float32, device=device)
X_test_std = torch.tensor(X_test_std, dtype=torch.float32, device=device)

# Create a dataset

dlg = torch.Generator(device=device)
dlg.manual_seed(1)
train_ds = TensorDataset(X_train_std, y_train)
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, generator=dlg)


# %% MODEL

# Design the multilayer perceptron

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)

        return x

# Initialize a multilayer perceptron object

input_size = X_train_std.shape[1]
hidden_size = 16
output_size = 3

model = Model(input_size, hidden_size, output_size)

# Loss function

loss_fun = nn.CrossEntropyLoss()

# %% TRAINING

# Parameters

learning_rate = 0.001
num_epochs = 100

# Learn from the data

loss_hist = torch.zeros(num_epochs)
acc_hist = torch.zeros(num_epochs)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    for x_batch, y_batch in train_dl:

        pred = model(x_batch) # (batch, 3)
        loss = loss_fun(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist[epoch] += loss.item() * y_batch.shape[0]
        correct = (torch.argmax(pred, dim=1) == y_batch).float()
        acc_hist[epoch] += correct.sum()

    loss_hist[epoch] /= X_train_std.shape[0]
    acc_hist[epoch] /= X_train_std.shape[0]

# Plot the training loss and accuracy

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(loss_hist.detach().cpu().numpy(), lw=3, zorder=1)
ax[0].set_title('Training loss', size=15)
ax[0].set_xlabel('Epoch', size=15)
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[0].grid(zorder=0)
ax[1].plot(acc_hist.detach().cpu().numpy(), lw=3)
ax[1].set_title('Training accuracy', size=15, zorder=1)
ax[1].set_xlabel('Epoch', size=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)
ax[1].grid(zorder=0)
plt.savefig(os.path.join(save_dir, 'Train_loss_acc.png'))


# %% TESTING

# Evaluate the model on the test set

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_std)
correct = (torch.argmax(y_test_pred, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')


# %% GENERAL

# Show plots

plt.show()
