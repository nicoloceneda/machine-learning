# %% SETUP
 
# Import the libraries

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
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
df_train = torchvision.datasets.MNIST(root='datasets', train=True, transform=transform, download=False)
df_test = torchvision.datasets.MNIST(root='datasets', train=False, transform=transform, download=False)

# Create a dataset

dlg = torch.Generator(device=device)
dlg.manual_seed(2)
train_dl = DataLoader(df_train, batch_size=64, shuffle=True, generator=dlg)


# %% MODEL

# Design the multilayer perceptron

image_shape = df_train[0][0].shape
input_size = image_shape[0] * image_shape[1] * image_shape[2]
output_size = len(df_train.classes)
hidden_size = [32, 16]

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, hidden_size[0]),
    nn.ReLU(),
    nn.Linear(hidden_size[0], hidden_size[1]),
    nn.ReLU(),
    nn.Linear(hidden_size[1], output_size),
)

# Loss function

loss_fun = nn.CrossEntropyLoss()


# %% TRAINING

# Parameters

learning_rate = 0.001
num_epochs = 20

# Learn from the data

acc_hist = torch.zeros(num_epochs)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    
    for x_batch, y_batch in train_dl:

        pred = model(x_batch)
        loss = loss_fun(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct = (torch.argmax(pred, dim=1) == y_batch).float()
        acc_hist[epoch] += correct.sum()

    acc_hist[epoch] /= len(train_dl.dataset)
    print(f'Epoch {epoch}  Accuracy {acc_hist[epoch].item():.4f}')


# %% TESTING

# Evaluate the model on the test set

model.eval()
with torch.no_grad():
    
    pred = model(df_test.data / 255.)
    correct = (torch.argmax(pred, dim=1) == df_test.targets).float()
    print(f'Test accuracy: {correct.mean():.4f}')

# %%
