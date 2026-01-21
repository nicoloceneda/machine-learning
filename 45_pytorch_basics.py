# %% SETUP

# Import the libraries

import os
import requests
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

# Use GPU if available

print('='*20)
print("Torch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print('='*20)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_device(device)

# Visualisation settings

np.set_printoptions(precision=3)

# Seed

torch.manual_seed(1)


# %% BASICS

# Creating tensors

print('From list')
t_a = torch.tensor([1, 2, 3], dtype=torch.int64)
print(t_a, '\n')

print('Ones')
t_ones = torch.ones(2, 3)
print(t_ones, '\n')

print('Rand')
t_rand = torch.rand(2,3)
print(t_rand, '\n')

# Manipulating data type and shape

print('To')
t_a_new = t_a.to(torch.float32)
print(t_a.dtype, ' --> ', t_a_new.dtype, '\n')

print('Transpose')
t = torch.rand(3,5)
t_tr = torch.transpose(t, 0, 1)
print(t.shape, ' --> ', t_tr.shape, '\n')

print('Reshape')
t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print(t.shape, ' --> ', t_reshape.shape, '\n')

print('Squeeze')
t = torch.zeros(1, 2, 1, 4, 1)
t_sqz = torch.squeeze(t, 2)
print(t.shape, ' --> ', t_sqz.shape, '\n')

# Mathematical operations

print('Element-wise product')
t1 = 2 * torch.rand(5, 2) - 1
print(t1)
t2 = torch.normal(mean=0, std=1, size=(5,2), device='mps:0')
print(t2)
t_elwise = torch.multiply(t1, t2)
print(t_elwise, '\n')

print('Mean')
t_mean = torch.mean(t1, axis=0)
print(t_mean, '\n')

print('Matrix multiplication')
t_matmul = torch.matmul(t1, torch.transpose(t2, 0, 1))
print(t_matmul, '\n')

print('L2 norm')
t_l2 = torch.linalg.norm(t1, ord=2, dim=1)
print(t_l2, '\n')

# Split, stack and concatenate

print('Chunk')
t = torch.rand(6)
print(t)
t_chunk = torch.chunk(t, 3)
print([i for i in t_chunk], '\n')

print('Split')
t = torch.rand(6)
print(t)
t_split = torch.split(t, split_size_or_sections=[4,2])
print([i for i in t_split], '\n')

print('Cat')
t1 = torch.ones(3)
print(t1)
t2 = torch.zeros(2)
print(t2)
t3 = torch.cat([t1, t2], axis=0)
print(t3, '\n')

print('Stack')
t1 = torch.ones(3)
print(t1)
t2 = torch.zeros(3)
print(t2)
t3 = torch.stack([t1, t2], axis=0)
print(t3, '\n')


# %% DATASET FROM TENSORS

# Data loader

print('Data loader')
t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)

for i in data_loader:
    print(i)
print()

print('Data loader with joint dataset')
t_x = torch.rand([4,3], dtype=torch.float32)
t_y = torch.arange(4)
joint_dataset = TensorDataset(t_x, t_y)
data_loader_joint = DataLoader(joint_dataset)

for i in data_loader_joint:
    print(f'x: {i[0]}, y: {i[1]}')
print()

# Batch, shuffle, repeat

print('Data loader with batches')
t = torch.arange(7, dtype=torch.float32)
data_loader = DataLoader(t, batch_size=3, drop_last=True)

for i, batch in enumerate(data_loader, 1):
    print(f'Batch {i}: {batch}')
print()

print('Data loader with joint dataset and shuffled batches')
data_loader_generator = torch.Generator(device=device)
data_loader_generator.manual_seed(1)
data_loader = DataLoader(joint_dataset, batch_size=2, shuffle=True, generator=data_loader_generator)

for epoch in range(2):
    print(f'Epoch {epoch+1}\n')
    for i, batch in enumerate(data_loader, 1):
        print(f'Batch {i}:\n x: {batch[0]}\n y: {batch[1]}\n')
print()


# %% DATASET FROM FILES ON LOCAL STORAGE DISK

# Download the dataset

def download_github_folder(repo, path, dest, ref="main"):
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    items = requests.get(api_url, timeout=30).json()
    os.makedirs(dest, exist_ok=True)

    for item in items:
        if item["type"] == "file":
            data = requests.get(item["download_url"], timeout=30).content
            with open(os.path.join(dest, item["name"]), "wb") as f:
                f.write(data)
        elif item["type"] == "dir":
            sub_dest = os.path.join(dest, item["name"])
            download_github_folder(repo, item["path"], sub_dest, ref)

download_github_folder(
    "rasbt/machine-learning-book",
    "ch12/cat_dog_images",
    "datasets/cat_dog",
)

# Build a dataset from image files stored on local storage disk




# %%
