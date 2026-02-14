# %% SETUP

# Import the libraries

import os
import pathlib
from itertools import islice
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from utils.data_loaders import donwload_cat_dog_dataset

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
dlg = torch.Generator(device=device)
dlg.manual_seed(1)
data_loader = DataLoader(joint_dataset, batch_size=2, shuffle=True, generator=dlg)

for epoch in range(2):
    print(f'Epoch {epoch+1}\n')
    for i, batch in enumerate(data_loader, 1):
        print(f'Batch {i}:\n x: {batch[0]}\n y: {batch[1]}\n')
print()


# %% DATASET FROM FILES ON LOCAL STORAGE DISK

# Download the dataset

donwload_cat_dog_dataset()

# Create a list of file names and labels

imgdir_path = pathlib.Path('datasets/cat_dog')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)

labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
print(labels)

# Visualise the raw images

fig = plt.figure(figsize=(10, 5))

for i, file in enumerate(file_list):

    img = Image.open(file)
    print('Image shape:', np.array(img).shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cat_dog_raw.png'))

# Create a joint dataset and apply transformation to resize images

class ImageDataset(Dataset):

    def __init__(self, file_list, labels, transform=None):
    
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
    
        img = Image.open(self.file_list[idx])
    
        if self.transform is not None:
    
            img = self.transform(img)
    
        label = self.labels[idx]
    
        return img, label

    def __len__(self):
    
        return len(self.labels)

img_height, img_width = 80, 120
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width))])

image_dataset = ImageDataset(file_list, labels, transform)

# Visualise the transformed images

fig = plt.figure(figsize=(10, 5))

for i, file in enumerate(image_dataset):

    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(file[0].numpy().transpose((1, 2, 0)))
    ax.set_title(f'{file[1]}', size=15)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cat_dog_transformed.png'))


# %% DATASETS FROM TORCHVISION

# CalebA dataset

celeba_dataset = torchvision.datasets.CelebA(root="datasets", split='train', target_type='attr', download=False)

# Visualise the images

fig = plt.figure(figsize=(12, 8))

for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):

    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f'{attributes[31]}', size=15)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'celeba_raw.png'))

# Mnist dataset

mnist_dataset = torchvision.datasets.MNIST(root="datasets", train=True, download=True)

# Visualise the images

fig = plt.figure(figsize=(15, 6))

for i, (image, label) in islice(enumerate(mnist_dataset), 10):

    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}', size=15)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mnist_raw.png'))


# %% GENERAL

# Show plots

plt.show()

