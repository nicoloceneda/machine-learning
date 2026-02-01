# %% SETUP

# Import the libraries

import torch
import torch.nn as nn

# Use GPU if available

print('='*20)
print("Torch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print('='*20)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_device(device)

# Seed

torch.manual_seed(1)


# %% DIRECTED ACYCLIC GRAPHS

# Create a graph

def compute_z(a, b, c):

    o1 = torch.sub(a, b)
    o2 = torch.mul(o1, 2)
    z = torch.add(o2, c)

    return z

# Carry out the computation

a = torch.tensor(1, dtype=torch.int64)
b = torch.tensor(2, dtype=torch.int64)
c = torch.tensor(3, dtype=torch.int64)
print('Rank 0 tensor:', compute_z(a, b, c))

a = torch.tensor([1], dtype=torch.int64)
b = torch.tensor([2], dtype=torch.int64)
c = torch.tensor([3], dtype=torch.int64)
print('Rank 1 tensor:', compute_z(a, b, c))

a = torch.tensor([[1]], dtype=torch.int64)
b = torch.tensor([[2]], dtype=torch.int64)
c = torch.tensor([[3]], dtype=torch.int64)
print('Rank 2 tensor:', compute_z(a, b, c))


# %% TENSOR GRADIENTS AND INITIALIZATION

# Tensor for which gradient is computed

w = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
print(w)

# Create a tensor with Glorot initialization

w = torch.empty(2, 3, requires_grad=True)
nn.init.xavier_normal_(w)
print(w)


# %% AUTOMATIC DIFFERENTIATION

# Compute the gradients

w = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
b = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
x = torch.tensor([1.4], dtype=torch.float32)
y = torch.tensor([2.1], dtype=torch.float32)
z = torch.add(torch.mul(w, x), b)
loss = torch.sum(torch.pow(y - z, 2))
loss.backward()
print(f'dL/dw:', w.grad)
print(f'dL/db:', b.grad)

# Verify manually
print(2 * ((w * x + b) - y) * x)
print(2 * ((w * x + b) - y))


# %%
