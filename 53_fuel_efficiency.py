# %% SETUP
 
# Import the libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

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

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
df = pd.read_csv(url, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

# Drop observations with missing values

df = df.dropna(axis=0)
df = df.reset_index(drop=True)

# Separate the data into train and test subsets

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Apply the standardization to scale the numerical features

column_numeric = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']
df_train_std, df_test_std = df_train.copy(), df_test.copy()
df_train_std['Cylinders'] = df_train_std['Cylinders'].astype(float)
df_test_std['Cylinders'] = df_test_std['Cylinders'].astype(float)

std_scaler = StandardScaler()
df_train_std.loc[:, column_numeric] = std_scaler.fit_transform(df_train[column_numeric])
df_test_std.loc[:, column_numeric] = std_scaler.transform(df_test[column_numeric])

# Apply bucketing to the (ordinal) categorical features 

boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_std['Model Year'].values)
df_train_std['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)
v = torch.tensor(df_test_std['Model Year'].values)
df_test_std['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)
column_numeric.append('Model Year Bucketed')

# One-hot encode (nominal) categorical features

origin_count = len(set(df_train_std['Origin']))
origin_encoded = one_hot(torch.tensor(df_train_std['Origin'].values) % origin_count)
X_train_numeric = torch.tensor(df_train_std[column_numeric].values)
X_train_std = torch.cat([X_train_numeric, origin_encoded], dim=1).float()
origin_encoded = one_hot(torch.tensor(df_test_std['Origin'].values) % origin_count)
X_test_numeric = torch.tensor(df_test_std[column_numeric].values)
X_test_std = torch.cat([X_test_numeric, origin_encoded], dim=1).float()

# Extract the labels

y_train = torch.tensor(df_train_std['MPG'].values, dtype=torch.float32)
y_test = torch.tensor(df_test_std['MPG'].values, dtype=torch.float32)

# Create a dataset

dlg = torch.Generator(device=device)
dlg.manual_seed(2)
train_ds = TensorDataset(X_train_std, y_train)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, generator=dlg)


# %% MODEL

model = nn.Sequential(
    nn.Linear(9, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# Loss function

loss_fun = nn.MSELoss()


# %% TRAINING

# Parameters

learning_rate = 0.001
num_epochs = 200
log_epochs = 20
torch.manual_seed(1)

# Learn from the data

loss_hist_train = torch.zeros(num_epochs)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    for x_batch, y_batch in train_dl:

        pred = model(x_batch)[:, 0] # (batch, 1) -> (batch,)
        loss = loss_fun(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist_train[epoch] += loss.item() * y_batch.shape[0]

    loss_hist_train[epoch] /= X_train_std.shape[0]

    if epoch % log_epochs == 0:

        print(f'Epoch {epoch} Loss {loss_hist_train[epoch].item():.4f}')


# %% TESTING

# Evaluate the model on the test set

model.eval()
with torch.no_grad():
    
    pred = model(X_test_std.float())[:, 0]
    loss = loss_fun(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}') 
