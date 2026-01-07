# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
data = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

# Convert feature from string to binary

data['Central Air'] = data['Central Air'].map({'N': 0, 'Y': 1})

# Drop observations with missing values

data = data.dropna(axis=0)

# Extract the class labels

y = data['SalePrice'].values

# Extract the features

features = [i for i in data.columns if i != 'SalePrice']
X = data[features].values

# Separate the data into train and test subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# %% MODEL

# Initialize a random forest object

forest = RandomForestRegressor(n_estimators=1000, criterion='squared_error', random_state=1, n_jobs=-1)


# %% TRAINING

# Learn from the data via the fit method

forest.fit(X_train, y_train)


# %% TESTING

# Predict the values of the samples in the train and test sets

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# Evaluate the performance of the model

mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Train:\n MSE={mse_train:.3f}\n MAE={mae_train:.3f}\n R2={r2_train:.3f}')
print(f'Test:\n MSE={mse_test:.3f}\n MAE={mae_test:.3f}\n R2={r2_test:.3f}')

# Plot the residuals against the predicted values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

ax1.scatter(y_train_pred, y_train_pred - y_train, edgecolor='black', label='Train')
ax2.scatter(y_test_pred, y_test_pred - y_test, edgecolor='black', label='Test')

for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)

ax1.set_ylabel('Residuals')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, 'Residuals.png'))


# %% GENERAL

# Show plots

plt.show()