# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

# Extract the class labels

y = data.iloc[:, 0].values

# Extract the features

X = data.iloc[:, 1:].values

# Separate the data into train and test subsets with the same proportions of class labels as the input dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Apply the standardization to scale the features

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)


# %% VARIANCE EXPLAINED

# Compute eigenvalues and eigenvectors

cov_mat = np.cov(X_train_std, rowvar=False)
evals, evecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n', evals)

# Compute the (cumulative) variance explained ratios

tot_var = sum(evals)
evals_sorted = sorted(evals, reverse=True)
var_exp = [i / tot_var for i in evals_sorted]
var_exp_cum = np.cumsum(var_exp)

# Plot the (cumulative) variance explained ratios

plt.figure()
plt.bar(range(1, X_train_std.shape[1]+1), var_exp, align='center', edgecolor='black', label='Explained variance')
plt.step(range(1, X_train_std.shape[1]+1), var_exp_cum, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.savefig(os.path.join(save_dir, 'Variance_explained.png'))


# %% PRINCIPAL COMPONENTS

# Sort the (evals, evecs) tuples from high to low

epairs = [(np.abs(evals[i]), evecs[:, i]) for i in range(len(evals))]
epairs.sort(key=lambda k: k[0], reverse=True)

# Build the projection matrix from top two evecs

w = np.hstack((epairs[0][1][:, np.newaxis], epairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

# Transform the data

X_train_pca = np.dot(X_train_std, w)

# Plot the transformed data in a scatter plot

plt.figure()
colors = ['r', 'b', 'g']

for l, c in zip(np.unique(y_train), colors):
    
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=f'Class {l}')

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.savefig(os.path.join(save_dir, 'Principal_components.png'))


# %% FEATURE CONTRIBUTIONS

# Compute the feature contributions

loadings = evecs * np.sqrt(evals)

# Plot the feature contributions of the first PC

fig, ax = plt.subplots()
ax.bar(range(13), loadings[:, 0], align='center', edgecolor='black')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(data.columns[1:], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Feature_contributions.png'))


# %% GENERAL

# Show plots

plt.show()