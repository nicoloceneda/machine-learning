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


# %% SCATTER MATRICES

# Compute the mean vector for each of the class labels

classes = sorted(np.unique(y_train))
mean_classes = []

for label in classes:

    mean_class = np.mean(X_train_std[y_train==label, :], axis=0)
    mean_classes.append(mean_class)
    print(f'Mean vector for class {label}:\n {mean_class}\n')

# Compute the winin class scatter matrix

num_feat = X_train_std.shape[1]
sw = np.zeros((num_feat, num_feat))

for label in classes:

    sw += np.cov(X_train_std[y_train==label], rowvar=False)

print(f'Within-class scatter matrix: {sw.shape[0]}x{sw.shape[1]}') 

# Cmpute the between class scatter matrix

mean_feat = np.mean(X_train_std, axis=0).reshape(num_feat, 1)
sb = np.zeros((num_feat, num_feat))

for label, mean_class in zip(classes, mean_classes):

    samples_class = X_train_std[y_train==label, :].shape[0]
    mean_class = mean_class.reshape(num_feat, 1)
    sb += samples_class * np.dot((mean_class - mean_feat), (mean_class - mean_feat).T)

print(f'Between-class scatter matrix: {sb.shape[0]}x{sb.shape[1]}') 


# %% DISCRIMINATORY INFORMATION EXPLAINED

# Compute eigenvalues and eigenvectors

sw_inv_sb = np.dot(np.linalg.inv(sw), sb)
evals, evecs = np.linalg.eig(sw_inv_sb)

# Compute the (cumulative) discriminatory information explained ratios

tot_di = sum(evals.real)
evals_sorted = sorted(evals.real, reverse=True)
di_exp = [(i / tot_di) for i in evals_sorted]
di_exp_cum = np.cumsum(di_exp)

# Plot the (cumulative) discriminatory information explained ratios

plt.figure()
plt.bar(range(1, X_train_std.shape[1]+1), di_exp, align='center', edgecolor='black', label='Individual discriminatory information')
plt.step(range(1, X_train_std.shape[1]+1), di_exp_cum, where='mid', label='Cumulative discriminatory information')
plt.ylabel('Discriminatory information ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Discriminatory_information_explained.png'))


# %% LINEAR DISCRIMINANTS

# Sort the (evals, evecs) tuples from high to low

epairs = [(np.abs(evals[i]), evecs[:, i]) for i in range(len(evals))]
epairs.sort(key=lambda k: k[0], reverse=True)

# Build the transformation matrix from top two evecs

w = np.hstack((epairs[0][1][:, np.newaxis].real, epairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

# Transform the data

X_train_lda = np.dot(X_train_std, w)

# Plot the transformed data in a scatter plot

plt.figure()
colors = ['r', 'b', 'g']
for l, c in zip(np.unique(y_train), colors):
    plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1] * -1, c=c, label=f'Class {l}')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.savefig(os.path.join(save_dir, 'Linear_discriminants.png'))


# %% GENERAL

# Show plots

plt.show()