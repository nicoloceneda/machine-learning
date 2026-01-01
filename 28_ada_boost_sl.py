# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_decision_regions

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None, 
                   usecols=[0, 1, 12], names=['Class label', 'Alcohol', 'OD280/OD315 of diluted wines'])

# Drop class 1

data = data[data['Class label'] != 1]

# Extract the class labels

y = data['Class label'].values

# Extract the features

X = data[['Alcohol', 'OD280/OD315 of diluted wines']].values

# Encode the categorical class labels

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Separate the data into train and test subsets with the same proportions of class labels as the input dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# %% MODEL

# Initialize an AdaBoost object with a decision tree as base estimator

tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1)
ada = AdaBoostClassifier(estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)


# %% DECISION REGIONS

# Plot the decision boundary

X0_min, X0_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
X1_min, X1_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
X0_grid, X1_grid = np.meshgrid(np.arange(X0_min, X0_max, 0.1), np.arange(X1_min, X1_max, 0.1))
X0X1_combs = np.array([X0_grid.ravel(), X1_grid.ravel()]).T

fig, ax = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(7,5))

for i, clf, label in zip([0,1], [tree, ada], ['Tree', 'AdaBoost']):

    clf.fit(X_train, y_train)
    Z = clf.predict(X0X1_combs)
    Z = Z.reshape(X0_grid.shape)

    ax[i].contourf(X0_grid, X1_grid, Z, alpha=0.3)
    ax[i].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', marker='+', s=50)
    ax[i].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='green', marker='+', s=50)
    ax[i].set_title(label)
    ax[0].set_ylabel('OD280/OD315 of diluted wines')
    plt.text(0, -0.2, s='Alcohol', ha='center', va='center', transform=ax[1].transAxes, fontsize=12)

fig.savefig(os.path.join(save_dir, 'Decision_regions.png'))


# %% TRAINING

# Learn from the data via the fit method

tree.fit(X_train, y_train)
ada.fit(X_train, y_train)


# %% TESTING

# Predict the classes of the samples in the test set

y_tree_train_predict = tree.predict(X_train)
y_tree_test_predict = tree.predict(X_test)
y_ada_train_predict = ada.predict(X_train)
y_ada_test_predict = ada.predict(X_test)

# Evaluate the performance of the model

print(f'Tree train accuracy: {accuracy_score(y_train, y_tree_train_predict):.3f}')
print(f'Tree test accuracy: {accuracy_score(y_test, y_tree_test_predict):.3f}')
print(f'Ada train accuracy: {accuracy_score(y_train, y_ada_train_predict):.3f}')
print(f'Ada test accuracy: {accuracy_score(y_test, y_ada_test_predict):.3f}')


# %% GENERAL

# Show plots

plt.show()

