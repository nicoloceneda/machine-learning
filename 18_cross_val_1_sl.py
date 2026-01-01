# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

# Extract the class labels

y = data.loc[:, 1].values

# Extract the features

X = data.loc[:, 2:].values

# Encode the categorical class labels

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Separate the data into train and test subsets with the same proportions of class labels as the input dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# %% MODEL

# Create a pipeline with data preprocessing, dimensionality reduction, and model

pipe = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())


# %% IN-SAMPLE PERFORMANCE

# Stratified K-fold cross-validation

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

scores = []

for k, (train, valid) in enumerate(kfold):

    pipe.fit(X_train[train], y_train[train])
    score = pipe.score(X_train[valid], y_train[valid])
    scores.append(score)

    print(f'Fold: {k+1}\n',
    f'Class distr. train: {np.bincount(y_train[train])}\n',
    f'Class distr. valid: {np.bincount(y_train[valid])}\n',
    f'Acc.: {score:.3f}')

print(f'Cross-validation accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')


# %% TRAINING

# Learn from the data via the fit method

pipe.fit(X_train, y_train)


# %% TESTING

# Predict the classes of the samples in the test set

y_pred = pipe.predict(X_test)

# Evaluate the performance of the model

print(f'Predicton accuracy: {pipe.score(X_test, y_test):.3f}')