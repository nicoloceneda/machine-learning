# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# Create a pipeline for data preprocessing and model

pipe = make_pipeline(StandardScaler(), SVC(random_state=1))


# %% TRAINING

# Grid search cross-validation

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']}, 
              {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)

# Run cross-validation and retrain the model on the entire training dataset

gs.fit(X_train, y_train)

# Print the best score and the best parameters

print(f'Best score: {gs.best_score_:.3f}')
print(f'Best parameters: {gs.best_params_}')


# %% TESTING

# Predict the classes of the samples in the test set

y_pred = gs.predict(X_test)

# Evaluate the performance of the model

print(f'Prediction accuracy: {gs.score(X_test, y_test):.3f}')