# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

# Create a pipeline for data preprocessing and model

pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', max_iter=10_000))


# %% IN-SAMPLE PERFORMANCE

# Validation curves

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe, X=X_train, y=y_train, param_name='logisticregression__C', param_range=param_range, cv=10, n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curves

plt.figure()
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training set')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation set')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.xscale('log')
plt.ylim([0.8, 1.0])
plt.grid()
plt.savefig(os.path.join(save_dir, 'Validation_curves.png'))


# %% TRAINING

# Learn from the data via the fit method

pipe.fit(X_train, y_train)


# %% TESTING

# Predict the classes of the samples in the test set

y_pred = pipe.predict(X_test)

# Evaluate the performance of the model

print(f'Predicton accuracy: {pipe.score(X_test, y_test):.3f}')


# %% GENERAL

# Show plots

plt.show()
