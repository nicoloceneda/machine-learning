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
import xgboost as xgb

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

# Initialize an XGBoost object

xgb = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=1)


# %% TRAINING

# Learn from the data via the fit method

xgb.fit(X_train, y_train)


# %% TESTING

# Predict the classes of the samples in the test set

y_xgb_train_predict = xgb.predict(X_train)
y_xgb_test_predict = xgb.predict(X_test)

# Evaluate the performance of the model

print(f'Tree train accuracy: {accuracy_score(y_train, y_xgb_train_predict):.3f}')
print(f'Tree test accuracy: {accuracy_score(y_test, y_xgb_test_predict):.3f}')


# %% DECISION REGIONS

# Plot the decision boundary

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined, y=y_combined, classifier=xgb, test_idx=range(95, 118))
plt.title('Decision boundary and training sample')
plt.xlabel('Alcohol')
plt.ylabel('OD280/OD315 of diluted wines')
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, 'Decision_boundary.png'))


# %% GENERAL

# Show plots

plt.show()