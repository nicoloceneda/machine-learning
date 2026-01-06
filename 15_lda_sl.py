# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

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


# %% MODEL

# Initialize an LDA object

lda = LDA(n_components=2)

# Initialize a logistic regression object

logreg = LogisticRegression(random_state=1, solver='lbfgs', multi_class='ovr')


# %% TRAINING

# Transform the data

X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

# Learn from the data via the fit method

logreg.fit(X_train_lda, y_train)


# %% DECISION REGIONS

# Plot the decision region and the data

plot_decision_regions(X=X_train_lda, y=y_train, classifier=logreg)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Decision_boundary.png'))


# %% GENERAL

# Show plots

plt.show()