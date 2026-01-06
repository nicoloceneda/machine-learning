# %% SETUP

# Import the libraries

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_decision_regions

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

data = load_iris()

# Extract the class labels

y = data.target

# Extract the features

X = data.data[:, [2, 3]]

# Separate the data into train and test subsets with the same proportions of class labels as the input dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Apply the standardization to scale the features

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)


# %% MODEL

# Initialize a random forest object

forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=-1)


# %% TRAINING

# Learn from the data via the fit method

forest.fit(X_train_std, y_train)


# %% TESTING

# Predict the classes of the samples in the test set

y_predict = forest.predict(X_test_std)

# Evaluate the performance of the model

print('Number of misclassifications: {}'.format(np.sum(y_test != y_predict)))
print('Prediction accuracy: {}'.format(accuracy_score(y_test, y_predict)))


# %% DECISION REGIONS

# Plot the decision region and the data

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=forest, test_idx=range(105, 150))
plt.title('Decision boundary and training sample')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, 'Decision_boundary.png'))


# %% GENERAL

# Show plots

plt.show()
