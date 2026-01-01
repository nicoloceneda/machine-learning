# %% SETUP
 
# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_decision_regions

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(data.head())

# Extract the class labels

y = data.iloc[:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Extract the features

X = data.iloc[:100, [0, 2]].values

# Plot the features in a scatter plot

plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='+', label='Setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='+', label='Versicolor')
plt.title('Scatter plot of the features')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, 'Scatter_plot_features.png'))


# %% MODEL

# Design the perceptron

class Perceptron:
    """ 
    Perceptron classifier

    Parameters:
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_epochs : int
        Number of epochs.

    Attributes:
    ----------
    w : array, shape = [n_features, ]
        Weights after fitting.
    b : scalar
        Bias unit after fitting.
    n_misclass : list
        Number of misclassifications (hence weight updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_epochs=100):

        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):
        """ 
        Fit training set

        Parameters:
        ----------
        X : array, shape = [n_samples, n_features]
            Feature values.
        y : array, shape = [n_samples, ]
            Target values.

        Returns:
        -------
        self : object
        """

        rgen = np.random.RandomState(seed=1)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = 0.0
        self.n_misclass = []

        for epoch in range(self.n_epochs):

            misclass = 0

            for Xi, yi in zip(X, y):

                update = yi - self.predict(Xi)
                self.w += self.eta * update * Xi
                self.b += self.eta * update
                misclass += int(update != 0)

            self.n_misclass.append(misclass)

        return self

    def net_input(self, X):
        """ 
        Calculate the net input
        """

        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        """ 
        Calculate the net input and return the class label prediction after the unit step function
        (Used in the fit method and in plot_decision_regions function)

        Parameters:
        ----------
        X : array, shape = [n_features, ] in fit method
            array, shape = [X0X1_combs.shape[0], n_features] in plot_decision_regions function

        Returns:
        -------
        prediction : int in fit method
                     array, shape = [X0X1_combs.shape[0], ] in plot_decision_regions function
        """

        return np.where(self.net_input(X) >= 0.0, 1, 0)

# Initialize a perceptron object

ppn = Perceptron(eta=0.1, n_epochs=10)


# %% TRAINING

# Learn from the data

ppn.fit(X, y)

# Plot the number of misclassifications per epoch

plt.figure()
plt.plot(range(1, ppn.n_epochs + 1), ppn.n_misclass, marker='o')
plt.title('Misclassifications per epoch')
plt.xlabel('Epoch')
plt.ylabel('Number of misclassifications')
plt.savefig(os.path.join(save_dir, 'Misclassifications_per_epoch.png'))


# %% DECISION REGIONS

# Plot the decision region and the data

plot_decision_regions(X, y, classifier=ppn)
plt.title('Decision boundary and training sample')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, 'Decision_boundary.png'))


# %% GENERAL

# Show plots

plt.show()

# %%
