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

# Apply the standardization to scale the features

X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Plot the features in a scatter plot

plt.figure()
plt.scatter(X_std[:50, 0], X_std[:50, 1], color="red", marker="+", label="Setosa")
plt.scatter(X_std[50:, 0], X_std[50:, 1], color="blue", marker="+", label="Versicolor")
plt.title("Scatter plot of the scaled features")
plt.xlabel("Sepal length [standardized]")
plt.ylabel("Petal length [standardized]")
plt.legend(loc="upper left")
plt.savefig(os.path.join(save_dir, 'Scatter_plot_features.png'))


# %% MODEL

# Design the logistic regression

class LogisticRegressionGD:
    """ 
    Logistic regression classifier

    Parameters:
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_epochs : int
        Number of epochs.

    Attributes:
    ----------
    w : array, shape = [n_features+1, ]
        Weights after fitting.
    loss_fun : list
        Logistic loss function value in each epoch.
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
        --------
        self : object
        """

        rgen = np.random.RandomState(seed=1)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = 0.0
        self.loss_fun = []

        for epoch in range(self.n_epochs):

            phi_z = self.activation(self.net_input(X))
            update = y - phi_z
            self.w += self.eta * 2 * np.dot(X.T, update) / X.shape[0]
            self.b += self.eta * 2 * np.sum(update) / X.shape[0]
            loss = - np.dot(y, np.log(phi_z)) - np.dot((1 - y), (1 - phi_z))
            self.loss_fun.append(loss)

        return self

    def net_input(self, X):
        """ 
        Calculate the net input
        """

        return np.dot(X, self.w) + self.b
    
    def activation(self, z):
        """ 
        Calculate the sigmoid activation (Used in the fit method)

        Parameters:
        ----------
        X : array, shape = [n_samples, n_features]

        Returns:
        --------
        activation : array, shape = [n_samples, ]
        """

        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """ 
        Calculate the net input, activation and return the class label prediction after the unit 
        step function (Used in plot_decision_regions function)

        Parameters:
        ----------
        X : array, shape = [X0X1_combs.shape[0], ]

        Returns:
        -------
        step_activ : array, shape = [X0X1_combs.shape[0], ]
        """

        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

# Initialize a logistic regression object

logreg = LogisticRegressionGD(eta=0.05, n_epochs=1000)


# %% TRAINING

# Learn from the data via the fit method

logreg.fit(X_std, y)

# Plot the loss function per epoch

plt.figure()
plt.plot(range(1, len(logreg.loss_fun) + 1), logreg.loss_fun, marker='o')
plt.title('Loss function per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss function')
plt.savefig(os.path.join(save_dir, 'Loss_per_epoch.png'))


# %% DECISION REGIONS

# Plot the decision region and the data

plot_decision_regions(X_std, y, classifier=logreg)
plt.title('Decision boundary and training sample')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, 'Decision_boundary.png'))


# %% GENERAL

# Show plots

plt.show()