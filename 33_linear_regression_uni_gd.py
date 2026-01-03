# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
data = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

# Convert feature from string to binary

data['Central Air'] = data['Central Air'].map({'N': 0, 'Y': 1})

# Drop observations with missing values

data = data.dropna(axis=0)

# Extract the class labels

y = data['SalePrice'].values

# Extract the features

X = data[['Gr Liv Area']].values

# Apply the standardization to scale the features and target variable

std_scaler_x = StandardScaler()
std_scaler_y = StandardScaler()
X_std = std_scaler_x.fit_transform(X)
y_std = std_scaler_y.fit_transform(y[:, np.newaxis]).flatten()


# %% MODEL

# Design the linear regression

class LinearRegressionGD:
    """ 
    Linear regression

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
    loss_fun : list
        Mean squared error loss function in each epoch.
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
            Training vectors.
        y : array, shape = [n_samples, ]
            Target values.

        Returns:
        -------
        self : object
        """

        rgen = np.random.RandomState(seed=1)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = 0.0
        self.loss_fun = []

        for epoch in range(self.n_epochs):

            update = y - self.net_input(X)
            self.w += self.eta * np.dot(X.T, update) / X.shape[0]
            self.b += self.eta * np.sum(update) / X.shape[0]
            loss = 0.5 * np.mean(update ** 2)
            self.loss_fun.append(loss)

        return self

    def net_input(self, X):
        """ 
        Calculate the net input
        """

        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """ 
        Calculate the net input (Used in plot_decision_regions function)

        Parameters:
        ----------
        X : array, shape = [X0X1_combs.shape[0], n_features]

        Returns:
        -------
        prediction : array, shape = [X0X1_combs.shape[0], ]
        """

        return self.net_input(X)

# Initialize a linear regression object

lr = LinearRegressionGD(eta=0.1, n_epochs=50)


# %%  TRAINING

# Learn from the data via the fit method

lr.fit(X_std, y_std)

print(f'Slope: {lr.w[0]:.3f}')
print(f'Intercept: {lr.b:.3f}')

# Plot the loss function per epoch

plt.figure()
plt.plot(range(1, lr.n_epochs + 1), lr.loss_fun, marker='o')
plt.title('Loss function per epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean squared errors')
plt.savefig(os.path.join(save_dir, 'Loss_per_epoch.png'))

# Plot the best fit line

plt.figure()
plt.scatter(X_std, y_std, color='blue', marker='o', edgecolor='black')
plt.plot(X_std, lr.predict(X_std), color='black', lw=2)
plt.title('Scatter plot of the dependent and independent variable')
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.savefig(os.path.join(save_dir, 'Best_fit_line.png'))


# %% GENERAL

# Show plots

plt.show()