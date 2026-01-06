# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Separate the data into train and test subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# %% MODEL

# Initialize a ransac regression object

rr = RANSACRegressor(LinearRegression(fit_intercept=True), max_trials=100, min_samples=0.95, residual_threshold=None, random_state=123)


# %% TRAINING

# Learn from the data via the fit method

rr.fit(X_train, y_train)

print(f'Slope: {rr.estimator_.coef_[0]:.3f}')
print(f'Intercept: {rr.estimator_.intercept_:.3f}')

# Plot the best fit line

X_fit = np.linspace(X_train.min()-1, X_train.max()+1, 1000)[:, np.newaxis]
y_fit_pred = rr.predict(X_fit)

inlier_mask = rr.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

plt.figure()
plt.scatter(X_train[inlier_mask], y_train[inlier_mask], color='blue', marker='o', edgecolor='black', label='Inliers')
plt.scatter(X_train[outlier_mask], y_train[outlier_mask], color='red', marker='o', edgecolor='black', label='Outliers')
plt.plot(X_fit, y_fit_pred, color='black', lw=2)
plt.title('Scatter plot of the dependent and independent variable')
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, 'Best_fit_line.png'))


# %% TESTING

# Predict the values of the samples in the train and test sets

y_train_pred = rr.predict(X_train)
y_test_pred = rr.predict(X_test)

# Evaluate the performance of the model

mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Train:\n MSE={mse_train:.3f}\n MAE={mae_train:.3f}\n R2={r2_train:.3f}')
print(f'Test:\n MSE={mse_test:.3f}\n MAE={mae_test:.3f}\n R2={r2_test:.3f}')


# %% GENERAL

# Show plots

plt.show()