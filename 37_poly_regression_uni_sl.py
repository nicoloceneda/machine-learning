# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

# Remove the outliers

data = data[data['Gr Liv Area'] < 4_000]

# Extract the class labels

y = data['SalePrice'].values

# Extract the features

X = data[['Overall Qual']].values

# Polynomial features

quadratic = PolynomialFeatures(degree=2, include_bias=False)
cubic = PolynomialFeatures(degree=3, include_bias=False)

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# %%

# Separate the data into train and test subsets

X_train, X_test, X_quad_train, X_quad_test, X_cubic_train, X_cubic_test, y_train, y_test = train_test_split(X, X_quad, X_cubic, y, test_size=0.3, random_state=123)


# %% MODEL

# Initialize a linear regression object

lr = LinearRegression(fit_intercept=True)
lr_quad = LinearRegression(fit_intercept=True)
lr_cubic = LinearRegression(fit_intercept=True)


# %% TRAINING

# Learn from the data via the fit method

lr.fit(X_train, y_train)
lr_quad.fit(X_quad_train, y_train)
lr_cubic.fit(X_cubic_train, y_train)

print('Linear Regression')
print(f'Slope: {lr.coef_[0]:.3f}')
print(f'Intercept: {lr.intercept_:.3f}')

print('\nQuadratic Regression')
print(f'Slopes: {[f'{c:.3f}' for c in lr_quad.coef_]}')
print(f'Intercept: {lr_quad.intercept_:.3f}')

print('\nCubic Regression')
print(f'Slopes: {[f'{c:.3f}' for c in lr_cubic.coef_]}')
print(f'Intercept: {lr_cubic.intercept_:.3f}')

# Plot the best fit line

X_fit = np.linspace(X_train.min()-1, X_train.max()+1, 1000)[:, np.newaxis]
y_fit_pred = lr.predict(X_fit)
y_fit_pred_quad = lr_quad.predict(quadratic.fit_transform(X_fit))
y_fit_pred_cubic = lr_cubic.predict(cubic.fit_transform(X_fit))

plt.figure()
plt.scatter(X_train, y_train, color='blue', marker='o', edgecolor='black')
plt.plot(X_fit, y_fit_pred, color='orange', lw=2, label='Linear')
plt.plot(X_fit, y_fit_pred_quad, color='red', lw=2, label='Quadratic')
plt.plot(X_fit, y_fit_pred_cubic, color='green', lw=2, label='Cubic')
plt.title('Scatter plot of the dependent and independent variable')
plt.xlabel('Overall Qual')
plt.ylabel('SalePrice')
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, 'Best_fit_line.png'))


# %% TESTING

# Predict the values of the samples in the train and test sets

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

y_train_pred_quad = lr_quad.predict(X_quad_train)
y_test_pred_quad = lr_quad.predict(X_quad_test)

y_train_pred_cubic = lr_cubic.predict(X_cubic_train)
y_test_pred_cubic = lr_cubic.predict(X_cubic_test)

# Evaluate the performance of the model

mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print('\nLinear Regression')
print(f'Train:\n MSE={mse_train:.3f}\n MAE={mae_train:.3f}\n R2={r2_train:.3f}')
print(f'Test:\n MSE={mse_test:.3f}\n MAE={mae_test:.3f}\n R2={r2_test:.3f}')

mse_train_quad = mean_squared_error(y_train, y_train_pred_quad)
mae_train_quad = mean_absolute_error(y_train, y_train_pred_quad)
r2_train_quad = r2_score(y_train, y_train_pred_quad)

mse_test_quad = mean_squared_error(y_test, y_test_pred_quad)
mae_test_quad = mean_absolute_error(y_test, y_test_pred_quad)
r2_test_quad = r2_score(y_test, y_test_pred_quad)

print('\nQuadratic Regression')
print(f'Train:\n MSE={mse_train_quad:.3f}\n MAE={mae_train_quad:.3f}\n R2={r2_train_quad:.3f}')
print(f'Test:\n MSE={mse_test_quad:.3f}\n MAE={mae_test_quad:.3f}\n R2={r2_test_quad:.3f}')

mse_train_cubic = mean_squared_error(y_train, y_train_pred_cubic)
mae_train_cubic = mean_absolute_error(y_train, y_train_pred_cubic)
r2_train_cubic = r2_score(y_train, y_train_pred_cubic)

mse_test_cubic = mean_squared_error(y_test, y_test_pred_cubic)
mae_test_cubic = mean_absolute_error(y_test, y_test_pred_cubic)
r2_test_cubic = r2_score(y_test, y_test_pred_cubic)

print('\nCubic Regression')
print(f'Train:\n MSE={mse_train_cubic:.3f}\n MAE={mae_train_cubic:.3f}\n R2={r2_train_cubic:.3f}')
print(f'Test:\n MSE={mse_test_cubic:.3f}\n MAE={mae_test_cubic:.3f}\n R2={r2_test_cubic:.3f}')


# %% GENERAL

# Show plots

plt.show()