# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

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


# %% MODEL

# Initialize a linear regression object

rr = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=0.95, residual_threshold=None, random_state=123)


# %% TRAINING

# Learn from the data via the fit method

rr.fit(X, y)

print(f'Slope: {rr.estimator_.coef_[0]:.3f}')
print(f'Intercept: {rr.estimator_.intercept_:.3f}')

# Plot the best fit line

inlier_mask = rr.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.linspace(X.min(), X.max(), 100)[:, np.newaxis]
line_y = rr.predict(line_X)

plt.figure()
plt.scatter(X[inlier_mask], y[inlier_mask], color='blue', marker='o', edgecolor='black', label='Inliers', zorder=2)
plt.scatter(X[outlier_mask], y[outlier_mask], color='red', marker='o', edgecolor='black', label='Outliers', zorder=1)
plt.plot(line_X, line_y, color='black', zorder=3)
plt.title('Scatter plot of the dependent and independent variable')
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, 'Best_fit_line.png'))


# %% GENERAL

# Show plots

plt.show()