# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap

from sklearn.linear_model import LinearRegression

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

lr = LinearRegression()


# %% TRAINING

# Learn from the data via the fit method

lr.fit(X, y)

print(f'Slope: {lr.coef_[0]:.3f}')
print(f'Intercept: {lr.intercept_:.3f}')

# Plot the best fit line

plt.figure()
plt.scatter(X, y, color='blue', marker='o', edgecolor='black')
plt.plot(X, lr.predict(X), color='black', lw=2)
plt.title('Scatter plot of the dependent and independent variable')
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.savefig(os.path.join(save_dir, 'Best_fit_line.png'))


# %% GENERAL

# Show plots

plt.show()