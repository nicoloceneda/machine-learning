# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Simulate the dataset

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5,3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)


# %% MODEL

# Initialize a complete linkage agglomerative hierarchical clustering object

ac = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')


# %% TRAINING AND TESTING

# Learn from the data via the fit method and predict the clusters of the samples

y_predict = ac.fit_predict(X)
print(f'Cluster labels: {y_predict}')