# %% SETUP

# Import the libraries

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Plot the features in a scatter plot

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolor='black', s=50, zorder=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Data.png'))


# %% MODEL

# Initialize the clustering objects

km = KMeans(n_clusters=2, random_state=0)
ac = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='complete')
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')


# %% TRAINING AND TESTING

# Learn from the data via the fit method and predict the clusters of the samples

y_predict_km = km.fit_predict(X)
y_predict_ac = ac.fit_predict(X)
y_predict_db = db.fit_predict(X)

# Plot the clusters 

fig, ax = plt.subplots(1, 3, figsize=(12, 3))

ax[0].scatter(X[y_predict_km==0, 0], X[y_predict_km==0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='cluster 1')
ax[0].scatter(X[y_predict_km==1, 0], X[y_predict_km==1, 1], c='red', edgecolor='black', marker='o', s=40, label='cluster 2')
ax[0].set_title('K-means clustering') 
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')
ax[0].legend()

ax[1].scatter(X[y_predict_ac==0, 0], X[y_predict_ac==0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='Cluster 1') 
ax[1].scatter(X[y_predict_ac==1, 0], X[y_predict_ac==1, 1], c='red', edgecolor='black', marker='o', s=40, label='Cluster 2')
ax[1].set_title('Agglomerative clustering')
ax[1].set_xlabel('Feature 1')
ax[1].set_ylabel('Feature 2')

ax[2].scatter(X[y_predict_db==0, 0], X[y_predict_db==0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='Cluster 1')
ax[2].scatter(X[y_predict_db==1, 0], X[y_predict_db==1, 1], c='red', edgecolor='black', marker='o', s=40, label='Cluster 2')
ax[2].set_title('Density clustering')
ax[2].set_xlabel('Feature 1')
ax[2].set_ylabel('Feature 2')

fig.tight_layout()
plt.savefig(os.path.join(save_dir, 'Clusters.png'))


# %% GENERAL

# Show plots

plt.show()

# %%
