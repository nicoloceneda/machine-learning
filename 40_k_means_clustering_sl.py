# %% SETUP

# Import the libraries

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

# Plot the features in a scatter plot

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolor='black', s=50, zorder=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Data.png'))


# %% MODEL

# Find the optimal number of clusters

distortions = []

for i in range(1, 11):

    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Distortion.png'))

# Initialize a k-means clustering object

km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)


# %% TRAINING

# Learn from the data via the fit method

km.fit(X)


# %% TESTING

# Predict the clusters of the samples

y_predict = km.predict(X)

# Evaluate the performance of the model

print(f'Distortion: {km.inertia_:.2f}')

# Plot the clusters and the centroids

plt.figure()
plt.scatter(X[y_predict==0, 0], X[y_predict==0, 1], s=50, color='blue', marker='o', edgecolor='black', label='Cluster 1', zorder=2)
plt.scatter(X[y_predict==1, 0], X[y_predict==1, 1], s=50, color='orange', marker='o', edgecolor='black', label='Cluster 2', zorder=2)
plt.scatter(X[y_predict==2, 0], X[y_predict==2, 1], s=50, color='green', marker='o', edgecolor='black', label='Cluster 3', zorder=2)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=150, marker='*', color='red', edgecolor='black', label='Centroids', zorder=3)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Clusters.png'))

# Plot the silhouette coefficients 

cluster_labels = np.unique(y_predict)
n_clusters = len(cluster_labels)

silhouette_vals = silhouette_samples(X, y_predict, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0 
yticks = []

plt.figure()

for i, c in enumerate(cluster_labels):

    c_silhouette_vals = silhouette_vals[y_predict == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Silhouette.png'))


# %% GENERAL

# Show plots

plt.show()
