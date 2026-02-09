# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

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

# Compute the condensed distance matrix

cond_dist = pdist(df, metric='euclidean')

# Display the pairwise distances

row_dist = pd.DataFrame(squareform(cond_dist), columns=labels, index=labels)
print(row_dist)

# Apply the complete linkage agglomerative hierarchical clustering

row_clusters = linkage(cond_dist, method='complete')
row_clusters_df = pd.DataFrame(row_clusters, 
                            columns=['row label 1', 'row label 2', 'distance', 'no. of items in cluster'],
                            index=[f'cluster {i + 1 + row_clusters.shape[0]}' for i in range(row_clusters.shape[0])])
print(row_clusters_df)

# Plot the results of the linkage matrix with a dendrogram

fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

axd.set_xticks([])
axd.set_yticks([])

df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')


for i in axd.spines.values():
    i.set_visible(False)

fig.colorbar(cax)
axm.set_xticks(np.arange(df_rowclust.shape[1]))
axm.set_xticklabels(df_rowclust.columns)
axm.set_yticks(np.arange(df_rowclust.shape[0]))
axm.set_yticklabels(df_rowclust.index)

plt.savefig(os.path.join(save_dir, 'Dendrogram.png'))


# %% GENERAL

# Show plots

plt.show()
