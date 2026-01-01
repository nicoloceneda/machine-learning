# %% SETUP

# Import the libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from utils import plot_decision_regions

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# Image path

script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]

save_dir = os.path.join('images', script_name)
os.makedirs(save_dir, exist_ok=True)


# %% DATA

# Import the dataset

data = load_digits()

# Plot the first 4 images

fig, ax = plt.subplots(1, 4)

for i in range(4):
    ax[i].imshow(data.images[i], cmap='Greys')

plt.savefig(os.path.join(save_dir, 'Digits.png'))

# Extract the class labels

y = data.target

# Extract the features

X = data.data


# %% MODEL

# Initialize a TSNE object

tsne = TSNE(n_components=2, init='pca', random_state=123)


# %% TRAINING

# Transform the data

X_tsne = tsne.fit_transform(X)


# %% STOCHASTIC NEIGHBOR EMBEDDING

# Plot the transformed data in a scatter plot

plt.figure()
ax = plt.subplot(aspect='equal')

for i in np.unique(y):

    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1])

for i in np.unique(y):

    xtext, ytext = np.median(X_tsne[y == i, :], axis=0)
    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

plt.savefig(os.path.join(save_dir, 'TSNE.png'))

# %% GENERAL

# Show plots

plt.show()