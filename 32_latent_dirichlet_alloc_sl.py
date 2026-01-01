# %% SETUP

# Import the libraries

import numpy as np
import pandas as pd
from utils import load_imdb_dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# %% DATA

# Import the dataset

df = load_imdb_dataset()


# %% BAG OD WORDS

# Compute the raw term frequencies

count = CountVectorizer(ngram_range=(1, 1), lowercase=True, stop_words='english', max_df=0.1, max_features=5_000)
X = count.fit_transform(df['review'].values)


# %% MODEL

# Initialize a latent dirichlet allocation object

lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')


# %% TRAINING

# Learn from the data via the fit method

X_topics = lda.fit_transform(X)

# Print the five most important words for each of the 10 topics

n_top_words = 5
feature_names = count.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):

    print(f'Topic {(topic_idx + 1)}:')
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Plot three movies from the horror movie categorty

horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):

    print(f'\nHorror movie #{(iter_idx + 1)}:')
    print(df['review'][movie_idx][:300], '...') 