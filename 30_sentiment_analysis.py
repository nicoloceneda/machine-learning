# %% SETUP

# Import the libraries

import numpy as np
import pandas as pd
from utils import load_imdb_dataset

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


# %% DATA

# Import the dataset

df = load_imdb_dataset()

# Clean the documents

def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    
    return text

df['review'] = df['review'].apply(preprocessor)

# Separate the data into train and test subsets

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


# %% BAG OD WORDS

docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet, and one and one is two']) 

# Compute the raw term frequencies

count = CountVectorizer(ngram_range=(1, 1), lowercase=True)
bag = count.fit_transform(docs)
print(f'Dictionary of raw terms:\n{count.vocabulary_}\n')
print(f'Raw term frequencies:\n {bag.toarray()}\n')

# Compute the term frequency inverse document frequency

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
bag_inv = tfidf.fit_transform(bag)
print(f'Term frequency inverse document frequency:\n{bag_inv.toarray()}')


# %% TOKENS

# Tokenizer that splits the documents using whitespace characters

def tokenizer(text):

    return text.split()

# Tokenizer that splits the documents using the porter stemming algorithm

porter = PorterStemmer()

def tokenizer_porter(text):

    return [porter.stem(word) for word in text.split()]

# Tokenizer that splits the documents using the porter stemming algorithm and stop words

stop = stopwords.words('english')

def tokenizer_stop(text):

    return [w for w in tokenizer_porter(text) if w not in stop]


# %% MODEL

# Create a pipeline for data preprocessing and model

tfidf = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, token_pattern=None)
pipe = Pipeline([('vect', tfidf), ('clf', LogisticRegression(solver='liblinear'))])


# %% TRAINING

# Grid search cross-validation

param_grid = [
    # Equivalent to computing tf-idf as in 'Bag of words' section
    {
    'vect__stop_words': [None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'vect__use_idf':[True],
    'vect__smooth_idf':[True],
    'clf__penalty': ['l2'],
    'clf__C': [1.0, 10.0]
    },
    # Model based on raw term frequencies
    {
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer],
    'vect__use_idf':[False],
    'vect__smooth_idf':[False],
    'vect__norm':[None],
    'clf__penalty': ['l2'],
    'clf__C': [1.0, 10.0]}]

gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

# Run cross-validation and retrain the model on the entire training dataset

gs.fit(X_train, y_train)

# Print the best score and the best parameters

print(f'Best score: {gs.best_score_:.3f}')
print(f'Best parameters: {gs.best_params_}')


# %% TESTING

# Predict the classes of the samples in the test set

y_pred = gs.predict(X_test)

# Evaluate the performance of the model

print(f'Prediction accuracy: {gs.score(X_test, y_test):.3f}')