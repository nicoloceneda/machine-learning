# %% SETUP

# Import the libraries

import pyprind
import numpy as np
import pandas as pd
from utils import load_imdb_dataset

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer


# %% DATA

# Import the dataset

df = load_imdb_dataset()

# Separate the data into train and test subsets

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


# %% TOKENS

# Function to clean the documents and apply the tokenizer

stop = stopwords.words('english')

def clean_tokenizer(text):

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]

    return tokenized


# %% MODEL

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=clean_tokenizer)
logreg = SGDClassifier(loss='log_loss', random_state=1)


# %% TRAINING

# Generator to read in and return one document at a time

def stream_docs(path):

    with open(path, 'r', encoding='utf-8') as csv:

        next(csv) # skip header

        for line in csv:

            text, label = line[:-3], int(line[-2])

            yield text, label

# Function to take a document stream and return a given number of documents

def get_minibatch(doc_string, size):

    docs, y = [], []

    try:

        for _ in range(size):

            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)

    except StopIteration:

        return None, None
    
    return docs, y

# Run out-of-core learning with 45 mini-batches of 1000 documents each

doc_stream = stream_docs(path='datasets/imdb/extracted/imdb_data.csv')
pbar = pyprind.ProgBar(45)

for _ in range(45):

    X_train, y_train = get_minibatch(doc_string=doc_stream, size=1_000)

    if not X_train:

        break

    X_train = vect.transform(X_train)
    logreg.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    pbar.update()


# %% TESTING

# Evaluate the performance of the model using the last 5000 documents

X_test, y_test = get_minibatch(doc_string=doc_stream, size=5_000)
X_test = vect.transform(X_test)

print(f'Prediction accuracy: {logreg.score(X_test, y_test):.3f}')

# Use the last 5000 documents to update the model

logreg = logreg.partial_fit(X_test, y_test)