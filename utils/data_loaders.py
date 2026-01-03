# Import the libraries

import os
import tarfile
import pyprind
import numpy as np
import pandas as pd

# Function to download or load the imdb dataset

def load_imdb_dataset(csv_path='datasets/imdb/extracted/imdb_data.csv'):
    """
    Loads the IMDB dataset from a CSV file. If the file does not exist, it extracts
    the original tarball, processes the individual text files into a single CSV,
    and then loads it.
    
    Parameters:
    ----------
    csv_path : str
        The path to the processed CSV file.
    
    Returns:
    ----------
    df : DataFrame
        The loaded IMDB dataset.
    """

    if not os.path.exists(csv_path):

        # check that the tarball archive exists
        
        tarball_path = 'datasets/imdb/original/aclImdb_v1.tar.gz'
        
        if not os.path.exists(tarball_path):

            raise FileNotFoundError(f'Could not find the original tarball at {tarball_path}.')

        # Unpack the gzip compressed tarball archive

        print('Extracting IMDB dataset...')

        extract_path = 'datasets/imdb/original'

        with tarfile.open(tarball_path, 'r:gz') as tar:

            tar.extractall(path=extract_path)

        # Assemble the text documents from the decompressed archive into a single csv file
        
        print("Processing individual text files into a single CSV...")

        pbar = pyprind.ProgBar(50_000)

        labels = {'pos': 1, 'neg': 0}
        data = []

        for s in ('test', 'train'):

            for l in ('pos', 'neg'):
            
                path = os.path.join('datasets/imdb/original/aclImdb', s, l)
            
                if not os.path.exists(path):
            
                    continue
            
                for file in sorted(os.listdir(path)):
            
                    if not file.endswith('.txt'):
            
                        continue
            
                    file_path = os.path.join(path, file)
            
                    with open(file_path, 'r', encoding='utf-8') as infile:
            
                        txt = infile.read()
            
                    data.append([txt, labels[l]])
                    pbar.update()

        df = pd.DataFrame(data, columns=['review', 'sentiment'])

        # Shuffle the dataframe

        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))
        
        # Ensure directory exists before saving

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f'\nSaved processed dataset to {csv_path}')
    
    else:
    
        print(f'Loading dataset from {csv_path}...')
        df = pd.read_csv(csv_path, encoding='utf-8')
    
    return df
