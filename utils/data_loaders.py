# Import the libraries

import os
import tarfile
from urllib.request import urlretrieve

import pyprind
import numpy as np
import pandas as pd
import requests

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

        # Check that the tarball archive exists
        
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

# Function to download or load the gutenberg dataset

def load_gutenberg(text_path='datasets/gutenberg/gutenberg.txt'):
    """
    Loads the Gutenberg text file. If the file does not exist, it downloads it
    first, then removes the Project Gutenberg header and footer.

    Parameters:
    ----------
    text_path : str
        The path to the downloaded text file.

    Returns:
    ----------
    text : str
        The loaded and cleaned Gutenberg text.
    """

    if not os.path.exists(text_path):

        download_url = 'https://www.gutenberg.org/files/1268/1268-0.txt'

        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        urlretrieve(download_url, text_path)

    else:

        print(f'Loading dataset from {text_path}...')

    with open(text_path, 'r', encoding='utf-8') as infile:

        text = infile.read()

    start_idx = text.find('THE MYSTERIOUS ISLAND')
    end_idx = text.find('End of the Project Gutenberg')

    if start_idx == -1 or end_idx == -1:

        raise ValueError('Could not find the Gutenberg start or end marker.')

    text = text[start_idx:end_idx]

    return text

# Function to the cat and dogs dataset

def donwload_cat_dog_dataset(
    dest_dir='datasets/cat_dog',
    repo='rasbt/machine-learning-book',
    path='ch12/cat_dog_images',
    ref='main'):
    """
    Downloads the cat_dog dataset from GitHub if it does not already exist.

    Parameters:
    ----------
    dest_dir : str
        Destination directory for the dataset.
    repo : str
        GitHub repository in the format "owner/name".
    path : str
        Path inside the repository containing the dataset.
    ref : str
        Git reference (branch, tag, or commit).
    """

    def dataset_exists(directory):
        if not os.path.isdir(directory):
            return False
        for _, _, files in os.walk(directory):
            if any(file.lower().endswith('.jpg') for file in files):
                return True
        return False

    def download_github_folder_recursive(repo_name, repo_path, dest, ref_name):
        api_url = f"https://api.github.com/repos/{repo_name}/contents/{repo_path}?ref={ref_name}"
        items = requests.get(api_url, timeout=30).json()
        os.makedirs(dest, exist_ok=True)

        for item in items:
            if item["type"] == "file":
                data = requests.get(item["download_url"], timeout=30).content
                with open(os.path.join(dest, item["name"]), "wb") as f:
                    f.write(data)
            elif item["type"] == "dir":
                sub_dest = os.path.join(dest, item["name"])
                download_github_folder_recursive(repo_name, item["path"], sub_dest, ref_name)

    if dataset_exists(dest_dir):
        print("The cat_dog dataset already exists.")
        return None

    download_github_folder_recursive(repo, path, dest_dir, ref)
    print("The cat_dog dataset has been downloaded.")
