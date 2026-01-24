# Import the libraries

import os
import tarfile
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
