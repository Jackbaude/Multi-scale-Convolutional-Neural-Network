import os
import urllib.request
import zipfile
import logging

def download_esc50():
    """Download and extract the ESC-50 dataset if not already present."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if dataset is already downloaded
    if os.path.exists('data/ESC-50'):
        logging.info("ESC-50 dataset already exists.")
        return 'data/ESC-50/audio', 'data/ESC-50/meta/esc50.csv'
    
    # Download the dataset
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    zip_path = "data/ESC-50.zip"
    
    logging.info("Downloading ESC-50 dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract the dataset
    logging.info("Extracting ESC-50 dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    
    # Rename the extracted directory
    os.rename('data/ESC-50-master', 'data/ESC-50')
    
    # Remove the zip file
    os.remove(zip_path)
    
    logging.info("ESC-50 dataset downloaded and extracted successfully.")
    return 'data/ESC-50/audio', 'data/ESC-50/meta/esc50.csv' 