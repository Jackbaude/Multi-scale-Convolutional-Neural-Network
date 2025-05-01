import os
import logging
from datetime import datetime

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'reports/figures',
        'reports/logs',
        'models',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging(name):
    """Set up logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'reports/logs/{name}_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name) 