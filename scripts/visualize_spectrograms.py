import os
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Add parent directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import ESC50Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create directories if they don't exist
os.makedirs('reports/figures/spectrograms', exist_ok=True)

def plot_spectrogram(spectrogram, title, save_path=None):
    """Plot spectrogram with proper formatting."""
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.squeeze().numpy(), 
              aspect='auto', 
              origin='lower',
              cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Initialize dataset
    dataset = ESC50Dataset('ESC-50/audio', 'ESC-50/meta/esc50.csv', train=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Visualize first 5 examples
    for i in range(5):
        # Get spectrogram and label
        spec, label = dataset[i]
        
        # Get category name from label
        category = dataset.meta_data.iloc[i]['category']
        
        # Plot and save
        save_path = f'reports/figures/spectrograms/example_{i+1}_{category.replace(" ", "_")}_{timestamp}.png'
        plot_spectrogram(spec, f'Example {i+1}: {category}', save_path)
        logging.info(f"Saved spectrogram for {category}")
    
    logging.info("Spectrogram visualization completed. Check reports/figures/spectrograms/ for the results.")

if __name__ == '__main__':
    main() 