import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from datetime import datetime
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import ESC50Dataset

def plot_audio_waveform(audio_path, label, save_path):
    """Plot the waveform of an audio clip."""
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Create figure
    plt.figure(figsize=(20, 5), dpi=300)
    
    # Plot waveform
    librosa.display.waveshow(y, sr=sr, color='blue', alpha=0.8)
    
    # Add title and labels
    plt.title(f'Audio Waveform: {label}', fontsize=24, pad=20)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up style
    sns.set_style("whitegrid")
    sns.set_context("poster")
    
    # Initialize dataset
    audio_dir = 'ESC-50/audio'
    meta_file = 'ESC-50/meta/esc50.csv'
    dataset = ESC50Dataset(audio_dir, meta_file, train=True)
    
    # Load metadata for labels
    meta_data = pd.read_csv(meta_file)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'reports/figures/audio_samples_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Select 5 random samples
    num_samples = 5
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"Visualizing {num_samples} random audio samples...")
    for i, idx in enumerate(random_indices):
        # Get sample
        filename = meta_data.iloc[idx]['filename']
        audio_path = os.path.join(audio_dir, filename)
        label = meta_data.iloc[idx]['category']  # Use category instead of target for human-readable labels
        
        # Create save path
        save_path = os.path.join(output_dir, f'sample_{i+1}_{label.replace(" ", "_")}.png')
        
        # Plot and save
        print(f"  - Processing sample {i+1}: {label}")
        plot_audio_waveform(audio_path, label, save_path)
    
    print(f"\nVisualizations saved in: {output_dir}/")

if __name__ == '__main__':
    main() 