import os
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants from esc50_mscnn.py
SAMPLE_RATE = 44100
N_MELS = 56
N_FFT = 2048
HOP_LENGTH = 512
MIXUP_ALPHA = 0.2
TIME_MASK_WIDTH = 20
FREQ_MASK_WIDTH = 10

# Create directories if they don't exist
os.makedirs('reports/figures/spectrograms', exist_ok=True)

def load_audio(file_path):
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(file_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    return waveform

def create_spectrogram(waveform):
    """Create mel spectrogram from waveform with the same parameters as the model."""
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )(waveform)
    
    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-6)
    
    # Normalize
    mean = mel_spec.mean()
    std = mel_spec.std()
    mel_spec = (mel_spec - mean) / (std + 1e-6)
    
    return mel_spec

def apply_mixup(spec1, spec2, alpha=MIXUP_ALPHA):
    """Apply mixup augmentation."""
    lam = np.random.beta(alpha, alpha)
    mixed_spec = lam * spec1 + (1 - lam) * spec2
    return mixed_spec

def apply_spectrogram_masking(spec, time_mask_width=TIME_MASK_WIDTH, freq_mask_width=FREQ_MASK_WIDTH):
    """Apply time and frequency masking."""
    t = spec.shape[-1]
    f = spec.shape[-2]
    
    # Time masking
    t0 = np.random.randint(0, t - time_mask_width)
    spec[:, :, t0:t0+time_mask_width] = 0
    
    # Frequency masking
    f0 = np.random.randint(0, f - freq_mask_width)
    spec[:, f0:f0+freq_mask_width, :] = 0
    
    return spec

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

def plot_multiple_spectrograms(spectrograms, titles, save_path=None):
    """Plot multiple spectrograms in a grid."""
    if not spectrograms:  # Handle empty list
        logging.warning("No spectrograms to plot")
        return
        
    n_plots = len(spectrograms)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols if n_cols > 0 else 0
    
    if n_rows == 0:  # Handle case where n_cols is 0
        logging.warning("No valid spectrograms to plot")
        return
    
    fig = plt.figure(figsize=(15, 5*n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for i, (spec, title) in enumerate(zip(spectrograms, titles)):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(spec.squeeze().numpy(), 
                      aspect='auto', 
                      origin='lower',
                      cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Mel Frequency')
        plt.colorbar(im, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Download ESC-50 dataset if not present
    if not os.path.exists('ESC-50'):
        logging.info("Downloading ESC-50 dataset...")
        os.system('wget https://github.com/karolpiczak/ESC-50/archive/master.zip')
        os.system('unzip master.zip')
        os.system('mv ESC-50-master ESC-50')
        os.system('rm master.zip')
    
    # Load metadata
    meta_file = 'ESC-50/meta/esc50.csv'
    meta_data = pd.read_csv(meta_file)
    
    # Select representative classes (using exact names from ESC-50)
    selected_classes = [
        'Dog',          # Animal sound
        'Rain',         # Natural sound
        'Crying baby',  # Human sound
        'Door knock',   # Interior sound
        'Siren'         # Exterior sound
    ]
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process each selected class
    for class_name in selected_classes:
        logging.info(f"Processing class: {class_name}")
        
        # Get all files for this class
        class_files = meta_data[meta_data['category'] == class_name]['filename'].tolist()
        
        if not class_files:
            logging.warning(f"No files found for class: {class_name}")
            continue
            
        # Process first 3 examples
        original_spectrograms = []
        augmented_spectrograms = []
        titles = []
        
        for i, filename in enumerate(class_files[:3]):
            audio_path = os.path.join('ESC-50/audio', filename)
            if not os.path.exists(audio_path):
                logging.warning(f"Audio file not found: {audio_path}")
                continue
                
            waveform = load_audio(audio_path)
            
            # Create original spectrogram
            original_spec = create_spectrogram(waveform)
            original_spectrograms.append(original_spec)
            
            # Create augmented spectrogram
            augmented_spec = original_spec.clone()
            augmented_spec = apply_spectrogram_masking(augmented_spec)
            
            # If we have multiple examples, apply mixup
            if i > 0:
                augmented_spec = apply_mixup(augmented_spec, original_spectrograms[i-1])
            
            augmented_spectrograms.append(augmented_spec)
            titles.append(f'{class_name} Example {i+1}')
        
        if not original_spectrograms:
            logging.warning(f"No valid spectrograms generated for class: {class_name}")
            continue
            
        # Save original spectrograms
        save_path = f'reports/figures/spectrograms/{class_name.replace(" ", "_")}_original_{timestamp}.png'
        plot_multiple_spectrograms(original_spectrograms, titles, save_path)
        
        # Save augmented spectrograms
        save_path = f'reports/figures/spectrograms/{class_name.replace(" ", "_")}_augmented_{timestamp}.png'
        plot_multiple_spectrograms(augmented_spectrograms, titles, save_path)
        
        # Save comparison of original vs augmented for each example
        for i in range(len(original_spectrograms)):
            comparison_spectrograms = [original_spectrograms[i], augmented_spectrograms[i]]
            comparison_titles = [f'Original {titles[i]}', f'Augmented {titles[i]}']
            save_path = f'reports/figures/spectrograms/{class_name.replace(" ", "_")}_comparison_example{i+1}_{timestamp}.png'
            plot_multiple_spectrograms(comparison_spectrograms, comparison_titles, save_path)
    
    logging.info("Spectrogram visualization completed. Check reports/figures/spectrograms/ for the results.")

if __name__ == '__main__':
    main() 