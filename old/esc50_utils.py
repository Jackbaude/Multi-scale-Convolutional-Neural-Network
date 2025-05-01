import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import logging
from datetime import datetime
import urllib.request
import zipfile
import json

# Constants
SAMPLE_RATE = 44100
N_MELS = 56
N_FFT = 2048
HOP_LENGTH = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names for ESC-50 dataset
CLASS_NAMES = [
    # Animals
    "Dog", "Rooster", "Pig", "Cow", "Frog", "Cat", "Hen", "Insect", "Sheep", "Crow",
    # Natural soundscapes & water sounds
    "Rain", "Sea waves", "Crackling fire", "Crickets", "Chirping birds", "Water drops", "Wind", "Pouring water", "Toilet flush", "Thunderstorm",
    # Human, non-speech sounds
    "Crying baby", "Sneezing", "Clapping", "Breathing", "Coughing", "Footsteps", "Laughing", "Brushing teeth", "Snoring", "Drinking",
    # Interior/domestic sounds
    "Door knock", "Mouse click", "Keyboard typing", "Door creak", "Can opening", "Washing machine", "Vacuum cleaner", "Clock alarm", "Clock tick", "Glass breaking",
    # Exterior/urban noises
    "Helicopter", "Chainsaw", "Siren", "Car horn", "Engine", "Train", "Church bells", "Airplane", "Fireworks", "Hand saw"
]

def setup_directories():
    """Create necessary directories for the project."""
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports/logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def setup_logging(script_name):
    """Set up logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'reports/logs/{script_name}_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return timestamp

def download_esc50():
    """Download and extract the ESC-50 dataset."""
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    zip_path = "ESC-50.zip"
    
    if not os.path.exists("ESC-50"):
        logging.info("Downloading ESC-50 dataset...")
        urllib.request.urlretrieve(url, zip_path)
        
        logging.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.rename("ESC-50-master", "ESC-50")
        os.remove(zip_path)
    
    return "ESC-50/audio", "ESC-50/meta/esc50.csv"

class ESC50Dataset(Dataset):
    """Dataset class for ESC-50 audio files."""
    def __init__(self, audio_dir, meta_file):
        self.audio_dir = audio_dir
        self.meta_data = pd.read_csv(meta_file)
        
    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.meta_data.iloc[idx]['filename'])
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Convert to mel spectrogram
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
        
        # Flatten the spectrogram
        features = mel_spec.squeeze().flatten()
        
        label = self.meta_data.iloc[idx]['target']
        return features, label

def save_model(model, model_path, fold, metrics, additional_info=None):
    """Save model and its metadata."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'fold': fold,
        'metrics': metrics
    }
    if additional_info:
        save_dict.update(additional_info)
    
    torch.save(save_dict, model_path)
    logging.info(f"Model for fold {fold} saved to {model_path}")

def save_results(results, results_path):
    """Save evaluation results to JSON file."""
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {results_path}")

def calculate_average_metrics(fold_metrics):
    """Calculate average metrics across folds."""
    return {
        'accuracy': np.mean([fold['metrics']['accuracy'] for fold in fold_metrics]),
        'precision': np.mean([fold['metrics']['precision'] for fold in fold_metrics]),
        'recall': np.mean([fold['metrics']['recall'] for fold in fold_metrics]),
        'f1': np.mean([fold['metrics']['f1'] for fold in fold_metrics])
    }

def log_metrics(metrics, prefix=''):
    """Log metrics to console and file."""
    for metric_name, value in metrics.items():
        logging.info(f'{prefix}{metric_name}: {value:.4f}') 