import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
import logging

logger = logging.getLogger(__name__)

class ESC50SVM(nn.Module):
    def __init__(self, C=1.0):
        super().__init__()
        self.C = C
        self.scaler = StandardScaler()
        self.svm = LinearSVC(C=C, max_iter=10000, dual=False, random_state=42)
        
        # Constants for audio processing (from paper)
        self.SAMPLE_RATE = 44100
        self.N_MELS = 40  # As per paper
        self.N_FFT = 1024  # As per paper
        self.HOP_LENGTH = 512  # 50% overlap as per paper
        
        # Create mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS
        )
        
    def extract_features(self, audio_path, device=None):
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Move waveform to device if specified
        if device is not None:
            waveform = waveform.to(device)
            self.mel_transform = self.mel_transform.to(device)
            
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale (as per paper)
        mel_spec = torch.log(mel_spec + 1e-6)
        
        # Normalize (as per paper)
        mean = mel_spec.mean()
        std = mel_spec.std()
        mel_spec = (mel_spec - mean) / (std + 1e-6)
        
        # Compute mean and std across time
        mean_features = mel_spec.mean(dim=2).squeeze()
        std_features = mel_spec.std(dim=2).squeeze()
        
        # Compute additional statistics
        min_features = mel_spec.min(dim=2)[0].squeeze()
        max_features = mel_spec.max(dim=2)[0].squeeze()
        median_features = mel_spec.median(dim=2)[0].squeeze()
        
        # Compute temporal features
        # 1. First derivative (delta)
        delta = torch.diff(mel_spec, dim=2)
        delta_mean = delta.mean(dim=2).squeeze()
        delta_std = delta.std(dim=2).squeeze()
        
        # 2. Second derivative (delta-delta)
        delta_delta = torch.diff(delta, dim=2)
        delta_delta_mean = delta_delta.mean(dim=2).squeeze()
        delta_delta_std = delta_delta.std(dim=2).squeeze()
        
        # Concatenate all features
        features = torch.cat([
            mean_features, 
            std_features,
            min_features,
            max_features,
            median_features,
            delta_mean,
            delta_std,
            delta_delta_mean,
            delta_delta_std
        ]).cpu().numpy()
        
        return features
        
    def fit(self, dataset, device):
        # Get the original dataset from the Subset
        original_dataset = dataset.dataset
        
        # Extract features
        X = []
        y = []
        
        # Add progress bar for feature extraction
        with tqdm(dataset.indices, desc='Extracting features') as pbar:
            for idx in pbar:
                audio_path = os.path.join(original_dataset.audio_dir, original_dataset.meta_data.iloc[idx]['filename'])
                features = self.extract_features(audio_path, device=device)
                X.append(features)
                y.append(original_dataset.meta_data.iloc[idx]['target'])
                pbar.set_postfix({'processed': len(X)})
            
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train SVM
        logger.info('Training SVM...')
        self.svm.fit(X, y)
        
        # Calculate training loss (not used for optimization, just for monitoring)
        train_losses = []
        for _ in range(100):  # Fixed number of iterations as per paper
            train_losses.append(1 - self.svm.score(X, y))
        
        return train_losses
        
    def predict(self, dataset, device):
        # Get the original dataset from the Subset
        original_dataset = dataset.dataset
        
        # Extract features
        X = []
        
        # Add progress bar for feature extraction
        with tqdm(dataset.indices, desc='Extracting features') as pbar:
            for idx in pbar:
                audio_path = os.path.join(original_dataset.audio_dir, original_dataset.meta_data.iloc[idx]['filename'])
                features = self.extract_features(audio_path, device=device)
                X.append(features)
                pbar.set_postfix({'processed': len(X)})
            
        X = np.array(X)
        
        # Scale features
        X = self.scaler.transform(X)
        
        # Make predictions
        return self.svm.predict(X)
    
    def predict_proba(self, dataset, device):
        # Get the original dataset from the Subset
        original_dataset = dataset.dataset
        
        # Extract features
        X = []
        
        # Add progress bar for feature extraction
        with tqdm(dataset.indices, desc='Extracting features') as pbar:
            for idx in pbar:
                audio_path = os.path.join(original_dataset.audio_dir, original_dataset.meta_data.iloc[idx]['filename'])
                features = self.extract_features(audio_path, device=device)
                X.append(features)
                pbar.set_postfix({'processed': len(X)})
            
        X = np.array(X)
        
        # Scale features
        X = self.scaler.transform(X)
        
        # Get decision function values
        decision_values = self.svm.decision_function(X)
        
        # Convert to probabilities using softmax
        probabilities = np.exp(decision_values)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities 