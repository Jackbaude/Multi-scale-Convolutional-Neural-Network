import os
import json
import torch
import torchaudio
from datetime import datetime

def extract_features(audio_path, mel_transform, sample_rate=44100, device=None):
    """Extract mel spectrogram features from audio file.
    
    Args:
        audio_path (str): Path to audio file
        mel_transform (torchaudio.transforms.MelSpectrogram): Mel spectrogram transform
        sample_rate (int): Target sample rate
        device (torch.device, optional): Device to move tensors to
    
    Returns:
        torch.Tensor: Normalized mel spectrogram features
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        
    # Move waveform to device if specified
    if device is not None:
        waveform = waveform.to(device)
        mel_transform = mel_transform.to(device)
        
    # Convert to mel spectrogram
    mel_spec = mel_transform(waveform)
    
    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-6)
    
    # Normalize
    mean = mel_spec.mean()
    std = mel_spec.std()
    mel_spec = (mel_spec - mean) / (std + 1e-6)
    
    return mel_spec

def save_model(model, path, fold=None):
    """Save model state dictionary."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def save_results(results, path):
    """Save training results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4) 