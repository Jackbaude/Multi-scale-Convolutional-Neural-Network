import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class ESC50Dataset(Dataset):
    def __init__(self, audio_dir, meta_file, transform=None, train=True):
        self.audio_dir = audio_dir
        self.meta_data = pd.read_csv(meta_file)
        self.transform = transform
        self.train = train
        
        # Constants for audio processing
        self.SAMPLE_RATE = 44100
        self.N_MELS = 56
        self.N_FFT = 2048
        self.HOP_LENGTH = 512
        
        # Create mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS
        )
        
    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.meta_data.iloc[idx]['filename'])
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-6)
        
        # Normalize
        mean = mel_spec.mean()
        std = mel_spec.std()
        mel_spec = (mel_spec - mean) / (std + 1e-6)
        
        # Apply data augmentation if training
        if self.train and self.transform:
            mel_spec = self.transform(mel_spec)
            
        label = self.meta_data.iloc[idx]['target']
        # Convert label to long tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return mel_spec, label 