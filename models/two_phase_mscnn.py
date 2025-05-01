import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
import logging
import os

# logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

class TwoPhaseMSCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        
        # Constants for audio processing
        self.SAMPLE_RATE = 44100
        self.N_MELS = 56  # As per MSCNN paper
        self.N_FFT = 2048  # As per MSCNN paper
        self.HOP_LENGTH = 512  # 50% overlap as per paper
        
        # Create mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS
        )
        
        # Phase 1: Raw waveform processing
        # Scale I: 11
        self.scale1_conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.scale1_conv2 = nn.Conv1d(64, 64, kernel_size=11, stride=1, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.scale1_pool = nn.MaxPool1d(2)
        
        # Scale II: 51
        self.scale2_conv1 = nn.Conv1d(1, 64, kernel_size=51, stride=1, padding=25)
        self.bn3 = nn.BatchNorm1d(64)
        self.scale2_conv2 = nn.Conv1d(64, 64, kernel_size=51, stride=1, padding=25)
        self.bn4 = nn.BatchNorm1d(64)
        self.scale2_pool = nn.MaxPool1d(2)
        
        # Scale III: 101
        self.scale3_conv1 = nn.Conv1d(1, 64, kernel_size=101, stride=1, padding=50)
        self.bn5 = nn.BatchNorm1d(64)
        self.scale3_conv2 = nn.Conv1d(64, 64, kernel_size=101, stride=1, padding=50)
        self.bn6 = nn.BatchNorm1d(64)
        self.scale3_pool = nn.MaxPool1d(2)
        
        # Phase 2: Spectrogram processing
        self.spec_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.spec_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.spec_pool = nn.MaxPool2d(2)
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        
        # Backend network
        self.backend_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.backend_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.backend_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Phase 1: Raw waveform processing
        # Ensure input is 3D (batch, channels, time)
        if x.dim() == 2:  # [batch, time]
            x = x.unsqueeze(1)  # -> [batch, 1, time]
        logger.debug(f"Input shape after initial processing: {x.shape}")

        # Optional: pad if input is shorter than N_FFT
        if x.shape[-1] < self.N_FFT:
            pad_len = self.N_FFT - x.shape[-1]
            x = F.pad(x, (0, pad_len))  # Pad time dimension only
            logger.debug(f"Input shape after padding: {x.shape}")

        # Scale I
        x1 = F.relu(self.bn1(self.scale1_conv1(x)))
        x1 = F.relu(self.bn2(self.scale1_conv2(x1)))
        x1 = self.scale1_pool(x1)
        logger.debug(f"Scale I output shape: {x1.shape}")

        # Scale II
        x2 = F.relu(self.bn3(self.scale2_conv1(x)))
        x2 = F.relu(self.bn4(self.scale2_conv2(x2)))
        x2 = self.scale2_pool(x2)
        logger.debug(f"Scale II output shape: {x2.shape}")

        # Scale III
        x3 = F.relu(self.bn5(self.scale3_conv1(x)))
        x3 = F.relu(self.bn6(self.scale3_conv2(x3)))
        x3 = self.scale3_pool(x3)
        logger.debug(f"Scale III output shape: {x3.shape}")

        # Phase 2: Spectrogram processing
        # Generate Mel spectrogram
        mel_spec = self.mel_transform(x)  # -> [batch, n_mels, time]
        mel_spec = torch.log(mel_spec + 1e-6)
        logger.debug(f"Mel spectrogram shape: {mel_spec.shape}")
        
        # Ensure correct shape: [batch, 1, n_mels, time]
        if mel_spec.dim() == 3:  # [batch, n_mels, time]
            mel_spec = mel_spec.unsqueeze(1)  # Add channel dimension
        elif mel_spec.dim() == 4:  # [batch, 1, n_mels, time]
            pass  # Already in correct shape
        else:
            raise ValueError(f"Unexpected mel spectrogram shape: {mel_spec.shape}")
        logger.debug(f"Mel spectrogram shape after processing: {mel_spec.shape}")

        # Process spectrogram
        spec = F.relu(self.bn7(self.spec_conv1(mel_spec)))
        spec = F.relu(self.bn8(self.spec_conv2(spec)))
        spec = self.spec_pool(spec)
        logger.debug(f"Spectrogram features shape: {spec.shape}")

        # Feature fusion
        # Reshape time-domain features to match spectrogram dims
        # First, we need to ensure all features have compatible time dimensions
        target_time = spec.shape[-1]
        target_height = spec.shape[-2]  # Get the height dimension from spectrogram
        
        # Reshape 1D tensors to 2D and interpolate
        # x1, x2, x3 are [batch, channels, time]
        # Need to reshape to [batch, channels, height, time]
        x1 = x1.unsqueeze(2).expand(-1, -1, target_height, -1)  # [batch, channels, height, time]
        x2 = x2.unsqueeze(2).expand(-1, -1, target_height, -1)
        x3 = x3.unsqueeze(2).expand(-1, -1, target_height, -1)
        
        # Now interpolate the time dimension
        x1 = F.interpolate(x1, size=(target_height, target_time), mode='nearest')
        x2 = F.interpolate(x2, size=(target_height, target_time), mode='nearest')
        x3 = F.interpolate(x3, size=(target_height, target_time), mode='nearest')
        
        logger.debug(f"Reshaped x1 shape: {x1.shape}")
        logger.debug(f"Reshaped x2 shape: {x2.shape}")
        logger.debug(f"Reshaped x3 shape: {x3.shape}")
        logger.debug(f"Final spec shape: {spec.shape}")

        # Concatenate time and spectrogram features
        fused = torch.cat([x1, x2, x3, spec], dim=1)
        logger.debug(f"Fused features shape: {fused.shape}")

        # Process fused features
        fused = F.relu(self.bn9(self.fusion_conv(fused)))
        logger.debug(f"After fusion conv shape: {fused.shape}")

        # Backend network
        x = F.relu(self.bn10(self.backend_conv1(fused)))
        x = F.relu(self.bn11(self.backend_conv2(x)))
        x = self.backend_pool(x)
        logger.debug(f"After backend network shape: {x.shape}")

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        logger.debug(f"Final output shape: {x.shape}")

        return x

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
            
        return waveform 