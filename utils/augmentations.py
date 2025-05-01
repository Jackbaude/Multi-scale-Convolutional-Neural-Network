import numpy as np
import torch

def mixup(x1, y1, x2, y2, alpha=0.2):
    """Mixup augmentation for both CNN and classical models."""
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y

def mask_spectrogram(spec, time_mask_width=20, freq_mask_width=10):
    """Time and frequency masking for spectrograms."""
    t = spec.shape[-1]
    f = spec.shape[-2]
    
    # Time masking
    t0 = np.random.randint(0, t - time_mask_width)
    spec[:, :, t0:t0+time_mask_width] = 0
    
    # Frequency masking
    f0 = np.random.randint(0, f - freq_mask_width)
    spec[:, f0:f0+freq_mask_width, :] = 0
    
    return spec

def apply_augmentations(features, labels, train=True, mixup_alpha=0.2, time_mask_width=20, freq_mask_width=10):
    """Apply augmentations to features and labels."""
    if not train:
        return features, labels
    
    # Apply mixup
    if np.random.random() < 0.5:
        idx = np.random.permutation(len(features))
        features, labels = mixup(features, labels, features[idx], labels[idx])
    
    # Apply masking if features are 2D (spectrogram)
    if len(features.shape) == 2:
        features = mask_spectrogram(features, time_mask_width, freq_mask_width)
    
    return features, labels 