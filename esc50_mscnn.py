import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import urllib.request
import zipfile
import logging
from datetime import datetime
import json

# Create reports directory if it doesn't exist
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('reports/logs', exist_ok=True)

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'reports/logs/training_{timestamp}.log'),
        logging.StreamHandler()
    ]
)

# Constants
SAMPLE_RATE = 44100
N_MELS = 56
N_FFT = 2048
HOP_LENGTH = 512
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
MIXUP_ALPHA = 0.2
TIME_MASK_WIDTH = 20
FREQ_MASK_WIDTH = 10
N_SPLITS = 5  # Number of folds for cross-validation

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

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out = torch.cat([out3, out5, out7], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

class MSCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.block1 = MultiScaleBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = MultiScaleBlock(96, 64)  # 32*3 = 96
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = MultiScaleBlock(192, 128)  # 64*3 = 192
        self.pool3 = nn.MaxPool2d(2)
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max = nn.AdaptiveMaxPool2d((1, 1))
        
        # Calculate the input size for the classifier
        # After 3 pooling layers, the spatial dimensions are reduced by 2^3 = 8
        # The number of channels after the last block is 128 * 3 = 384
        # After global pooling and concatenation: 384 * 2 = 768
        self.classifier = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Ensure input is 4D (batch, channel, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # First block
        x = self.block1(x)
        x = self.pool1(x)
        
        # Second block
        x = self.block2(x)
        x = self.pool2(x)
        
        # Third block
        x = self.block3(x)
        x = self.pool3(x)
        
        # Global pooling
        avg = self.global_avg(x).squeeze(-1).squeeze(-1)
        max_ = self.global_max(x).squeeze(-1).squeeze(-1)
        
        # Concatenate and flatten
        x = torch.cat([avg, max_], dim=1)
        
        # Apply dropout and classifier
        x = self.dropout(x)
        out = self.classifier(x)
        
        return out

class ESC50Dataset(Dataset):
    def __init__(self, audio_dir, meta_file, transform=None, train=True):
        self.audio_dir = audio_dir
        self.meta_data = pd.read_csv(meta_file)
        self.transform = transform
        self.train = train
        
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
        
        # Apply data augmentation if training
        if self.train and self.transform:
            mel_spec = self.transform(mel_spec)
            
        label = self.meta_data.iloc[idx]['target']
        # Convert label to long tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return mel_spec, label

def mixup(x1, y1, x2, y2, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y

def mask_spectrogram(spec, time_mask_width=TIME_MASK_WIDTH, freq_mask_width=FREQ_MASK_WIDTH):
    t = spec.shape[-1]
    f = spec.shape[-2]
    
    # Time masking
    t0 = np.random.randint(0, t - time_mask_width)
    spec[:, :, t0:t0+time_mask_width] = 0
    
    # Frequency masking
    f0 = np.random.randint(0, f - freq_mask_width)
    spec[:, f0:f0+freq_mask_width, :] = 0
    
    return spec

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, device, epochs, fold):
    train_losses = []
    val_losses = []
    test_losses = []
    val_accuracies = []
    test_accuracies = []
    
    # Create a dictionary to store training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'val_accuracy': [],
        'test_accuracy': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f'Fold {fold+1}, Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply mixup
            if np.random.random() < 0.5:
                idx = torch.randperm(inputs.size(0))
                inputs, labels = mixup(inputs, labels, inputs[idx], labels[idx])
                labels = labels.long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Test evaluation
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        test_loss = test_loss / len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        test_accuracy = 100 * test_correct / test_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        history['val_accuracy'].append(val_accuracy)
        history['test_accuracy'].append(test_accuracy)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Log metrics
        logging.info(f'Fold {fold+1}, Epoch {epoch+1}/{epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
        logging.info(f'Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        logging.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses, test_losses, val_accuracies, test_accuracies, history

def download_esc50():
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    zip_path = "ESC-50.zip"
    
    if not os.path.exists("ESC-50"):
        print("Downloading ESC-50 dataset...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.rename("ESC-50-master", "ESC-50")
        os.remove(zip_path)
        
    return "ESC-50/audio", "ESC-50/meta/esc50.csv"

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Download and prepare dataset
    audio_dir, meta_file = download_esc50()
    
    # Create dataset
    dataset = ESC50Dataset(audio_dir, meta_file, transform=mask_spectrogram, train=True)
    
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Store results for each fold
    all_fold_results = {
        'fold_histories': [],
        'final_val_accuracies': [],
        'final_test_accuracies': [],
        'final_val_losses': [],
        'final_test_losses': []
    }
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logging.info(f'\nStarting Fold {fold + 1}/{N_SPLITS}')
        
        # Split into train, validation, and test sets
        train_size = int(0.8 * len(train_idx))
        train_idx, test_idx = train_idx[:train_size], train_idx[train_size:]
        
        # Create data loaders for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
        
        # Initialize model for this fold
        model = MSCNN().to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Train model for this fold
        train_losses, val_losses, test_losses, val_accuracies, test_accuracies, history = train_model(
            model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, device, EPOCHS, fold
        )
        
        # Store fold results
        all_fold_results['fold_histories'].append(history)
        all_fold_results['final_val_accuracies'].append(val_accuracies[-1])
        all_fold_results['final_test_accuracies'].append(test_accuracies[-1])
        all_fold_results['final_val_losses'].append(val_losses[-1])
        all_fold_results['final_test_losses'].append(test_losses[-1])
        
        # Save model for this fold
        model_path = f'models/esc50_mscnn_model_fold{fold+1}_{timestamp}.pth'
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model for fold {fold+1} saved to {model_path}")
        
        # Plot and save results for this fold
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold+1} - Training, Validation, and Test Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title(f'Fold {fold+1} - Validation and Test Accuracy')
        
        plt.subplot(1, 3, 3)
        plt.plot(history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Fold {fold+1} - Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(f'reports/figures/training_results_fold{fold+1}_{timestamp}.png')
        plt.close()
    
    # Calculate and log cross-validation results
    avg_val_accuracy = np.mean(all_fold_results['final_val_accuracies'])
    std_val_accuracy = np.std(all_fold_results['final_val_accuracies'])
    avg_test_accuracy = np.mean(all_fold_results['final_test_accuracies'])
    std_test_accuracy = np.std(all_fold_results['final_test_accuracies'])
    avg_val_loss = np.mean(all_fold_results['final_val_losses'])
    std_val_loss = np.std(all_fold_results['final_val_losses'])
    avg_test_loss = np.mean(all_fold_results['final_test_losses'])
    std_test_loss = np.std(all_fold_results['final_test_losses'])
    
    logging.info("\nCross-Validation Results:")
    logging.info(f"Average Validation Accuracy: {avg_val_accuracy:.2f}% ± {std_val_accuracy:.2f}%")
    logging.info(f"Average Test Accuracy: {avg_test_accuracy:.2f}% ± {std_test_accuracy:.2f}%")
    logging.info(f"Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
    logging.info(f"Average Test Loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
    
    # Save all fold results
    results_path = f'reports/logs/cross_validation_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(all_fold_results, f, indent=4)
    logging.info(f"Cross-validation results saved to {results_path}")
    
    # Plot cross-validation summary
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, N_SPLITS+1), all_fold_results['final_val_accuracies'], label='Validation')
    plt.bar(range(1, N_SPLITS+1), all_fold_results['final_test_accuracies'], label='Test', alpha=0.5)
    plt.axhline(y=avg_val_accuracy, color='r', linestyle='--', label=f'Val Avg: {avg_val_accuracy:.2f}%')
    plt.axhline(y=avg_test_accuracy, color='b', linestyle='--', label=f'Test Avg: {avg_test_accuracy:.2f}%')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title('Cross-Validation Results - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, N_SPLITS+1), all_fold_results['final_val_losses'], label='Validation')
    plt.bar(range(1, N_SPLITS+1), all_fold_results['final_test_losses'], label='Test', alpha=0.5)
    plt.axhline(y=avg_val_loss, color='r', linestyle='--', label=f'Val Avg: {avg_val_loss:.4f}')
    plt.axhline(y=avg_test_loss, color='b', linestyle='--', label=f'Test Avg: {avg_test_loss:.4f}')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.title('Cross-Validation Results - Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'reports/figures/cross_validation_summary_{timestamp}.png')
    plt.close()
    
    logging.info("Cross-validation completed. Results saved to reports directory.")

if __name__ == '__main__':
    main() 