import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import sounddevice as sd
import time
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
        logging.FileHandler(f'reports/logs/evaluation_{timestamp}.log'),
        logging.StreamHandler()
    ]
)

# Constants
SAMPLE_RATE = 44100
N_MELS = 56
N_FFT = 2048
HOP_LENGTH = 512

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
        self.classifier = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        
        avg = self.global_avg(x).squeeze(-1).squeeze(-1)
        max_ = self.global_max(x).squeeze(-1).squeeze(-1)
        x = torch.cat([avg, max_], dim=1)
        x = self.dropout(x)
        out = self.classifier(x)
        return out

def preprocess_audio(audio_path):
    # Load audio
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
    
    return mel_spec, waveform.squeeze().numpy()

def predict_sound(model, audio_path, device):
    # Preprocess audio
    mel_spec, waveform = preprocess_audio(audio_path)
    
    # Move to device and add batch dimension
    mel_spec = mel_spec.to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(mel_spec)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    # Get class names
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence = confidence.item()
    
    return predicted_class, confidence, waveform

def play_audio(waveform, sample_rate=SAMPLE_RATE):
    sd.play(waveform, sample_rate)
    sd.wait()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load model
    model = MSCNN().to(device)
    model.load_state_dict(torch.load('esc50_mscnn_model.pth', map_location=device))
    model.eval()
    logging.info("Model loaded successfully")
    
    # Get test files from ESC-50 dataset
    audio_dir = "ESC-50/audio"
    meta_file = "ESC-50/meta/esc50.csv"
    
    if not os.path.exists(audio_dir):
        logging.error("ESC-50 dataset not found. Please run the training script first.")
        return
    
    # Load metadata
    meta_data = pd.read_csv(meta_file)
    
    # Select some test examples (one from each category)
    test_examples = meta_data.groupby('category').first().reset_index()
    
    # Store evaluation results
    results = {
        'examples': [],
        'total_correct': 0,
        'total_examples': len(test_examples)
    }
    
    # Evaluate each example
    for idx, row in test_examples.iterrows():
        audio_path = os.path.join(audio_dir, row['filename'])
        true_class = CLASS_NAMES[row['target']]
        
        logging.info(f"\nExample {idx + 1}: {true_class}")
        logging.info(f"Audio file: {audio_path}")
        
        # Get prediction
        predicted_class, confidence, waveform = predict_sound(model, audio_path, device)
        
        # Check if prediction is correct
        is_correct = predicted_class == true_class
        if is_correct:
            results['total_correct'] += 1
        
        # Store example results
        example_result = {
            'example_id': idx + 1,
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_correct': is_correct
        }
        results['examples'].append(example_result)
        
        # Log results
        logging.info(f"True class: {true_class}")
        logging.info(f"Predicted class: {predicted_class}")
        logging.info(f"Confidence: {confidence:.2%}")
        logging.info(f"Correct: {is_correct}")
        
        # Play audio
        logging.info("Playing audio...")
        play_audio(waveform)
        time.sleep(1)  # Small delay between samples
    
    # Calculate and log overall accuracy
    accuracy = results['total_correct'] / results['total_examples']
    logging.info(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Save results to JSON
    results_path = f'reports/logs/evaluation_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    main() 