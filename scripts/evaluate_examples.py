import os
import sys
import torch
import torchaudio
import numpy as np
import argparse
import random
from datetime import datetime
import logging
import sounddevice as sd
import time
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.setup import setup_directories
from utils.dataset import ESC50Dataset
from models.cnn import ESC50CNN
from models.crnn import ESC50CRNN
from models.mscnn import MSCNN
from models.mscrnn import MSCRNN

def load_model(model_name, timestamp, fold):
    """Load a trained model from disk."""
    model_path = f'built_models/esc50_esc50{model_name}_model_fold{fold}_{timestamp}.pth'
    
    # Initialize the appropriate model
    if model_name == 'cnn':
        model = ESC50CNN(num_classes=50)
    elif model_name == 'crnn':
        model = ESC50CRNN(num_classes=50)
    elif model_name == 'mscnn':
        model = MSCNN(num_classes=50)
    elif model_name == 'mscrnn':
        model = MSCRNN(num_classes=50)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Load the model state
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        # Try alternative path pattern
        alt_model_path = f'built_models/esc50_{model_name}_model_fold{fold}_{timestamp}.pth'
        model.load_state_dict(torch.load(alt_model_path))
    
    model.eval()
    return model

def plot_spectrogram(waveform, sample_rate, title):
    """Plot the spectrogram of an audio waveform."""
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    else:
        waveform = waveform.squeeze(0)
    
    # Create spectrogram
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=512,
        power=2.0
    )(waveform)
    
    # Convert to log scale
    spectrogram = torch.log(spectrogram + 1e-9)
    
    # Create figure with plt instead of Figure
    plt.figure(figsize=(12, 6))
    
    # Plot spectrogram
    plt.imshow(spectrogram.numpy(), 
              aspect='auto', 
              origin='lower',
              cmap='viridis')
    
    # Customize plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Time (frames)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    
    # Add colorbar
    plt.colorbar(format='%+2.0f dB')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()

def play_audio(audio_path, title):
    """Play audio file using sounddevice and display spectrogram."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Plot spectrogram
        fig = plot_spectrogram(waveform, sample_rate, title)
        plt.show(block=False)  # Show plot without blocking
        plt.pause(0.1)  # Small pause to ensure plot is displayed
        
        # Convert to mono if needed and ensure proper shape
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)
            
        # Convert to numpy array and ensure proper shape
        audio_data = waveform.numpy()
        if len(audio_data.shape) > 1:
            audio_data = audio_data.squeeze()
            
        # Play the audio
        sd.play(audio_data, sample_rate)
        
        # Wait for either audio to finish or user input
        while sd.get_stream().active:
            if input("Press Enter to skip to next example...") == "":
                sd.stop()
                break
            time.sleep(0.1)  # Small delay to prevent high CPU usage
        
        plt.close()  # Close the plot when done
    except Exception as e:
        print(f"Error playing audio: {e}")
        print("Skipping audio playback...")
        plt.close()  # Ensure plot is closed even if there's an error

def main():
    parser = argparse.ArgumentParser(description='Evaluate examples from a trained model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model (cnn, crnn, mscnn, mscrnn)')
    parser.add_argument('--timestamp', type=str, required=True, help='Timestamp of the model (e.g., 20250430_205742)')
    parser.add_argument('--fold', type=int, required=True, help='Fold number to evaluate')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to evaluate')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Setup directories
    setup_directories()

    # Load model
    logger.info(f"Loading {args.model_name} model from fold {args.fold}...")
    model = load_model(args.model_name, args.timestamp, args.fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load dataset
    dataset = ESC50Dataset('ESC-50/audio', 'ESC-50/meta/esc50.csv', train=True)
    
    # Get category names and mapping
    with open('ESC-50/meta/esc50.csv', 'r') as f:
        # Read all lines and create a mapping of target to category
        lines = f.readlines()
        target_to_category = {}
        category_to_target = {}
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            target = int(parts[2])  # target is in column 2
            category = parts[3]     # category is in column 3
            target_to_category[target] = category
            category_to_target[category] = target
    
    # Find examples for specific categories
    target_categories = ['engine', 'airplane']
    examples = []
    
    for category in target_categories:
        if category in category_to_target:
            target = category_to_target[category]
            # Find all examples of this category
            category_indices = dataset.meta_data[dataset.meta_data['target'] == target].index.tolist()
            if category_indices:
                # Take up to 2 examples per category
                examples.extend(random.sample(category_indices, min(2, len(category_indices))))
    
    logger.info(f"\nEvaluating {len(examples)} examples...")
    for i, idx in enumerate(examples):
        # Get the example
        audio_path = os.path.join(dataset.audio_dir, dataset.meta_data.iloc[idx]['filename'])
        true_label = dataset.meta_data.iloc[idx]['target']
        true_category = target_to_category[true_label]
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = dataset[idx][0].unsqueeze(0).to(device)
            output = model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_category = target_to_category[predicted_label]
            confidence = torch.softmax(output, dim=1)[0][predicted_label].item()
        
        # Print results
        print(f"\nExample {i+1}/{len(examples)}:")
        print(f"Audio file: {os.path.basename(audio_path)}")
        print(f"True category: {true_category}")
        print(f"Predicted category: {predicted_category} (confidence: {confidence:.2%})")
        
        # Play the audio and show spectrogram
        print("Playing audio and showing spectrogram...")
        title = f"True: {true_category} | Predicted: {predicted_category} ({confidence:.2%})"
        play_audio(audio_path, title)

if __name__ == '__main__':
    main() 