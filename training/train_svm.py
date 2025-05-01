import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import logging
from datetime import datetime
from tqdm import tqdm
import json
import torch.nn as nn
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.setup import setup_directories
from utils.dataset import ESC50Dataset
from utils.metrics import (
    plot_training_results,
    save_fold_results,
    calculate_cross_validation_metrics,
    save_cross_validation_results
)
from models.svm import ESC50SVM

# Constants
SAMPLE_RATE = 44100
N_MELS = 40  # As per paper
N_FFT = 1024  # As per paper
HOP_LENGTH = 512  # 50% overlap as per paper
BATCH_SIZE = 64
N_SPLITS = 5  # Number of folds for cross-validation

# SVM parameters
C = 1.0  # L2 regularization as per paper

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'reports/logs/svm_training_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_model(model, train_dataset, val_dataset, test_dataset, device, fold):
    # Train model
    logger.info(f'Training SVM for fold {fold+1}')
    train_losses = model.fit(train_dataset, device)
    
    # Evaluate on validation set
    logger.info(f'Evaluating on validation set for fold {fold+1}')
    val_predictions = model.predict(val_dataset, device)
    val_labels = np.array([val_dataset.dataset.meta_data.iloc[idx]['target'] for idx in val_dataset.indices])
    val_accuracy = np.mean(val_predictions == val_labels)
    
    # Calculate validation loss
    val_features = []
    for idx in val_dataset.indices:
        audio_path = os.path.join(val_dataset.dataset.audio_dir, val_dataset.dataset.meta_data.iloc[idx]['filename'])
        features = model.extract_features(audio_path, device=device)
        val_features.append(features)
    val_features = np.array(val_features)
    val_features = model.scaler.transform(val_features)
    
    # Calculate hinge loss
    decision_values = model.svm.decision_function(val_features)
    val_loss = np.mean(np.maximum(0, 1 - decision_values[np.arange(len(val_labels)), val_labels]))
    
    # Evaluate on test set
    logger.info(f'Evaluating on test set for fold {fold+1}')
    test_predictions = model.predict(test_dataset, device)
    test_labels = np.array([test_dataset.dataset.meta_data.iloc[idx]['target'] for idx in test_dataset.indices])
    test_accuracy = np.mean(test_predictions == test_labels)
    
    # Calculate test loss
    test_features = []
    for idx in test_dataset.indices:
        audio_path = os.path.join(test_dataset.dataset.audio_dir, test_dataset.dataset.meta_data.iloc[idx]['filename'])
        features = model.extract_features(audio_path, device=device)
        test_features.append(features)
    test_features = np.array(test_features)
    test_features = model.scaler.transform(test_features)
    
    # Calculate hinge loss
    decision_values = model.svm.decision_function(test_features)
    test_loss = np.mean(np.maximum(0, 1 - decision_values[np.arange(len(test_labels)), test_labels]))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (Fold {fold+1})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(['Validation', 'Test'], [val_accuracy, test_accuracy])
    plt.ylabel('Accuracy (%)')
    plt.title(f'Validation and Test Accuracy (Fold {fold+1})')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'reports/figures/svm_training_fold{fold+1}_{timestamp}.png')
    plt.close()
    
    # Store results
    results = {
        'train_losses': train_losses,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'val_predictions': val_predictions.tolist(),
        'test_predictions': test_predictions.tolist(),
        'val_labels': val_labels.tolist(),
        'test_labels': test_labels.tolist()
    }
    
    return results

def main():
    # Setup directories
    setup_directories()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize dataset
    dataset = ESC50Dataset('ESC-50/audio', 'ESC-50/meta/esc50.csv', train=True)
    
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
        logger.info(f'\nStarting Fold {fold + 1}/{N_SPLITS}')
        
        # Split into train, validation, and test sets
        train_size = int(0.8 * len(train_idx))
        train_idx, test_idx = train_idx[:train_size], train_idx[train_size:]
        
        # Create datasets for this fold
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        
        # Initialize model for this fold
        model = ESC50SVM(C=C)
        model = model.to(device)
        
        # Train and evaluate model
        fold_results = train_model(model, train_dataset, val_dataset, test_dataset, device, fold)
        
        # Store fold results
        all_fold_results['fold_histories'].append(fold_results)
        all_fold_results['final_val_accuracies'].append(fold_results['val_accuracy'])
        all_fold_results['final_test_accuracies'].append(fold_results['test_accuracy'])
        all_fold_results['final_val_losses'].append(fold_results['val_loss'])
        all_fold_results['final_test_losses'].append(fold_results['test_loss'])
        
        # Save model and results for this fold
        model_path = f'models/esc50_svm_model_fold{fold+1}_{timestamp}.pth'
        results_path = f'reports/logs/fold{fold+1}_results_{timestamp}.json'
        torch.save(model.state_dict(), model_path)
        with open(results_path, 'w') as f:
            json.dump(fold_results, f)
        logger.info(f'Model for fold {fold+1} saved to {model_path}')
        logger.info(f'Results for fold {fold+1} saved to {results_path}')
    
    # Calculate and log cross-validation metrics
    cv_metrics = calculate_cross_validation_metrics(all_fold_results)
    
    # Save all cross-validation results
    save_cross_validation_results(all_fold_results, timestamp)
    
    # Plot final cross-validation results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.boxplot([all_fold_results['final_val_accuracies'], all_fold_results['final_test_accuracies']], 
                labels=['Validation', 'Test'])
    plt.ylabel('Accuracy (%)')
    plt.title('Cross-Validation Accuracy Distribution')
    plt.ylim(0, 100)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([all_fold_results['final_val_losses'], all_fold_results['final_test_losses']], 
                labels=['Validation', 'Test'])
    plt.ylabel('Loss')
    plt.title('Cross-Validation Loss Distribution')
    
    plt.tight_layout()
    plt.savefig(f'reports/figures/svm_cross_validation_{timestamp}.png')
    plt.close()

if __name__ == '__main__':
    main() 