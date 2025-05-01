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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.setup import setup_directories
from utils.dataset import ESC50Dataset
from utils.metrics import (
    plot_training_results,
    save_fold_results,
    calculate_cross_validation_metrics,
    save_cross_validation_results,
    plot_all_folds_results
)
from models.cnn import ESC50CNN
from utils.augmentations import apply_augmentations

# Constants
BATCH_SIZE = 32
N_SPLITS = 2  # Number of folds for cross-validation

# Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'reports/logs/cnn_training_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        
        # Add tqdm progress bar for training
        train_loop = tqdm(train_loader, desc=f'Fold {fold+1}, Epoch {epoch+1}/{epochs}')
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply data augmentation
            if np.random.random() < 0.5:
                idx = torch.randperm(inputs.size(0))
                inputs, labels = apply_augmentations(inputs, labels, train=True)
                labels = labels.long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item())
            
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
        logger.info(f'Fold {fold+1}, Epoch {epoch+1}/{epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
        logger.info(f'Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses, test_losses, val_accuracies, test_accuracies, history

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
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize model for this fold
        model = ESC50CNN(num_classes=50)
        model = model.to(device)
        
        # Initialize optimizer and loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=5
        )
        
        # Train and evaluate model
        train_losses, val_losses, test_losses, val_accuracies, test_accuracies, history = train_model(
            model, train_loader, val_loader, test_loader,
            criterion, optimizer, scheduler, device, NUM_EPOCHS, fold
        )
        
        # Store fold results
        all_fold_results['fold_histories'].append(history)
        all_fold_results['final_val_accuracies'].append(val_accuracies[-1])
        all_fold_results['final_test_accuracies'].append(test_accuracies[-1])
        all_fold_results['final_val_losses'].append(val_losses[-1])
        all_fold_results['final_test_losses'].append(test_losses[-1])
        
        # Save model and results for this fold
        fold_results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'val_accuracies': val_accuracies,
            'test_accuracies': test_accuracies,
            'history': history
        }
        save_fold_results(model, fold_results, fold, timestamp)
        
        # Plot training results
        plot_training_results(
            train_losses, val_losses, test_losses,
            val_accuracies, test_accuracies, history,
            fold, timestamp, model_name='cnn'
        )
    
    # Calculate and log cross-validation metrics
    cv_metrics = calculate_cross_validation_metrics(all_fold_results)
    
    # Save all cross-validation results
    save_cross_validation_results(all_fold_results, timestamp)
    
    # Plot comprehensive results for all folds
    plot_all_folds_results(all_fold_results, timestamp, model_name='cnn')

if __name__ == '__main__':
    main() 