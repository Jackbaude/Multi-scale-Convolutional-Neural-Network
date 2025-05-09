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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.setup import setup_directories
from utils.dataset import ESC50Dataset
from utils.metrics import (
    plot_training_results,
    save_fold_results,
    save_cross_validation_results,
    calculate_cross_validation_metrics,
    plot_all_folds_results
)
from models.mscrnn import MSCRNN
from utils.augmentations import apply_augmentations

# Constants
AUDIO_DIR = 'ESC-50/audio'
META_FILE = 'ESC-50/meta/esc50.csv'
NUM_CLASSES = 50
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.5
N_SPLITS = 2

# Setup directories first
setup_directories()

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'reports/logs/mscrnn_training_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_fold_datasets(fold, n_splits):
    """Create train, validation, and test datasets for a specific fold."""
    # Initialize dataset
    dataset = ESC50Dataset(AUDIO_DIR, META_FILE, train=True)
    
    # Create k-fold splitter
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # Get indices for this fold
    all_indices = np.arange(len(dataset))
    for i, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
        if i == fold:
            break
    
    # Split training set into train and test
    train_size = int(TRAIN_TEST_SPLIT * len(train_idx))
    train_idx, test_idx = train_idx[:train_size], train_idx[train_size:]
    
    # Create datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    return train_dataset, val_dataset, test_dataset

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, device, epochs, fold):
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    test_losses = []
    val_accuracies = []
    test_accuracies = []
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'val_accuracy': [],
        'test_accuracy': [],
        'learning_rate': []
    }
    
    # Initialize mixed precision scaler with new syntax
    scaler = torch.amp.GradScaler('cuda')
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc=f'Fold {fold+1}', position=0)
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        
        # Create progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Training', leave=False, position=1)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply augmentations during training
            inputs, labels = apply_augmentations(inputs, labels, train=True)
            labels = labels.long()  # Convert labels to Long type after augmentations
            
            # Clear CUDA cache periodically
            if train_pbar.n % 10 == 0:
                torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            
            # Use mixed precision training with new syntax
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', leave=False, position=1)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Test evaluation
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Testing', leave=False, position=1)
            for inputs, labels in test_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        test_loss = test_loss / len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        test_accuracy = 100 * test_correct / test_total
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        history['val_accuracy'].append(val_accuracy)
        history['test_accuracy'].append(test_accuracy)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'test_loss': f'{test_loss:.4f}',
            'val_acc': f'{val_accuracy:.2f}%',
            'test_acc': f'{test_accuracy:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Log metrics
        logger.info(f'Fold {fold+1}, Epoch {epoch+1}/{epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
        logger.info(f'Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Clear CUDA cache after each epoch
        torch.cuda.empty_cache()
    
    # Final evaluation on test set
    logger.info('Performing final evaluation on test set...')
    model.eval()
    final_true_labels = []
    final_pred_probas = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Final Evaluation', leave=False)
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Store predictions
            final_true_labels.extend(labels.cpu().numpy())
            final_pred_probas.extend(outputs.softmax(dim=1).cpu().numpy())
    
    # Convert to numpy arrays
    final_true_labels = np.array(final_true_labels)
    final_pred_probas = np.array(final_pred_probas)
    
    # Add final predictions to history (convert numpy arrays to lists for JSON serialization)
    history['final_predictions'] = {
        'y_true': final_true_labels.tolist(),
        'y_pred_proba': final_pred_probas.tolist(),
        'fold': fold,
        'final_accuracy': test_accuracies[-1],
        'final_loss': test_losses[-1]
    }
    
    logger.info(f'Final evaluation complete. Collected {len(final_true_labels)} predictions.')
    
    return train_losses, val_losses, test_losses, val_accuracies, test_accuracies, history

def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Enable memory optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Log hyperparameters
    logger.info('Hyperparameters:')
    logger.info(f'Batch Size: {BATCH_SIZE}')
    logger.info(f'Number of Epochs: {NUM_EPOCHS}')
    logger.info(f'Learning Rate: {LEARNING_RATE}')
    logger.info(f'Hidden Size: {HIDDEN_SIZE}')
    logger.info(f'Number of Layers: {NUM_LAYERS}')
    logger.info(f'Dropout: {DROPOUT}')
    logger.info(f'Number of Splits: {N_SPLITS}')
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Initialize cross-validation results storage
    all_fold_results = {
        'fold_histories': [],
        'final_val_accuracies': [],
        'final_test_accuracies': [],
        'final_val_losses': [],
        'final_test_losses': []
    }
    
    # Perform k-fold cross-validation
    for fold in range(N_SPLITS):
        logger.info(f'\nStarting Fold {fold+1}/{N_SPLITS}')
        
        # Create datasets for this fold
        train_dataset, val_dataset, test_dataset = create_fold_datasets(fold, N_SPLITS)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS
        )
        
        # Initialize model for this fold
        model = MSCRNN(
            num_classes=NUM_CLASSES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
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
            fold, timestamp, model_name='mscrnn'
        )
        
        # Clear CUDA cache after each fold
        torch.cuda.empty_cache()
    
    # Calculate and log cross-validation metrics
    cv_metrics = calculate_cross_validation_metrics(all_fold_results)
    
    # Save all cross-validation results
    save_cross_validation_results(all_fold_results, timestamp)
    
    # Plot comprehensive results for all folds
    plot_all_folds_results(all_fold_results, timestamp, model_name='mscrnn')

if __name__ == '__main__':
    main() 