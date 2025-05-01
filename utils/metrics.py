import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from datetime import datetime
import torch

def calculate_average_metrics(metrics_list):
    """Calculate average and standard deviation of metrics across folds."""
    metrics = {
        'mean': {},
        'std': {}
    }
    
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        metrics['mean'][key] = np.mean(values)
        metrics['std'][key] = np.std(values)
    
    return metrics

def log_metrics(logger, metrics, prefix=''):
    """Log metrics with mean and standard deviation."""
    for key, value in metrics['mean'].items():
        std = metrics['std'][key]
        logger.info(f'{prefix}{key}: {value:.4f} ± {std:.4f}')

def save_results(y_true, y_pred, class_names, save_path):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.close()
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(f'{save_path}/classification_report.txt', 'w') as f:
        f.write(report)
        
    return cm, report

def plot_training_history(history, save_path):
    """Plot training history including loss and accuracy curves."""
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracy curves
    plt.subplot(1, 3, 2)
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.plot(history['test_accuracy'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_classification_report(y_true, y_pred, class_names, save_path):
    """Save classification report to file."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(save_path, 'w') as f:
        f.write(report)

def plot_training_results(train_losses, val_losses, test_losses, val_accuracies, test_accuracies, history, fold, timestamp, model_name):
    """Plot training results and save to file."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'reports/figures/{model_name}_training_results_fold{fold+1}_{timestamp}.png')
    plt.close()

def save_fold_results(model, fold_results, fold, timestamp):
    """Save model and results for a single fold."""
    # Save model
    model_dir = 'built_models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'esc50_{model.__class__.__name__.lower()}_model_fold{fold+1}_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model for fold {fold+1} saved to {model_path}")
    
    # Save fold results
    results_path = f'reports/logs/fold{fold+1}_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(fold_results, f, indent=4)
    logging.info(f"Results for fold {fold+1} saved to {results_path}")

def calculate_cross_validation_metrics(all_fold_results):
    """Calculate and log cross-validation metrics."""
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
    
    return {
        'avg_val_accuracy': avg_val_accuracy,
        'std_val_accuracy': std_val_accuracy,
        'avg_test_accuracy': avg_test_accuracy,
        'std_test_accuracy': std_test_accuracy,
        'avg_val_loss': avg_val_loss,
        'std_val_loss': std_val_loss,
        'avg_test_loss': avg_test_loss,
        'std_test_loss': std_test_loss
    }

def save_cross_validation_results(all_fold_results, timestamp):
    """Save all cross-validation results."""
    results_path = f'reports/logs/cross_validation_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(all_fold_results, f, indent=4)
    logging.info(f"Cross-validation results saved to {results_path}")

def plot_all_folds_results(all_fold_results, timestamp, model_name):
    """Plot comprehensive results for all folds including losses, accuracies, and learning rates."""
    n_folds = len(all_fold_results['fold_histories'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot training losses for all folds
    for fold in range(n_folds):
        history = all_fold_results['fold_histories'][fold]
        ax1.plot(history['train_loss'], label=f'Fold {fold+1}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Across Folds')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation and test losses for all folds
    for fold in range(n_folds):
        history = all_fold_results['fold_histories'][fold]
        ax2.plot(history['val_loss'], '--', label=f'Val Fold {fold+1}')
        ax2.plot(history['test_loss'], ':', label=f'Test Fold {fold+1}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation and Test Loss Across Folds')
    ax2.legend()
    ax2.grid(True)
    
    # Plot validation and test accuracies for all folds
    for fold in range(n_folds):
        history = all_fold_results['fold_histories'][fold]
        ax3.plot(history['val_accuracy'], '--', label=f'Val Fold {fold+1}')
        ax3.plot(history['test_accuracy'], ':', label=f'Test Fold {fold+1}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Validation and Test Accuracy Across Folds')
    ax3.legend()
    ax3.grid(True)
    
    # Plot learning rates for all folds
    for fold in range(n_folds):
        history = all_fold_results['fold_histories'][fold]
        ax4.plot(history['learning_rate'], label=f'Fold {fold+1}')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule Across Folds')
    ax4.legend()
    ax4.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'reports/figures/{model_name}_all_folds_results_{timestamp}.png')
    plt.close() 