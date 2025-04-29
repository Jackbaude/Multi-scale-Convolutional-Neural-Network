import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score
)
from torch.utils.data import DataLoader
import json
import logging
from datetime import datetime
from esc50_mscnn import (
    ESC50Dataset, MSCNN, download_esc50,
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    BATCH_SIZE, CLASS_NAMES
)

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('reports/analysis', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'reports/analysis/model_analysis_{timestamp}.log'),
        logging.StreamHandler()
    ]
)

def load_model(model_path, device):
    """Load a trained model from disk."""
    model = MSCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    """Evaluate model performance on a dataset."""
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, title, save_path):
    """Plot precision-recall curve for each class."""
    plt.figure(figsize=(10, 8))
    for i in range(len(CLASS_NAMES)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
        plt.plot(recall, precision, label=f'{CLASS_NAMES[i]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_scores, title, save_path):
    """Plot ROC curve for each class."""
    plt.figure(figsize=(10, 8))
    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_metrics(y_true, y_pred, title, save_path):
    """Plot precision, recall, and F1-score for each class."""
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    plt.figure(figsize=(15, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, CLASS_NAMES, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_model(model_path, test_loader, device, fold):
    """Perform comprehensive analysis of a model."""
    logging.info(f"\nAnalyzing model from fold {fold}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Get predictions
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)
    
    # Calculate class distribution
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    
    logging.info("\nClass Distribution in Test Set:")
    for class_idx, count in class_distribution.items():
        logging.info(f"{CLASS_NAMES[class_idx]}: {count} samples")
    
    # Calculate metrics with zero_division parameter
    accuracy = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    logging.info("\nOverall Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    
    # Calculate per-class metrics
    class_metrics = {}
    for class_idx in range(len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_idx]
        if class_idx in class_distribution:
            class_precision = precision_score(y_true == class_idx, y_pred == class_idx, zero_division=0)
            class_recall = recall_score(y_true == class_idx, y_pred == class_idx, zero_division=0)
            class_f1 = f1_score(y_true == class_idx, y_pred == class_idx, zero_division=0)
            
            class_metrics[class_name] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1,
                'support': class_distribution[class_idx]
            }
    
    logging.info("\nPer-Class Metrics:")
    for class_name, metrics in class_metrics.items():
        logging.info(f"\n{class_name}:")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {metrics['f1']:.4f}")
        logging.info(f"  Support: {metrics['support']}")
    
    # Generate classification report with zero_division parameter
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    logging.info("\nDetailed Classification Report:")
    logging.info(report)
    
    # Create plots
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        f'Confusion Matrix - Fold {fold}',
        f'reports/analysis/confusion_matrix_fold{fold}_{timestamp}.png'
    )
    
    plot_precision_recall_curve(
        y_true, y_probs,
        f'Precision-Recall Curves - Fold {fold}',
        f'reports/analysis/precision_recall_fold{fold}_{timestamp}.png'
    )
    
    plot_roc_curve(
        y_true, y_probs,
        f'ROC Curves - Fold {fold}',
        f'reports/analysis/roc_curves_fold{fold}_{timestamp}.png'
    )
    
    plot_class_metrics(
        y_true, y_pred,
        f'Class-wise Metrics - Fold {fold}',
        f'reports/analysis/class_metrics_fold{fold}_{timestamp}.png'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics,
        'class_distribution': class_distribution,
        'classification_report': report
    }

def plot_class_distribution(class_distribution, title, save_path):
    """Plot the distribution of classes in the dataset."""
    plt.figure(figsize=(15, 6))
    classes = [CLASS_NAMES[idx] for idx in class_distribution.keys()]
    counts = list(class_distribution.values())
    
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def find_latest_model_files():
    """Find the most recent model files in the models directory."""
    if not os.path.exists('models'):
        return None
    
    # Get all model files
    model_files = [f for f in os.listdir('models') if f.startswith('esc50_mscnn_model_fold') and f.endswith('.pth')]
    if not model_files:
        return None
    
    # Group files by fold
    fold_files = {}
    for file in model_files:
        # Extract fold number and timestamp
        parts = file.split('_')
        fold = int(parts[3][4:])  # fold1, fold2, etc.
        timestamp = '_'.join(parts[4:]).replace('.pth', '')
        
        # Keep only the latest file for each fold
        if fold not in fold_files or timestamp > fold_files[fold][1]:
            fold_files[fold] = (file, timestamp)
    
    # Sort by fold number and return file paths
    return [os.path.join('models', fold_files[fold][0]) for fold in sorted(fold_files.keys())]

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Find latest model files
    model_files = find_latest_model_files()
    if not model_files:
        logging.error("No model files found in the models directory. Please train the model first.")
        return
    
    logging.info(f"Found {len(model_files)} model files to analyze")
    
    # Download and prepare dataset
    audio_dir, meta_file = download_esc50()
    
    # Create dataset
    dataset = ESC50Dataset(audio_dir, meta_file, transform=None, train=False)
    
    # Initialize k-fold cross-validation
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results for all folds
    all_results = {}
    
    # Analyze each fold
    for fold, (_, test_idx) in enumerate(kfold.split(dataset)):
        # Create test loader for this fold
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
        
        # Get model path for this fold
        if fold < len(model_files):
            model_path = model_files[fold]
            logging.info(f"Analyzing model: {model_path}")
            
            # Analyze model
            results = analyze_model(model_path, test_loader, device, fold+1)
            all_results[f'fold_{fold+1}'] = results
            
            # Plot class distribution for this fold
            plot_class_distribution(
                results['class_distribution'],
                f'Class Distribution - Fold {fold+1}',
                f'reports/analysis/class_distribution_fold{fold+1}_{timestamp}.png'
            )
        else:
            logging.warning(f"No model file found for fold {fold+1}")
    
    # Calculate and log average metrics across folds
    if all_results:
        # Calculate average metrics
        avg_metrics = {
            'accuracy': np.mean([r['accuracy'] for r in all_results.values()]),
            'precision': np.mean([r['precision'] for r in all_results.values()]),
            'recall': np.mean([r['recall'] for r in all_results.values()]),
            'f1': np.mean([r['f1'] for r in all_results.values()])
        }
        
        # Calculate average class metrics
        avg_class_metrics = {}
        for class_name in CLASS_NAMES:
            class_metrics = {
                'precision': np.mean([r['class_metrics'][class_name]['precision'] for r in all_results.values() if class_name in r['class_metrics']]),
                'recall': np.mean([r['class_metrics'][class_name]['recall'] for r in all_results.values() if class_name in r['class_metrics']]),
                'f1': np.mean([r['class_metrics'][class_name]['f1'] for r in all_results.values() if class_name in r['class_metrics']]),
                'support': np.mean([r['class_metrics'][class_name]['support'] for r in all_results.values() if class_name in r['class_metrics']])
            }
            avg_class_metrics[class_name] = class_metrics
        
        logging.info("\nAverage Metrics Across Folds:")
        for metric, value in avg_metrics.items():
            logging.info(f"{metric.capitalize()}: {value:.4f}")
        
        logging.info("\nAverage Per-Class Metrics Across Folds:")
        for class_name, metrics in avg_class_metrics.items():
            logging.info(f"\n{class_name}:")
            logging.info(f"  Precision: {metrics['precision']:.4f}")
            logging.info(f"  Recall: {metrics['recall']:.4f}")
            logging.info(f"  F1-Score: {metrics['f1']:.4f}")
            logging.info(f"  Average Support: {metrics['support']:.1f}")
        
        # Save all results to JSON
        results_path = f'reports/analysis/model_analysis_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump({
                'average_metrics': avg_metrics,
                'average_class_metrics': avg_class_metrics,
                'fold_results': all_results
            }, f, indent=4)
        logging.info(f"Analysis results saved to {results_path}")
    else:
        logging.error("No results were generated. Please check if the model files exist and are valid.")

if __name__ == '__main__':
    main() 