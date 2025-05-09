import os
import sys
import json
import argparse
import glob
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import plot_all_folds_results

def find_fold_files(timestamp):
    """Find all fold result files with the given timestamp."""
    pattern = f'reports/logs/fold*_results_{timestamp}.json'
    return sorted(glob.glob(pattern))

def load_fold_results(fold_files):
    """Load results from multiple fold JSON files."""
    all_fold_results = {
        'fold_histories': [],
        'final_val_accuracies': [],
        'final_test_accuracies': [],
        'final_val_losses': [],
        'final_test_losses': [],
        'y_true': [],
        'y_pred': [],
        'y_pred_proba': []
    }
    
    for fold_file in fold_files:
        try:
            with open(fold_file, 'r') as f:
                fold_data = json.load(f)
                
            # Extract the history from the fold results
            history = fold_data['history']
            all_fold_results['fold_histories'].append(history)
            
            # Extract final metrics
            all_fold_results['final_val_accuracies'].append(history['val_accuracy'][-1])
            all_fold_results['final_test_accuracies'].append(history['test_accuracy'][-1])
            all_fold_results['final_val_losses'].append(history['val_loss'][-1])
            all_fold_results['final_test_losses'].append(history['test_loss'][-1])
            
            # Extract predictions if available
            if 'predictions' in fold_data:
                all_fold_results['y_true'].extend(fold_data['predictions']['y_true'])
                all_fold_results['y_pred'].extend(fold_data['predictions']['y_pred'])
                all_fold_results['y_pred_proba'].extend(fold_data['predictions']['y_pred_proba'])
            
        except Exception as e:
            print(f"Error loading {fold_file}: {str(e)}")
            continue
    
    return all_fold_results

def plot_roc_curves(y_true, y_pred_proba, class_names, save_path):
    """Plot ROC curves for each class and save to file."""
    n_classes = len(class_names)
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(f'{save_path}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def save_metrics_to_csv(y_true, y_pred, y_pred_proba, class_names, save_path):
    """Save detailed metrics to CSV file."""
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(report).transpose()
    
    # Add ROC AUC scores
    roc_auc_scores = []
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
        roc_auc_scores.append(auc(fpr, tpr))
    
    metrics_df['roc_auc'] = roc_auc_scores + [np.mean(roc_auc_scores)]  # Add mean ROC AUC
    
    # Save to CSV
    metrics_df.to_csv(f'{save_path}/detailed_metrics.csv')
    
    return metrics_df

def plot_individual_metrics(all_fold_results, timestamp, model_name):
    """Create individual plots for each metric."""
    n_folds = len(all_fold_results['fold_histories'])
    
    # Set style for better visualization
    sns.set_style("whitegrid")
    sns.set_context("poster")
    
    # Create directory for individual plots
    plot_dir = f'reports/figures/{model_name}_individual_{timestamp}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Define colors for folds
    colors = sns.color_palette("husl", n_folds)
    
    # Plot training losses
    plt.figure(figsize=(20, 12), dpi=300)
    for fold in range(n_folds):
        history = all_fold_results['fold_histories'][fold]
        plt.plot(history['train_loss'], label=f'Fold {fold+1}', linewidth=3, color=colors[fold])
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f'{model_name.upper()} Training Loss Across Folds', fontsize=24, pad=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot validation and test losses
    plt.figure(figsize=(20, 12), dpi=300)
    for fold in range(n_folds):
        history = all_fold_results['fold_histories'][fold]
        plt.plot(history['val_loss'], '--', label=f'Val Fold {fold+1}', linewidth=3, color=colors[fold])
        plt.plot(history['test_loss'], ':', label=f'Test Fold {fold+1}', linewidth=3, color=colors[fold])

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f'{model_name.upper()} Validation and Test Loss Across Folds', fontsize=24, pad=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/val_test_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot validation and test accuracies
    plt.figure(figsize=(20, 12), dpi=300)
    for fold in range(n_folds):
        history = all_fold_results['fold_histories'][fold]
        plt.plot(history['val_accuracy'], '--', label=f'Val Fold {fold+1}', linewidth=3, color=colors[fold])
        plt.plot(history['test_accuracy'], ':', label=f'Test Fold {fold+1}', linewidth=3, color=colors[fold])

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.title(f'{model_name.upper()} Validation and Test Accuracy Across Folds', fontsize=24, pad=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/val_test_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot learning rates
    plt.figure(figsize=(20, 12), dpi=300)
    for fold in range(n_folds):
        history = all_fold_results['fold_histories'][fold]
        plt.plot(history['learning_rate'], label=f'Fold {fold+1}', linewidth=3, color=colors[fold])
        
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Learning Rate', fontsize=20)
    plt.title(f'{model_name.upper()} Learning Rate Schedule Across Folds', fontsize=24, pad=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # If we have predictions, plot ROC curves and save metrics
    if all_fold_results['y_true'] and all_fold_results['y_pred'] and all_fold_results['y_pred_proba']:
        # Convert predictions to numpy arrays
        y_true = np.array(all_fold_results['y_true'])
        y_pred = np.array(all_fold_results['y_pred'])
        y_pred_proba = np.array(all_fold_results['y_pred_proba'])
        
        # Get class names (assuming they're 0-indexed)
        class_names = [f'Class_{i}' for i in range(y_pred_proba.shape[1])]
        
        # Plot ROC curves
        plot_roc_curves(y_true, y_pred_proba, class_names, plot_dir)
        
        # Save detailed metrics to CSV
        save_metrics_to_csv(y_true, y_pred, y_pred_proba, class_names, plot_dir)
    
    return plot_dir

def compare_models(all_model_results, timestamp):
    """Create comparison plots for multiple models."""
    # Set style for better visualization
    sns.set_style("whitegrid")
    sns.set_context("poster")
    
    # Create directory for comparison plots
    plot_dir = f'reports/figures/model_comparison_{timestamp}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Define colors for models
    colors = sns.color_palette("husl", len(all_model_results))
    
    # Plot average validation accuracy across folds for each model
    plt.figure(figsize=(20, 12), dpi=300)
    for idx, (model_name, results) in enumerate(all_model_results.items()):
        avg_val_acc = np.mean(results['final_val_accuracies'])
        std_val_acc = np.std(results['final_val_accuracies'])
        plt.errorbar(idx, avg_val_acc, yerr=std_val_acc, fmt='o', 
                    label=f'{model_name.upper()}', color=colors[idx],
                    markersize=10, capsize=10, capthick=2)
    
    plt.xlabel('Model', fontsize=20)
    plt.ylabel('Average Validation Accuracy (%)', fontsize=20)
    plt.title('Model Comparison: Average Validation Accuracy', fontsize=24, pad=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/validation_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot average test accuracy across folds for each model
    plt.figure(figsize=(20, 12), dpi=300)
    for idx, (model_name, results) in enumerate(all_model_results.items()):
        avg_test_acc = np.mean(results['final_test_accuracies'])
        std_test_acc = np.std(results['final_test_accuracies'])
        plt.errorbar(idx, avg_test_acc, yerr=std_test_acc, fmt='o', 
                    label=f'{model_name.upper()}', color=colors[idx],
                    markersize=10, capsize=10, capthick=2)
    
    plt.xlabel('Model', fontsize=20)
    plt.ylabel('Average Test Accuracy (%)', fontsize=20)
    plt.title('Model Comparison: Average Test Accuracy', fontsize=24, pad=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot learning curves for all models
    plt.figure(figsize=(20, 12), dpi=300)
    for idx, (model_name, results) in enumerate(all_model_results.items()):
        for fold in range(len(results['fold_histories'])):
            history = results['fold_histories'][fold]
            plt.plot(history['val_accuracy'], '--', 
                    label=f'{model_name.upper()} Fold {fold+1}', 
                    color=colors[idx], alpha=0.5)
    
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Validation Accuracy (%)', fontsize=20)
    plt.title('Model Comparison: Validation Accuracy Learning Curves', fontsize=24, pad=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_dir

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze and graph training results from multiple fold JSON files.')
    parser.add_argument('--model_name', type=str, help='Name of the model (e.g., cnn, mscnn)')
    parser.add_argument('--timestamp', type=str, help='Timestamp of the fold files to analyze (e.g., 20250430_205742)')
    parser.add_argument('--fold_files', nargs='+', help='List of specific fold result JSON files to analyze')
    parser.add_argument('--compare_models', action='store_true', help='Compare multiple models')
    args = parser.parse_args()
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'reports/logs/analysis_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    if args.compare_models:
        # Define all model files to compare
        model_files = {
            'esc50cnn': [
                'reports/logs/fold1_results_20250501_200550.json',
                'reports/logs/fold2_results_20250501_200550.json'
            ],
            'esc50crnn': [
                'reports/logs/fold1_results_20250501_215803.json',
                'reports/logs/fold2_results_20250501_215803.json'
            ],
            'mscnn': [
                'reports/logs/fold1_results_20250501_203451.json',
                'reports/logs/fold2_results_20250501_203451.json'
            ],
            'mscrnn': [
                'reports/logs/fold1_results_20250501_211425.json',
                'reports/logs/fold2_results_20250501_211425.json'
            ]
        }
        
        all_model_results = {}
        for model_name, fold_files in model_files.items():
            logger.info(f"Loading results for {model_name}...")
            all_model_results[model_name] = load_fold_results(fold_files)
        
        logger.info("Generating comparison plots...")
        plot_dir = compare_models(all_model_results, timestamp)
        logger.info(f"Comparison analysis complete. Results saved in: {plot_dir}/")
        
    else:
        # Original single model analysis code
        if args.fold_files:
            fold_files = args.fold_files
        elif args.timestamp:
            fold_files = find_fold_files(args.timestamp)
            if not fold_files:
                logger.error(f"No fold files found with timestamp {args.timestamp}")
                return
        else:
            logger.error("Either --timestamp or --fold_files must be provided")
            return
        
        logger.info("Analyzing the following files:")
        for file in fold_files:
            logger.info(f"  - {file}")
        
        all_fold_results = load_fold_results(fold_files)
        
        if not all_fold_results['fold_histories']:
            logger.error("No valid fold results found!")
            return
        
        plot_dir = plot_individual_metrics(all_fold_results, timestamp, args.model_name)
        plot_all_folds_results(all_fold_results, timestamp, args.model_name)
        
        shutil.copy2(
            f'reports/figures/{args.model_name}_all_folds_results_{timestamp}.png',
            f'{plot_dir}/comprehensive_results.png'
        )
        
        logger.info(f"Analysis complete. Results saved with timestamp: {timestamp}")
        logger.info(f"All plots saved in: {plot_dir}/")

if __name__ == '__main__':
    main() 