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
        'final_test_losses': []
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
            
        except Exception as e:
            print(f"Error loading {fold_file}: {str(e)}")
            continue
    
    return all_fold_results

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
    
    return plot_dir

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze and graph training results from multiple fold JSON files.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model (e.g., cnn, mscnn)')
    parser.add_argument('--timestamp', type=str, help='Timestamp of the fold files to analyze (e.g., 20250430_205742)')
    parser.add_argument('--fold_files', nargs='+', help='List of specific fold result JSON files to analyze')
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
    
    # Get fold files
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
    
    # Log the files being analyzed
    logger.info("Analyzing the following files:")
    for file in fold_files:
        logger.info(f"  - {file}")
    
    # Load and process fold results
    logger.info(f"Loading results from {len(fold_files)} fold files...")
    all_fold_results = load_fold_results(fold_files)
    
    if not all_fold_results['fold_histories']:
        logger.error("No valid fold results found!")
        return
    
    # Plot individual metrics and get the plot directory
    logger.info("Generating individual metric plots...")
    plot_dir = plot_individual_metrics(all_fold_results, timestamp, args.model_name)
    
    # Plot comprehensive results in the same directory
    logger.info("Generating comprehensive plot...")
    plot_all_folds_results(all_fold_results, timestamp, args.model_name)
    
    # Copy the comprehensive plot to the individual plots directory
    import shutil
    shutil.copy2(
        f'reports/figures/{args.model_name}_all_folds_results_{timestamp}.png',
        f'{plot_dir}/comprehensive_results.png'
    )
    
    logger.info(f"Analysis complete. Results saved with timestamp: {timestamp}")
    logger.info(f"All plots saved in: {plot_dir}/")

if __name__ == '__main__':
    main() 