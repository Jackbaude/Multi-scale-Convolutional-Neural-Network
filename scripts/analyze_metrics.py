import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import glob
from datetime import datetime

# Set global plotting parameters for poster presentation
plt.rcParams.update({
    'font.size': 32,
    'axes.labelsize': 36,
    'axes.titlesize': 40,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 32,
    'figure.titlesize': 44,
    'lines.linewidth': 3,
    'axes.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.5
})

# ESC-50 class names
ESC50_CLASSES = [
    'Dog', 'Rooster', 'Pig', 'Cow', 'Frog', 'Cat', 'Hen', 'Insects', 'Sheep', 'Crow',
    'Rain', 'Sea waves', 'Crackling fire', 'Crickets', 'Chirping birds', 'Water drops',
    'Wind', 'Pouring water', 'Toilet flush', 'Thunderstorm', 'Crying baby', 'Sneezing',
    'Clapping', 'Breathing', 'Coughing', 'Footsteps', 'Laughing', 'Brushing teeth',
    'Snoring', 'Drinking/sipping', 'Door knock', 'Mouse click', 'Keyboard typing',
    'Door, wood creaks', 'Can opening', 'Washing machine', 'Vacuum cleaner', 'Clock alarm',
    'Clock tick', 'Glass breaking', 'Helicopter', 'Chainsaw', 'Siren', 'Car horn',
    'Engine', 'Train', 'Church bells', 'Airplane', 'Fireworks', 'Hand saw'
]

def load_fold_results(timestamp):
    """Load results from all fold files with the given timestamp."""
    pattern = f'reports/logs/fold*_results_{timestamp}.json'
    fold_files = sorted(glob.glob(pattern))
    
    all_fold_results = {
        'y_true': [],
        'y_pred_proba': [],
        'history': []
    }
    
    for fold_file in fold_files:
        with open(fold_file, 'r') as f:
            fold_data = json.load(f)
            if 'history' in fold_data:
                all_fold_results['history'].append(fold_data['history'])
                if 'final_predictions' in fold_data['history']:
                    predictions = fold_data['history']['final_predictions']
                    all_fold_results['y_true'].extend(predictions['y_true'])
                    all_fold_results['y_pred_proba'].extend(predictions['y_pred_proba'])
    
    return all_fold_results

def plot_comprehensive_metrics(timestamps, model_names, save_path):
    """Create separate plots for each metric across all models."""
    # Define metrics to plot with their possible alternative names
    metrics = [
        ('train_loss', 'Training Loss'),
        ('val_loss', 'Validation Loss'),
        ('test_loss', 'Test Loss'),
        ('train_accuracy', 'Training Accuracy'),
        ('val_accuracy', 'Validation Accuracy'),
        ('test_accuracy', 'Test Accuracy'),
        ('learning_rate', 'Learning Rate')
    ]
    
    # Use a distinct color for each model with higher contrast
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create a directory for metric plots
    metrics_dir = os.path.join(save_path, 'individual_metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Print available metrics for debugging
    print("\nAvailable metrics in history:")
    for timestamp, model_name in zip(timestamps, model_names):
        results = load_fold_results(timestamp)
        if results['history']:
            print(f"\n{model_name} metrics:")
            for key in results['history'][0].keys():
                print(f"  - {key}")
                if key == 'test_accuracy':
                    print(f"    Values: {results['history'][0][key]}")
    
    # Plot each metric
    for metric_name, title in metrics:
        plt.figure(figsize=(24, 16))
        has_data = False
        
        # Plot each model's data
        for model_idx, (timestamp, model_name, color) in enumerate(zip(timestamps, model_names, colors)):
            results = load_fold_results(timestamp)
            
            # Collect all values for this metric across folds
            all_values = []
            for history in results['history']:
                if metric_name in history:
                    values = history[metric_name]
                    if isinstance(values, list):
                        all_values.append(values)
                        if metric_name == 'test_accuracy':
                            print(f"\n{model_name} {metric_name} values:")
                            print(f"  Fold {len(all_values)}: {values}")
            
            if all_values:
                has_data = True
                # Plot individual folds with low alpha
                for fold_idx, values in enumerate(all_values):
                    epochs = range(1, len(values) + 1)
                    plt.plot(epochs, values, 
                            color=color, alpha=0.2,
                            linestyle='--', linewidth=2)
                
                # Calculate and plot mean with solid line
                mean_values = np.mean(all_values, axis=0)
                epochs = range(1, len(mean_values) + 1)
                plt.plot(epochs, mean_values, 
                        color=color, linewidth=4,
                        label=f'{model_name}')
                
                if metric_name == 'test_accuracy':
                    print(f"\n{model_name} mean {metric_name}:")
                    print(f"  Values: {mean_values}")
        
        if has_data:
            # Customize plot
            plt.title(title, fontsize=44, pad=30)
            plt.xlabel('Epoch', fontsize=36, labelpad=20)
            plt.ylabel(title, fontsize=36, labelpad=20)
            
            # Add grid with better visibility
            plt.grid(True, which='major', alpha=0.4, linewidth=1.5)
            plt.grid(True, which='minor', alpha=0.2, linewidth=1)
            
            # Add legend with better positioning and formatting
            plt.legend(fontsize=32, loc='center right', 
                      bbox_to_anchor=(1.15, 0.5),
                      frameon=True, framealpha=0.9,
                      edgecolor='black')
            
            # Adjust y-axis limits for accuracy plots
            if 'accuracy' in metric_name:
                plt.ylim([0, 1.05])
            
            # Make ticks more visible
            plt.tick_params(axis='both', which='major', length=10, width=2)
            plt.tick_params(axis='both', which='minor', length=5, width=1)
            
            # Adjust layout and save with high DPI
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, f'{metric_name}.png'), 
                       dpi=600, bbox_inches='tight')
        plt.close()
        
    # Create a combined figure with all metrics
    fig, axes = plt.subplots(4, 2, figsize=(40, 60))
    axes = axes.flatten()
    
    for idx, (metric_name, title) in enumerate(metrics):
        if idx < len(axes):
            metric_file = os.path.join(metrics_dir, f'{metric_name}.png')
            if os.path.exists(metric_file):
                img = plt.imread(metric_file)
                axes[idx].imshow(img)
                axes[idx].axis('off')
                axes[idx].set_title(title, fontsize=44, pad=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'all_metrics_combined.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()

def plot_comparative_confusion_matrices(timestamps, model_names, save_path):
    """Plot individual confusion matrices for each model."""
    # Set color palette for better visibility
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    for timestamp, model_name in zip(timestamps, model_names):
        results = load_fold_results(timestamp)
        y_true = np.array(results['y_true'])
        y_pred_proba = np.array(results['y_pred_proba'])
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure with larger size for better visibility
        plt.figure(figsize=(50, 40))
        
        # Plot confusion matrix with larger annotations
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap,
                    xticklabels=ESC50_CLASSES, yticklabels=ESC50_CLASSES,
                    annot_kws={'size': 20})
        
        plt.title(f'{model_name} Confusion Matrix', fontsize=44, pad=30)
        plt.xlabel('Predicted Label', fontsize=36, labelpad=20)
        plt.ylabel('True Label', fontsize=36, labelpad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=24)
        plt.yticks(fontsize=24)
        
        # Add colorbar with larger font
        cbar = plt.gca().collections[0].colorbar
        cbar.ax.tick_params(labelsize=24)
        
        # Adjust layout and save with high DPI
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{model_name.lower()}_confusion_matrix.png'), 
                    dpi=600, bbox_inches='tight')
        plt.close()
        
        # Save the raw confusion matrix values
        np.save(os.path.join(save_path, f'{model_name.lower()}_confusion_matrix_raw.npy'), cm)
        np.save(os.path.join(save_path, f'{model_name.lower()}_confusion_matrix_norm.npy'), cm_norm)

def plot_comparative_roc_curves(timestamps, model_names, save_path):
    """Plot ROC curves for multiple models on the same plot."""
    plt.figure(figsize=(30, 24))
    
    # Use a more distinct color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    mean_aucs = []
    for timestamp, model_name, color in zip(timestamps, model_names, colors):
        results = load_fold_results(timestamp)
        y_true = np.array(results['y_true'])
        y_pred_proba = np.array(results['y_pred_proba'])
        
        # Calculate mean ROC curve
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        for i in range(len(ESC50_CLASSES)):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        mean_aucs.append(mean_auc)
        
        # Plot with thicker lines and larger markers
        plt.plot(mean_fpr, mean_tpr, color=color, lw=4,
                 label=f'{model_name} (AUC = {mean_auc:.3f})',
                 marker='o', markersize=8, markevery=10)
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=3)
    
    # Set axis limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=32, labelpad=20)
    plt.ylabel('True Positive Rate', fontsize=32, labelpad=20)
    plt.title('Comparative ROC Curves', fontsize=36, pad=30)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=28, frameon=True, framealpha=0.9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{save_path}/comparative_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_aucs

def plot_combined_confusion_matrix(timestamps, model_names, save_path):
    """Create a single confusion matrix that combines results from all models."""
    # Initialize combined confusion matrix
    combined_cm = np.zeros((len(ESC50_CLASSES), len(ESC50_CLASSES)))
    
    # Collect predictions from all models
    for timestamp in timestamps:
        results = load_fold_results(timestamp)
        y_true = np.array(results['y_true'])
        y_pred_proba = np.array(results['y_pred_proba'])
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Add to combined confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        combined_cm += cm
    
    # Normalize the combined confusion matrix
    combined_cm_norm = combined_cm.astype('float') / combined_cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with larger size for better visibility
    plt.figure(figsize=(50, 40))
    
    # Use a perceptually uniform colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Plot the combined confusion matrix
    sns.heatmap(combined_cm_norm, 
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                xticklabels=ESC50_CLASSES, 
                yticklabels=ESC50_CLASSES,
                annot_kws={'size': 20})
    
    # Customize the plot
    plt.title('Combined Confusion Matrix Across All Models', fontsize=44, pad=30)
    plt.xlabel('Predicted Label', fontsize=36, labelpad=20)
    plt.ylabel('True Label', fontsize=36, labelpad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=24)
    
    # Add colorbar with larger font
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)
    
    # Adjust layout and save with high DPI
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'combined_confusion_matrix.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()
    
    # Also save the raw confusion matrix values
    np.save(os.path.join(save_path, 'combined_confusion_matrix_raw.npy'), combined_cm)
    np.save(os.path.join(save_path, 'combined_confusion_matrix_norm.npy'), combined_cm_norm)

def main():
    # Get timestamps from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics.py <timestamp1> [timestamp2 timestamp3 timestamp4]")
        sys.exit(1)
    
    timestamps = sys.argv[1:]
    model_names = ['MSCRNN', 'MSCNN', 'CNN', 'CRNN'][:len(timestamps)]
    
    # Create output directory
    timestamp_str = '_'.join(timestamps)
    output_dir = f'reports/figures/comparative_metrics_{timestamp_str}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot comprehensive metrics
    plot_comprehensive_metrics(timestamps, model_names, output_dir)
    
    # Plot comparative confusion matrices
    plot_comparative_confusion_matrices(timestamps, model_names, output_dir)
    
    # Plot combined confusion matrix
    plot_combined_confusion_matrix(timestamps, model_names, output_dir)
    
    # Plot comparative ROC curves
    mean_aucs = plot_comparative_roc_curves(timestamps, model_names, output_dir)
    
    # Save comparative metrics with better formatting
    with open(f'{output_dir}/comparative_metrics.txt', 'w') as f:
        f.write("Model Performance Metrics\n")
        f.write("=======================\n\n")
        for model_name, mean_auc in zip(model_names, mean_aucs):
            f.write(f"{model_name}:\n")
            f.write(f"  Mean ROC AUC = {mean_auc:.3f}\n\n")
    
    print(f"Comparative analysis complete. Results saved in: {output_dir}")

if __name__ == '__main__':
    main() 