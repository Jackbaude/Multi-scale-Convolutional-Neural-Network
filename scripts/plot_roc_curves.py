import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Set global plotting parameters for better visualization
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'axes.linewidth': 1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
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

def load_cross_validation_results(timestamp):
    """Load cross-validation results from JSON file."""
    results_path = f'reports/logs/cross_validation_results_{timestamp}.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def plot_roc_curves(timestamp, save_path):
    """Plot ROC curves for each class with better labels."""
    results = load_cross_validation_results(timestamp)
    
    # Collect predictions from all folds
    all_y_true = []
    all_y_pred_proba = []
    
    for fold_history in results['fold_histories']:
        if 'final_predictions' in fold_history:
            y_true = np.array(fold_history['final_predictions']['y_true'])
            y_pred_proba = np.array(fold_history['final_predictions']['y_pred_proba'])
            
            all_y_true.extend(y_true)
            all_y_pred_proba.extend(y_pred_proba)
    
    # Convert to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred_proba = np.array(all_y_pred_proba)
    
    # Create figure with larger size
    plt.figure(figsize=(20, 16))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Use a color map that's easier to distinguish
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ESC50_CLASSES)))
    
    # Plot ROC curves for each class
    for i, color in zip(range(len(ESC50_CLASSES)), colors):
        fpr[i], tpr[i], _ = roc_curve(all_y_true == i, all_y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot with thicker lines and markers
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{ESC50_CLASSES[i]} (AUC = {roc_auc[i]:.2f})',
                 marker='o', markersize=4, markevery=10)
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set axis limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', labelpad=10)
    plt.ylabel('True Positive Rate', labelpad=10)
    plt.title('ROC Curves for CRNN Model', pad=20)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    
    # Create a legend with better formatting
    # Group classes by their AUC scores for better organization
    sorted_indices = np.argsort([roc_auc[i] for i in range(len(ESC50_CLASSES))])
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [handles[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
    # Create legend with two columns
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=8, ncol=2, frameon=True, framealpha=0.9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save AUC scores
    auc_scores = {ESC50_CLASSES[i]: roc_auc[i] for i in range(len(ESC50_CLASSES))}
    with open(os.path.join(save_path, 'auc_scores.txt'), 'w') as f:
        for class_name, score in sorted(auc_scores.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{class_name}: {score:.3f}\n")

def main():
    # Use the most recent CRNN results timestamp
    timestamp = '20250505_142723'
    
    # Create output directory
    output_dir = f'reports/figures/crnn_roc_curves_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ROC curves
    plot_roc_curves(timestamp, output_dir)
    
    print(f"ROC curves analysis complete. Results saved in: {output_dir}")

if __name__ == '__main__':
    main() 