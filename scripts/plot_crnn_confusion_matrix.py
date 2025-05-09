import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set global plotting parameters for better visualization
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

def load_cross_validation_results(timestamp):
    """Load cross-validation results from JSON file."""
    results_path = f'reports/logs/cross_validation_results_{timestamp}.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def plot_confusion_matrix(timestamp, save_path):
    """Plot confusion matrix for the CRNN model using cross-validation results."""
    results = load_cross_validation_results(timestamp)
    
    # Collect predictions from all folds
    all_y_true = []
    all_y_pred = []
    
    for fold_history in results['fold_histories']:
        if 'final_predictions' in fold_history:
            y_true = np.array(fold_history['final_predictions']['y_true'])
            y_pred_proba = np.array(fold_history['final_predictions']['y_pred_proba'])
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
    
    # Convert to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with larger size for better visibility
    plt.figure(figsize=(50, 40))
    
    # Use a perceptually uniform colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Plot the confusion matrix
    sns.heatmap(cm_norm, 
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                xticklabels=ESC50_CLASSES, 
                yticklabels=ESC50_CLASSES,
                annot_kws={'size': 20})
    
    # Customize the plot
    plt.title('CRNN Model Confusion Matrix', fontsize=44, pad=30)
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
    plt.savefig(os.path.join(save_path, 'crnn_confusion_matrix.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()
    
    # Also save the raw confusion matrix values
    np.save(os.path.join(save_path, 'crnn_confusion_matrix_raw.npy'), cm)
    np.save(os.path.join(save_path, 'crnn_confusion_matrix_norm.npy'), cm_norm)

def main():
    # Use the most recent CRNN results timestamp
    timestamp = '20250505_142723'
    
    # Create output directory
    output_dir = f'reports/figures/crnn_confusion_matrix_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(timestamp, output_dir)
    
    print(f"Confusion matrix analysis complete. Results saved in: {output_dir}")

if __name__ == '__main__':
    main() 