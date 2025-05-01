import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from esc50_utils import (
    setup_directories, setup_logging, download_esc50,
    ESC50Dataset, save_model, save_results,
    calculate_average_metrics, log_metrics,
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, DEVICE
)

# Model parameters
BATCH_SIZE = 32
K = 5
METRIC = 'euclidean'
N_SPLITS = 5
RANDOM_STATE = 42

class KNNModel:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        if self.metric == 'euclidean':
            # Compute pairwise distances
            distances = torch.cdist(X, self.X_train)
        else:  # cosine
            # Normalize vectors
            X_norm = X / torch.norm(X, dim=1, keepdim=True)
            X_train_norm = self.X_train / torch.norm(self.X_train, dim=1, keepdim=True)
            # Compute cosine similarity
            distances = 1 - torch.mm(X_norm, X_train_norm.t())
        
        # Get k nearest neighbors
        _, indices = torch.topk(distances, self.k, largest=False)
        neighbor_labels = self.y_train[indices]
        
        # Get most common label
        preds = torch.mode(neighbor_labels, dim=1)[0]
        return preds

def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(DEVICE)
            preds = model.predict(features)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    return metrics

def main():
    # Setup
    setup_directories()
    timestamp = setup_logging('knn_training')
    
    # Download and prepare dataset
    audio_dir, meta_file = download_esc50()
    
    # Create dataset
    dataset = ESC50Dataset(audio_dir, meta_file)
    
    # Initialize k-fold cross-validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Store results for each fold
    all_fold_results = {
        'fold_metrics': [],
        'average_metrics': None
    }
    
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, [dataset[i][1] for i in range(len(dataset))])):
        logging.info(f'\nStarting Fold {fold + 1}/{N_SPLITS}')
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Prepare training data
        X_train = []
        y_train = []
        for features, labels in train_loader:
            X_train.append(features)
            y_train.append(labels)
        X_train = torch.cat(X_train).to(DEVICE)
        y_train = torch.cat(y_train).to(DEVICE)
        
        # Initialize and train model
        model = KNNModel(k=K, metric=METRIC)
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate(model, test_loader)
        
        # Store fold results
        all_fold_results['fold_metrics'].append({
            'fold': fold + 1,
            'metrics': metrics
        })
        
        # Save model for this fold
        model_path = f'models/esc50_knn_model_fold{fold+1}_{timestamp}.pth'
        save_model(model, model_path, fold + 1, metrics, {
            'X_train': X_train.cpu(),
            'y_train': y_train.cpu(),
            'k': K,
            'metric': METRIC
        })
        
        # Log metrics
        log_metrics(metrics, f'Fold {fold+1} ')
    
    # Calculate average metrics across folds
    avg_metrics = calculate_average_metrics(all_fold_results['fold_metrics'])
    all_fold_results['average_metrics'] = avg_metrics
    
    # Log average metrics
    logging.info("\nAverage Metrics Across Folds:")
    log_metrics(avg_metrics)
    
    # Save all fold results
    results_path = f'reports/logs/knn_cross_validation_results_{timestamp}.json'
    save_results(all_fold_results, results_path)

if __name__ == '__main__':
    main() 