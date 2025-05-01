import os
import torch
import torch.nn as nn
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
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
C = 1.0
KERNEL = 'linear'
GAMMA = 'scale'
N_SPLITS = 5
RANDOM_STATE = 42

class SVMModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)
        
    def hinge_loss(self, outputs, targets):
        # Hinge loss implementation
        correct_scores = outputs[torch.arange(len(targets)), targets]
        margins = outputs - correct_scores.unsqueeze(1) + 1
        margins[torch.arange(len(targets)), targets] = 0
        loss = torch.max(torch.zeros_like(margins), margins).sum(dim=1).mean()
        return loss

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for features, labels in tqdm(train_loader, desc='Training'):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            
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
    timestamp = setup_logging('svm_training')
    
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
    
    # Get first sample to determine input dimension
    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[0]
    num_classes = 50  # ESC-50 has 50 classes
    
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, [dataset[i][1] for i in range(len(dataset))])):
        logging.info(f'\nStarting Fold {fold + 1}/{N_SPLITS}')
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Initialize model
        model = SVMModel(input_dim, num_classes).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        criterion = model.hinge_loss
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}')
        
        # Evaluate model
        metrics = evaluate(model, test_loader)
        
        # Store fold results
        all_fold_results['fold_metrics'].append({
            'fold': fold + 1,
            'metrics': metrics
        })
        
        # Save model for this fold
        model_path = f'models/esc50_svm_model_fold{fold+1}_{timestamp}.pth'
        save_model(model, model_path, fold + 1, metrics)
        
        # Log metrics
        log_metrics(metrics, f'Fold {fold+1} ')
    
    # Calculate average metrics across folds
    avg_metrics = calculate_average_metrics(all_fold_results['fold_metrics'])
    all_fold_results['average_metrics'] = avg_metrics
    
    # Log average metrics
    logging.info("\nAverage Metrics Across Folds:")
    log_metrics(avg_metrics)
    
    # Save all fold results
    results_path = f'reports/logs/svm_cross_validation_results_{timestamp}.json'
    save_results(all_fold_results, results_path)

if __name__ == '__main__':
    main() 