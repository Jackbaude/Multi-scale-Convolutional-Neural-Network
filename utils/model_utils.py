import os
import json
import torch
from datetime import datetime

def save_model(model, path, fold=None):
    """Save model state dictionary."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def save_results(results, path):
    """Save training results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4) 