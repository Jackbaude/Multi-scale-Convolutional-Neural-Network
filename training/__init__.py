"""
ESC-50 Training Scripts
"""

from .train_mscnn import main as train_mscnn
from .train_svm import main as train_svm
from .train_knn import main as train_knn

__all__ = ['train_mscnn', 'train_svm', 'train_knn'] 