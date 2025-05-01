import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def score(self, X, y):
        return self.model.score(X, y) 