from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)
        return self

    def predict(self, X):
        distances = cdist(X, self.X_train)
        indices = np.argpartition(distances, self.n_neighbors, axis=1)
        indices = indices[:, :self.n_neighbors]
        nearest_neighbors = self.y_train[indices]
        sum = np.sum(nearest_neighbors, axis=1)
        predictions = np.sign(sum)
        return predictions
