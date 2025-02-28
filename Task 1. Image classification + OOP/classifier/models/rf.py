import numpy as np
from sklearn.ensemble import RandomForestClassifier
from interfaces import MnistClassifierInterface

#Random Forest realization
class MnistRFClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train, **kwargs):
        # Flatten images: (N, 28, 28) -> (N, 784)
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        self.model.fit(X_train_flat, y_train)

    def predict(self, X_test):
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        return self.model.predict(X_test_flat)