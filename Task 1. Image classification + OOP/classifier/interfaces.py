from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model on the given training data.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict labels for the given test data.
        """
        pass
