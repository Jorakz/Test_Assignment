from interfaces import MnistClassifierInterface
from models.rf import MnistRFClassifier
from models.nn import MnistNNClassifier
from models.cnn import MnistCNNClassifier

class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm: str, device: str = 'cpu'):
        """
        A unified MNIST classifier.
        Parameter 'algorithm': 'rf' for Random Forest, 'nn' for Feed-Forward Neural Network,
        and 'cnn' for Convolutional Neural Network.
        """
        self.algorithm = algorithm.lower()
        if self.algorithm == 'rf':
            self.model = MnistRFClassifier()
        elif self.algorithm == 'nn':
            # Initialize Feed-Forward NN using the specified device (CPU/CUDA)
            self.model = MnistNNClassifier(device=device)
        elif self.algorithm == 'cnn':
            # Initialize Convolutional NN using the specified device
            self.model = MnistCNNClassifier(device=device)
        else:
            raise ValueError(f"Unknown algorithm type '{algorithm}'. Expected 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model on (X_train, y_train). Additional keyword arguments (such as epochs, batch_size, X_val, y_val)
        are passed to the underlying classifier.
        """
        # Delegate training to the underlying model
        result = self.model.train(X_train, y_train, **kwargs)
        return result  # Return training result (e.g., history for neural networks)

    def predict(self, X_test):
        """
        Predicts labels for X_test.
        """
        # Delegate prediction to the underlying model
        return self.model.predict(X_test)