import numpy as np
import torch
from sklearn.metrics import classification_report

# Import our unified classifier wrapper
from mnist_classifier import MnistClassifier

# Import dataset utilities and visualization functions
from loading_dataset import load_mnist, visualize_samples, plot_class_distribution

# Import result plotting utilities
from result_plot import plot_metrics, plot_confusion_matrix


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a model by predicting labels on X_test, printing the accuracy and classification report,
    and returning the predictions.
    """
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{model_name} Classification Report:\n", classification_report(y_test, predictions))
    return predictions


def main():
    # Load MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist()

    # Visualize sample images from the training and test datasets
    visualize_samples(X_train, y_train, dataset_name="Training")
    visualize_samples(X_test, y_test, dataset_name="Test")

    # Plot side-by-side class distributions for training and test datasets
    plot_class_distribution(y_train, y_test)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dictionary mapping model names to algorithm identifiers
    models = {
        "Random Forest Classifier": "rf",
        "Feed-Forward Neural Network": "nn",
        "Convolutional Neural Network": "cnn"
    }

    # Train and evaluate each model using the unified interface
    for model_name, algo in models.items():
        print(f"\n=== {model_name} ===")
        classifier = MnistClassifier(algo, device=device)
        # Train the model; for RF the additional parameters (epochs, batch_size, etc.) will be ignored
        history = classifier.train(X_train, y_train, epochs=10, batch_size=128, X_val=X_test, y_val=y_test)
        # Evaluate the model on test data
        predictions = evaluate_model(classifier, X_test, y_test, model_name=model_name)
        # If training history exists (for neural networks), plot the training metrics
        if history is not None:
            plot_metrics(history, model_name)
        # Plot the confusion matrix for the model
        plot_confusion_matrix(y_test, predictions, model_name)


if __name__ == "__main__":
    main()