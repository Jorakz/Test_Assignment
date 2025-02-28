from torchvision import datasets, transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np


def load_mnist():
    """
    Downloads and loads the MNIST dataset and converts it into numpy arrays.

    Returns:
      - X_train, y_train: Images and labels for the training dataset.
      - X_test, y_test: Images and labels for the test dataset.
    """
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test



def visualize_samples(X, y, dataset_name="Dataset", num_samples=8):
    """
    Visualizes a grid of 8 sample images from a dataset.

    Parameters:
      - X: Numpy array of images.
      - y: Corresponding labels.
      - dataset_name: Name of the dataset (used in the title).
      - num_samples: Number of images to display (default 8).
    """
    plt.figure(figsize=(8, 8))
    # Create a 2x4 grid of images
    for i in range(num_samples):
        plt.subplot(2, 4, i + 1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {y[i]}", fontsize=8)
        plt.axis('off')
    plt.suptitle(f"{dataset_name} Sample Images", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y_train, y_test, dataset_names=("Training", "Test")):
    """
    Plots the class distribution for both training and test datasets on side-by-side subplots.

    Parameters:
      - y_train: Numpy array of training labels.
      - y_test: Numpy array of test labels.
      - dataset_names: Tuple containing names for the datasets (default: ("Training", "Test")).

    This function creates two separate bar charts in one figure:
      - The left subplot shows the class distribution for the training dataset.
      - The right subplot shows the class distribution for the test dataset.
    Each bar is annotated with its count.
    """
    # Define the digit classes 0-9
    digits = np.arange(10)
    # Count frequency for each digit in both datasets
    train_counts = np.array([np.sum(y_train == d) for d in digits])
    test_counts = np.array([np.sum(y_test == d) for d in digits])

    # Create a figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the training class distribution on the left
    ax1.bar(digits, train_counts, color='skyblue')
    ax1.set_title(f"{dataset_names[0]} Class Distribution")
    ax1.set_xlabel("Digit Label")
    ax1.set_ylabel("Frequency")
    ax1.set_xticks(digits)
    # Annotate each bar with its count
    for i, count in enumerate(train_counts):
        ax1.text(i, count + max(train_counts) * 0.01, str(int(count)),
                 ha='center', va='bottom', fontsize=10)

    # Plot the test class distribution on the right
    ax2.bar(digits, test_counts, color='salmon')
    ax2.set_title(f"{dataset_names[1]} Class Distribution")
    ax2.set_xlabel("Digit Label")
    ax2.set_ylabel("Frequency")
    ax2.set_xticks(digits)
    # Annotate each bar with its count
    for i, count in enumerate(test_counts):
        ax2.text(i, count + max(test_counts) * 0.01, str(int(count)),
                 ha='center', va='bottom', fontsize=10)

    plt.suptitle("Class Distribution Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
