# MNIST Image Classifier - OOP

An object-oriented implementation of various machine learning models for MNIST handwritten digit classification. This repository demonstrates how to use different classifier architectures (Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network) with a common interface.

## Features

- **Unified Interface**: All classifiers implement the same abstract base class
- **Multiple Model Architectures**: Random Forest, Feed-Forward Neural Network, and CNN
- **Visualization Tools**: Utilities for data exploration and model performance analysis
- **Interactive Demo**: Jupyter notebook for hands-on exploration

## Project Structure

```
├── interfaces.py            # Abstract base class for MNIST classifiers
├── loading_dataset.py       # Functions to load the MNIST dataset
├── mnist_classifier.py      # Unified classifier wrapper
├── result_plot.py           # Visualization utilities
├── test.py                  # Main script to run all models and compare results
├── models/
│   ├── rf.py                # Random Forest classifier implementation
│   ├── nn.py                # Feed-forward Neural Network implementation
│   └── cnn.py               # Convolutional Neural Network implementation
├── demo.ipynb               # Jupyter notebook for interactive demonstration
└── requirements.txt         # Project dependencies
```

## Requirements

This project requires the following packages:

- Python 3.6+
- PyTorch
- scikit-learn
- NumPy
- Matplotlib
- seaborn
- torchvision

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Jorakz/mnist-classifier-oop.git

# Change to project directory
cd mnist-classifier-oop

# Install requirements
pip install -r requirements.txt
```

## Usage

### Using the Demo Notebook

For an interactive demonstration:

```bash
jupyter notebook demo.ipynb
```

The notebook provides:
- Dataset exploration and visualization
- Step-by-step model training and evaluation
- Performance comparison between models
- Analysis of misclassifications
- Visualizations of model predictions

### Running the Full Test Suite

To train and evaluate all models:

```bash
python test.py
```

This will:
1. Load the MNIST dataset
2. Train each classifier
3. Evaluate performance
4. Generate visualizations

### Using Individual Classifiers

Each classifier implements the same interface, making them interchangeable:

```python
from mnist_classifier import MnistClassifier
from loading_dataset import load_mnist

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Create and train a CNN model
classifier = MnistClassifier('cnn', device='cpu')
classifier.train(X_train, y_train, epochs=10, batch_size=128, X_val=X_test, y_val=y_test)

# Make predictions
predictions = classifier.predict(X_test)
```

## Model Architectures

### Random Forest
- Implementation: scikit-learn's RandomForestClassifier
- Configuration: 100 trees, random state 42
- Features: Flattened 28×28 images (784 features)
- Typical accuracy: ~97%

### Feed-Forward Neural Network
- Architecture: 784 → 512 → 10
- Activation: ReLU
- Regularization: 20% dropout
- Optimizer: Adam (lr=0.001)
- Typical accuracy: ~98%

### Convolutional Neural Network
- Architecture:
  - Conv layer: 32 filters, 3×3 kernel, padding=1
  - Max pooling: 2×2
  - Fully connected: 32×14×14 → 128 → 10
- Activation: ReLU
- Regularization: 25% dropout
- Optimizer: Adam (lr=0.001)
- Typical accuracy: ~99%

## Performance

The models generally achieve the following accuracy on the MNIST test set:
- Random Forest: ~96-97%
- Feed-Forward NN: ~97-98%
- CNN: ~98-99%

Detailed performance metrics and visualizations are available in the demo notebook.

## Contributing

Contributions are welcome! Here are some ways you can contribute:
- Add new classifier implementations
- Improve existing models
- Enhance visualization tools
- Expand the demo notebook with additional examples

## License

[MIT](LICENSE)
