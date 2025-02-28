import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time


def train_model(data_dir, model_path="animal_classifier.pth", batch_size=32, epochs=10, learning_rate=0.001):
    """
    Train a CNN model on animal images dataset

    Args:
        data_dir: Directory containing image folders (each folder is a class)
        model_path: Path to save the trained model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")

    # Check if we're running on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Just resize and normalize for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # Print class information
    class_names = full_dataset.classes

    print(class_names)
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Total images: {len(full_dataset)}")

    # Count images per class
    class_counts = {class_names[i]: 0 for i in range(len(class_names))}
    for _, class_idx in full_dataset.samples:
        class_counts[class_names[class_idx]] += 1

    print("Images per class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Use a fixed random seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply validation transform to validation dataset
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model - load pretrained ResNet18 and modify the final layer
    model = models.resnet18(weights='IMAGENET1K_V1')  # Using the newer PyTorch API
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Use different learning rates for feature extraction vs classification layer
    # The base layers are pretrained, so use a smaller learning rate
    optimizer = optim.Adam([
        {'params': list(model.parameters())[:-2], 'lr': learning_rate / 10},  # All layers except fc
        {'params': list(model.parameters())[-2:], 'lr': learning_rate}  # Just fc layer
    ])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = correct / total
        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc)

        # Update learning rate based on validation loss
        scheduler.step(epoch_loss)

        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}")
        print(f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

        # Save the best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            # Create directory if it doesn't exist
            os.makedirs("CNN_model", exist_ok=True)
            model_save_path = os.path.join("CNN_model", model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'class_names': class_names
            }, model_save_path)
            print(f"Model saved with validation accuracy: {best_val_acc:.4f}")

    # Calculate training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes")

    # After training, evaluate on validation set and create confusion matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join("CNN_model", "confusion_matrix.png"))
    print("Confusion matrix saved")

    # Generate and save classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(os.path.join("CNN_model", "classification_report.txt"), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
            f.write(f"  Recall: {report[class_name]['recall']:.4f}\n")
            f.write(f"  F1-score: {report[class_name]['f1-score']:.4f}\n")
            f.write(f"  Support: {report[class_name]['support']}\n\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")

    # Plot and save training/validation curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, 'g-', label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, 'm-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join("CNN_model", "training_curves.png"))
    print("Training curves saved")

    return model, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN for animal classification')
    parser.add_argument('--data_dir', type=str, default='CNN_data/raw-img',
                        help='Directory containing the animal image folders')
    parser.add_argument('--model_path', type=str, default='animal_classifier.pth',
                        help='File name to save the trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')

    args = parser.parse_args()

    train_model(args.data_dir, args.model_path, args.batch_size, args.epochs, args.learning_rate)