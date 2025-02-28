import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_metrics(history, model_name):
    """
    Plots training and validation loss/accuracy curves.

    Parameters:
      - history: dictionary containing lists for 'train_loss', 'train_acc',
                 and optionally 'val_loss' and 'val_acc'.
      - model_name: string indicating the model name for the plot title.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    if history.get('val_loss'):
        plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.grid()
    plt.legend()

    # Accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    if history.get('val_acc'):
        plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.grid()
    plt.legend()


    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots a confusion matrix using seaborn heatmap.

    Parameters:
      - y_true: Ground truth labels.
      - y_pred: Predicted labels.
      - model_name: string indicating the model name for the plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    #plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()


 