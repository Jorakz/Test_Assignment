import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from interfaces import MnistClassifierInterface
# Feed-Forward Neural Network realization
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MnistNNClassifier(MnistClassifierInterface):
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = FeedForwardNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train(self, X_train, y_train, epochs=10, batch_size=128, X_val=None, y_val=None, **kwargs):
        # Preprocess: flatten images and normalize pixel values
        X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255.0
        y_train = y_train.astype('int64')
        X_train_tensor = torch.tensor(X_train)
        y_train_tensor = torch.tensor(y_train)
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == batch_y.data)
                total += batch_X.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double().item() / total
            self.history["train_loss"].append(epoch_loss)
            self.history["train_acc"].append(epoch_acc)

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                self.model.eval()
                # Preprocess validation data (flatten and normalize)
                X_val_proc = X_val.reshape((X_val.shape[0], -1)).astype('float32') / 255.0
                X_val_tensor = torch.tensor(X_val_proc).to(self.device)
                y_val_tensor = torch.tensor(y_val.astype('int64')).to(self.device)
                with torch.no_grad():
                    outputs_val = self.model(X_val_tensor)
                    loss_val = self.criterion(outputs_val, y_val_tensor)
                    _, preds_val = torch.max(outputs_val, 1)
                    correct_val = torch.sum(preds_val == y_val_tensor).item()
                    total_val = y_val.shape[0]
                val_loss = loss_val.item()
                val_acc = correct_val / total_val
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        return self.history

    def predict(self, X_test):
        self.model.eval()
        # Preprocess test data in the same way
        X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255.0
        X_test_tensor = torch.tensor(X_test).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
