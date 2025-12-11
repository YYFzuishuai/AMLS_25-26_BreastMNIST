import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
Simple CNN model for BreastMNIST classification.

This module defines:
- A lightweight convolutional neural network (SimpleCNN) tailored for 28×28
  grayscale images (BreastMNIST).
- A CNNModel wrapper class that handles training and evaluation logic, including
  device placement, loss, optimizer, and metric computation.

The CNN serves as the deep learning counterpart (Model B) to the classical
methods (e.g., SVM variants) and is used to study the effect of model capacity,
data augmentation, and training budget.
"""


class SimpleCNN(nn.Module):
    """
    A small convolutional neural network for 28×28 grayscale images.

    Architecture
    -----------
    - Conv(1 → 32, 3×3, padding=1) + ReLU
    - MaxPool 2×2  (28 → 14)
    - Conv(32 → 64, 3×3, padding=1) + ReLU
    - MaxPool 2×2  (14 → 7)
    - Flatten
    - Linear(64*7*7 → 128) + ReLU + Dropout(0.5)
    - Linear(128 → num_classes)

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes. For BreastMNIST, this is 2 (benign vs malignant).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # Spatial size: 28 -> 14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # Spatial size: 14 -> 7
        )

        # Fully connected classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),             # Flatten to shape [B, 64 * 7 * 7]
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images with shape [B, 1, 28, 28].

        Returns
        -------
        torch.Tensor
            Logits of shape [B, num_classes].
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNModel:
    """
    Wrapper for training and evaluating the SimpleCNN model.

    This class encapsulates:
    - Model instantiation and device placement (CPU / GPU).
    - Optimizer and loss function setup.
    - Epoch-based training loop.
    - Evaluation over a DataLoader with computation of common metrics.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes, default is 2 for BreastMNIST.
    lr : float, optional
        Learning rate for the Adam optimizer. Default is 1e-3.
    epochs : int, optional
        Number of training epochs. Default is 5.
    device : torch.device or str, optional
        Device on which to run the model. If None, automatically uses GPU
        if available, otherwise CPU.
    """

    def __init__(self, num_classes: int = 2, lr: float = 1e-3, epochs: int = 5, device=None):
        # Select device: user-specified, CUDA if available, otherwise CPU
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CNN and move it to the chosen device
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)

        # Loss function for multi-class classification (works with integer labels)
        self.criterion = nn.CrossEntropyLoss()

        # Adam optimizer over all model parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Number of epochs to train for
        self.epochs = epochs

    def train(self, train_loader):
        """
        Train the CNN model for the configured number of epochs.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader providing (image, label) batches for training.
            - images: [B, 1, 28, 28]
            - labels: typically [B, 1] for BreastMNIST and need to be squeezed.
        """
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0

            for images, labels in train_loader:
                # Move data to the appropriate device
                images = images.to(self.device)                       # [B, 1, 28, 28]
                # BreastMNIST labels may come as shape [B, 1]; squeeze to [B]
                labels = labels.squeeze().long().to(self.device)

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)

                # Compute cross-entropy loss
                loss = self.criterion(outputs, labels)

                # Backpropagation
                loss.backward()

                # Parameter update
                self.optimizer.step()

                # Accumulate loss (multiply by batch size to later average per sample)
                running_loss += loss.item() * images.size(0)

            # Average loss over the entire training dataset
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch + 1}/{self.epochs}]  Loss: {epoch_loss:.4f}")

    def evaluate(self, test_loader):
        """
        Evaluate the trained CNN model on a test set.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            DataLoader providing (image, label) batches for evaluation.

        Returns
        -------
        dict
            Dictionary containing:
            - "accuracy": float
            - "precision": float
            - "recall": float
            - "f1": float
        """
        self.model.eval()
        all_labels = []
        all_preds = []

        # Disable gradient tracking for faster and memory-efficient inference
        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Predicted class = argmax over logits
                preds = torch.argmax(outputs, dim=1)

                # Store results on CPU for metric computation
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        # Concatenate all batches into single arrays
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        # Compute evaluation metrics using sklearn
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }