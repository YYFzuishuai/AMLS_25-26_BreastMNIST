import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SimpleCNN(nn.Module):

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28 -> 14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14 -> 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),             # 64 * 7 * 7
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNModel:

    def __init__(self, num_classes: int = 2, lr: float = 1e-3, epochs: int = 5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)         # [B, 1, 28, 28]
                labels = labels.squeeze().long().to(self.device)  # BreastMNIST: [B, 1] -> [B]

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch + 1}/{self.epochs}]  Loss: {epoch_loss:.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }
