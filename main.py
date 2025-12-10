import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BreastMNIST

# ============= find Model_A / Model_B =============
PROJECT_DIR = os.path.dirname(__file__)
MODEL_A_DIR = os.path.join(PROJECT_DIR, "Code", "Model_A")
MODEL_B_DIR = os.path.join(PROJECT_DIR, "Code", "Model_B")

for p in (MODEL_A_DIR, MODEL_B_DIR):
    if p not in sys.path:
        sys.path.append(p)

# ============= Model A & Model B =============
from svm_model import SVMModel
from pca_svm import PCASVMModel
from hog_svm import HOGSVMModel
from cnn_model import CNNModel


# ======================= Model A =======================

def load_breastmnist_flatten():
    transform = transforms.ToTensor()

    train_dataset = BreastMNIST(split="train", download=True, transform=transform)
    test_dataset = BreastMNIST(split="test", download=True, transform=transform)

    X_train = train_dataset.imgs.reshape(len(train_dataset), -1).astype(np.float32) / 255.0
    y_train = train_dataset.labels.reshape(-1).astype(int)

    X_test = test_dataset.imgs.reshape(len(test_dataset), -1).astype(np.float32) / 255.0
    y_test = test_dataset.labels.reshape(-1).astype(int)

    return X_train, y_train, X_test, y_test


def run_baseline_svm(X_train, y_train, X_test, y_test):
    print("\n=== Running Baseline SVM ===")
    model = SVMModel()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    print("\nModel A (Flatten + SVM) results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def run_pca_svm(X_train, y_train, X_test, y_test):
    print("\n=== Running PCA + SVM ===")
    model = PCASVMModel(n_components=100)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    print("\nModel A (PCA + SVM) results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def run_pca_svm_capacity(X_train, y_train, X_test, y_test):
    print("\n=== Running PCA + SVM Capacity Experiment ===")

    configs = [
        {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 1.0, "gamma": 0.01},
        {"C": 10.0, "gamma": 0.01},
    ]

    for cfg in configs:
        print(f"\nConfig: C={cfg['C']}, gamma={cfg['gamma']}")
        model = PCASVMModel(n_components=100, C=cfg["C"], gamma=cfg["gamma"])
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


def run_hog_svm(X_train, y_train, X_test, y_test):
    print("\n=== Running HOG + SVM ===")
    model = HOGSVMModel(C=1.0, gamma="scale")
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    print("\nModel A (HOG + SVM) results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# ======================= Model B：CNN =======================

def load_breastmnist_torch(batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),              # [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])

    train_dataset = BreastMNIST(split="train", download=True, transform=transform)
    test_dataset = BreastMNIST(split="test", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_breastmnist_torch_aug(batch_size: int = 64):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = BreastMNIST(split="train", download=True, transform=transform_train)
    test_dataset = BreastMNIST(split="test", download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def run_cnn_baseline():
    print("\n=== Running Model B: CNN Baseline ===")
    train_loader, test_loader = load_breastmnist_torch(batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNNModel(num_classes=2, lr=1e-3, epochs=5, device=device)
    model.train(train_loader)
    metrics = model.evaluate(test_loader)

    print("\nModel B (CNN) results on BreastMNIST:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def run_cnn_capacity():
    print("\n=== Running Model B: CNN Capacity Experiment ===")
    train_loader, test_loader = load_breastmnist_torch(batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    configs = [
        {"epochs": 5, "lr": 1e-3},
        {"epochs": 10, "lr": 1e-3},
        {"epochs": 15, "lr": 1e-3},
    ]

    for cfg in configs:
        print(f"\nConfig: epochs={cfg['epochs']}, lr={cfg['lr']}")
        model = CNNModel(
            num_classes=2,
            lr=cfg["lr"],
            epochs=cfg["epochs"],
            device=device
        )
        model.train(train_loader)
        metrics = model.evaluate(test_loader)

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

def run_cnn_augment():
    print("\n=== Running Model B: CNN with Data Augmentation ===")
    train_loader, test_loader = load_breastmnist_torch_aug(batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNNModel(num_classes=2, lr=1e-3, epochs=15, device=device)
    model.train(train_loader)
    metrics = model.evaluate(test_loader)

    print("\nModel B (CNN + Augmentation) results on BreastMNIST:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# ==============================================

if __name__ == "__main__":
    print("=== Loading BreastMNIST Dataset (for Model A) ===")
    X_train, y_train, X_test, y_test = load_breastmnist_flatten()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ------- Model A -------
    run_baseline_svm(X_train, y_train, X_test, y_test)
    run_pca_svm(X_train, y_train, X_test, y_test)
    run_pca_svm_capacity(X_train, y_train, X_test, y_test)
    run_hog_svm(X_train, y_train, X_test, y_test)

    # ------- Model B -------
    run_cnn_baseline()
    run_cnn_capacity()
    run_cnn_augment()