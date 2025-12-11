import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BreastMNIST

# ============= Path setup for Model_A / Model_B modules =============
"""
This script serves as the main entry point for the AMLS assignment.

It:
- Loads the BreastMNIST dataset in different formats (flattened NumPy arrays and
  PyTorch tensors).
- Runs classical models (Model A: SVM variants) on flattened images and
  feature-engineered representations (PCA, HOG).
- Runs a convolutional neural network (Model B: CNN) on the image tensors.
- Performs experiments on model capacity and data augmentation for both
  classical and deep learning approaches.
"""

# Project directory where this script is located
PROJECT_DIR = os.path.dirname(__file__)

# Expected locations of the Model_A and Model_B code folders
MODEL_A_DIR = os.path.join(PROJECT_DIR, "Code", "Model_A")
MODEL_B_DIR = os.path.join(PROJECT_DIR, "Code", "Model_B")

# Add model directories to Python path so that imports work when running `python main.py`
for p in (MODEL_A_DIR, MODEL_B_DIR):
    if p not in sys.path:
        sys.path.append(p)

# ============= Import Model A & Model B implementations =============
from svm_model import SVMModel
from pca_svm import PCASVMModel
from hog_svm import HOGSVMModel
from cnn_model import CNNModel


# ======================= Model A: classical ML =======================

def load_breastmnist_flatten():
    """
    Load BreastMNIST data and return flattened NumPy arrays.

    The function downloads the BreastMNIST dataset (if not already present)
    using the medmnist API, converts images to tensors and then to flattened
    NumPy arrays in [0, 1] range.

    Returns
    -------
    X_train : np.ndarray
        Training images flattened to shape (n_train, 784) and normalized.
    y_train : np.ndarray
        Training labels as 1D integer array.
    X_test : np.ndarray
        Test images flattened to shape (n_test, 784) and normalized.
    y_test : np.ndarray
        Test labels as 1D integer array.
    """
    # Convert PIL images to tensors; we will later reshape and convert to NumPy
    transform = transforms.ToTensor()

    train_dataset = BreastMNIST(split="train", download=True, transform=transform)
    test_dataset = BreastMNIST(split="test", download=True, transform=transform)

    # imgs are stored as NumPy arrays (n_samples, 28, 28)
    X_train = train_dataset.imgs.reshape(len(train_dataset), -1).astype(np.float32) / 255.0
    y_train = train_dataset.labels.reshape(-1).astype(int)

    X_test = test_dataset.imgs.reshape(len(test_dataset), -1).astype(np.float32) / 255.0
    y_test = test_dataset.labels.reshape(-1).astype(int)

    return X_train, y_train, X_test, y_test


def run_baseline_svm(X_train, y_train, X_test, y_test):
    """
    Run baseline SVM on raw flattened BreastMNIST images.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Flattened images in shape (n_samples, 784).
    y_train, y_test : np.ndarray
        Integer labels.
    """
    print("\n=== Running Baseline SVM ===")
    model = SVMModel()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    print("\nModel A (Flatten + SVM) results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def run_pca_svm(X_train, y_train, X_test, y_test):
    """
    Run PCA + SVM pipeline on BreastMNIST.

    This evaluates how PCA-based dimensionality reduction affects performance.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Flattened images.
    y_train, y_test : np.ndarray
        Integer labels.
    """
    print("\n=== Running PCA + SVM ===")
    model = PCASVMModel(n_components=100)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    print("\nModel A (PCA + SVM) results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def run_pca_svm_capacity(X_train, y_train, X_test, y_test):
    """
    Run capacity experiment for PCA + SVM by varying SVM hyperparameters.

    Here we fix the number of PCA components and vary C and gamma for the RBF SVM
    to study how model capacity and regularisation affect performance.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Flattened images.
    y_train, y_test : np.ndarray
        Integer labels.
    """
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
    """
    Run HOG + SVM pipeline on BreastMNIST.

    This configuration uses hand-crafted HOG features as input to an RBF-SVM,
    allowing comparison with raw pixels and PCA-based representations.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Flattened images.
    y_train, y_test : np.ndarray
        Integer labels.
    """
    print("\n=== Running HOG + SVM ===")
    model = HOGSVMModel(C=1.0, gamma="scale")
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    print("\nModel A (HOG + SVM) results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# ======================= Model B: CNN =======================

def load_breastmnist_torch(batch_size: int = 64):
    """
    Load BreastMNIST as PyTorch datasets and return DataLoaders (without augmentation).

    The images are:
    - Converted to tensors in [0, 1]
    - Normalised with mean 0.5 and std 0.5 (approximately centering around 0)

    Parameters
    ----------
    batch_size : int, optional
        Batch size for the DataLoaders. Default is 64.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for the training split.
    test_loader : DataLoader
        DataLoader for the test split.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),              # Convert to [0, 1] tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to roughly [-1, 1]
    ])

    train_dataset = BreastMNIST(split="train", download=True, transform=transform)
    test_dataset = BreastMNIST(split="test", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_breastmnist_torch_aug(batch_size: int = 64):
    """
    Load BreastMNIST with data augmentation for training.

    Training transformations:
    - Random horizontal flip
    - Random small rotation
    - Tensor conversion and normalization

    Test transformations:
    - Only tensor conversion and normalization (no augmentation)

    Parameters
    ----------
    batch_size : int, optional
        Batch size for both training and test DataLoaders.

    Returns
    -------
    train_loader : DataLoader
        Augmented training DataLoader.
    test_loader : DataLoader
        Non-augmented test DataLoader.
    """
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
    """
    Run baseline CNN training and evaluation on BreastMNIST without augmentation.

    Uses:
    - Fixed learning rate (1e-3)
    - Fixed number of epochs (5)
    - Standard normalisation but no explicit data augmentation
    """
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
    """
    Run capacity experiment for the CNN by varying the number of training epochs.

    This experiment keeps the architecture and learning rate fixed, and increases
    the training budget (epochs) to see how performance scales.

    Configurations tested:
    - 5 epochs, 1e-3 learning rate
    - 10 epochs, 1e-3 learning rate
    - 15 epochs, 1e-3 learning rate
    """
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
    """
    Run CNN training with data augmentation on BreastMNIST.

    This experiment evaluates how simple geometric augmentations
    (horizontal flip + small rotations) affect the generalisation performance
    compared to the non-augmented baseline.
    """
    print("\n=== Running Model B: CNN with Data Augmentation ===")
    train_loader, test_loader = load_breastmnist_torch_aug(batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use a slightly larger training budget here to allow the model to benefit from augmentation
    model = CNNModel(num_classes=2, lr=1e-3, epochs=15, device=device)
    model.train(train_loader)
    metrics = model.evaluate(test_loader)

    print("\nModel B (CNN + Augmentation) results on BreastMNIST:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# ==================== Entry point ====================

if __name__ == "__main__":
    """
    Main entry point.

    When run as a script, this will:
    - Load BreastMNIST for classical models (flattened NumPy arrays)
    - Run Model A experiments (baseline SVM, PCA+SVM, capacity, HOG+SVM)
    - Run Model B experiments (CNN baseline, CNN capacity, CNN with augmentation)
    """

    print("=== Loading BreastMNIST Dataset (for Model A) ===")
    X_train, y_train, X_test, y_test = load_breastmnist_flatten()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ------- Model A: classical ML -------
    run_baseline_svm(X_train, y_train, X_test, y_test)
    run_pca_svm(X_train, y_train, X_test, y_test)
    run_pca_svm_capacity(X_train, y_train, X_test, y_test)
    run_hog_svm(X_train, y_train, X_test, y_test)

    # ------- Model B: deep learning -------
    run_cnn_baseline()
    run_cnn_capacity()
    run_cnn_augment()