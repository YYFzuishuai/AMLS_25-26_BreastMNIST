import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
PCA + SVM model for BreastMNIST classification.

This module implements a classical machine learning pipeline where:
1. Images (flattened 28×28 grayscale) are reduced to a lower-dimensional
   representation using Principal Component Analysis (PCA).
2. An RBF-kernel Support Vector Machine (SVM) is trained on the PCA-reduced data.

The PCA step helps denoise the input, capture the most informative components,
and reduce computational cost, especially useful for high-dimensional data such
as flattened images.
"""


class PCASVMModel:
    """
    PCA + RBF-SVM classifier for BreastMNIST.

    This class performs dimensionality reduction with PCA and trains an SVM
    classifier on the transformed feature space.

    Parameters
    ----------
    n_components : int, optional
        Number of principal components to keep. Controls model capacity:
        fewer components → simpler model; more components → richer representation.
        Default is 100.
    C : float, optional
        Regularization parameter for the SVM. Larger values reduce regularization.
        Default is 1.0.
    gamma : {"scale", "auto"} or float, optional
        Kernel coefficient for the RBF kernel. Controls how far a single sample’s
        influence extends. Default is "scale".
    """

    def __init__(self, n_components=100, C=1.0, gamma="scale"):
        # PCA module to reduce input dimensionality
        self.pca = PCA(n_components=n_components)

        # RBF-kernel SVM classifier
        self.model = SVC(kernel="rbf", C=C, gamma=gamma)

    def train(self, train_x, train_y):
        """
        Train the PCA+SVM model on training data.

        Parameters
        ----------
        train_x : np.ndarray
            Flattened training images with shape (n_samples, 784).
        train_y : np.ndarray
            Ground-truth labels for training data.

        Notes
        -----
        - PCA is fitted on the training data only (important for preventing leakage).
        - SVM is then trained on the transformed representation.
        """
        # Fit PCA on training data and reduce dimensionality
        train_x = self.pca.fit_transform(train_x)

        # Train SVM using the PCA-transformed features
        self.model.fit(train_x, train_y)

    def evaluate(self, test_x, test_y):
        """
        Evaluate the trained PCA+SVM model on the test set.

        Parameters
        ----------
        test_x : np.ndarray
            Flattened test images with shape (n_samples, 784).
        test_y : np.ndarray
            Ground-truth labels for testing.

        Returns
        -------
        dict
            Dictionary containing metrics:
            - accuracy
            - precision
            - recall
            - f1
        """
        # Apply PCA transformation learned from the training set
        test_x = self.pca.transform(test_x)

        # Predict labels for the test set
        pred = self.model.predict(test_x)

        # Compute evaluation metrics
        return {
            "accuracy": accuracy_score(test_y, pred),
            "precision": precision_score(test_y, pred),
            "recall": recall_score(test_y, pred),
            "f1": f1_score(test_y, pred),
        }