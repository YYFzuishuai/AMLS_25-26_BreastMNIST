import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
Simple baseline SVM model for BreastMNIST classification.

This module defines a minimal SVM-based classifier using an RBF kernel.
Unlike the HOG or PCA variants, this model operates directly on raw flattened
images without additional feature extraction or dimensionality reduction.

It serves as a baseline to compare:
- Raw pixel representation vs. engineered features (HOG)
- Raw pixel representation vs. compressed latent space (PCA)
"""


class SVMModel:
    """
    Baseline RBF-SVM classifier for BreastMNIST images.

    This class applies an SVM directly to flattened 28×28 grayscale images.
    Although not as feature-rich as HOG or PCA pipelines, it provides a
    simple reference point for evaluating the effect of preprocessing and
    feature engineering.

    Parameters
    ----------
    None (kernel and hyperparameters are fixed for simplicity)
    """

    def __init__(self):
        # Standard SVM with RBF kernel; C and gamma = default ("scale")
        self.model = SVC(kernel='rbf')

    def train(self, train_x, train_y):
        """
        Train the SVM classifier on raw pixel inputs.

        Parameters
        ----------
        train_x : np.ndarray
            Flattened training images, shape (n_samples, 784).
        train_y : np.ndarray
            Ground-truth labels for training samples.

        Notes
        -----
        This baseline applies no preprocessing. It is expected to perform
        worse than feature-engineered methods but serves as an important
        comparison for evaluating model capacity and sensitivity to raw data.
        """
        self.model.fit(train_x, train_y)

    def evaluate(self, test_x, test_y):
        """
        Evaluate the trained baseline SVM on the test set.

        Parameters
        ----------
        test_x : np.ndarray
            Flattened test images.
        test_y : np.ndarray
            Ground-truth test labels.

        Returns
        -------
        dict
            A dictionary containing accuracy, precision, recall, and f1-score.
        """
        # Predict class labels
        pred = self.model.predict(test_x)

        # Compute evaluation metrics
        return {
            'accuracy': accuracy_score(test_y, pred),
            'precision': precision_score(test_y, pred),
            'recall': recall_score(test_y, pred),
            'f1': f1_score(test_y, pred)
        }