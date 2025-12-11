import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
HOG + SVM model for BreastMNIST classification.

This module defines:
- A helper function to compute Histogram of Oriented Gradients (HOG) features
  from flattened 28x28 grayscale images.
- A HOGSVMModel class that wraps an RBF-kernel SVM trained on HOG features.

The model is intended for binary classification on BreastMNIST
(benign vs. malignant) but is generic enough for other 28x28 grayscale datasets.
"""


def compute_hog_features(x_flat):
    """
    Compute HOG (Histogram of Oriented Gradients) features for a batch of images.

    Parameters
    ----------
    x_flat : np.ndarray
        Array of flattened grayscale images with shape (n_samples, 784).
        Each row is expected to correspond to a 28x28 image.

    Returns
    -------
    np.ndarray
        Array of HOG feature vectors with shape (n_samples, n_features),
        where n_features depends on the HOG configuration.
    """
    # Number of samples in the batch
    n_samples = x_flat.shape[0]

    # Reshape flattened vectors back to 2D images (28x28)
    images = x_flat.reshape(n_samples, 28, 28)

    features = []
    for img in images:
        # Compute HOG descriptor for a single image.
        # The parameters below control the granularity and normalization
        # of the gradient-based features.
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        features.append(feat)

    # Stack all HOG vectors into a single 2D array of shape (n_samples, n_features)
    return np.array(features, dtype=np.float32)


class HOGSVMModel:
    """
    HOG + RBF-SVM classifier for BreastMNIST images.

    This class:
    - Converts flattened 28x28 images into HOG features.
    - Trains an RBF-kernel Support Vector Machine on those features.
    - Evaluates the model using standard binary classification metrics.

    Parameters
    ----------
    C : float, optional
        Regularization parameter for the SVM. Higher values reduce regularization
        and can lead to more complex decision boundaries. Default is 1.0.
    gamma : {"scale", "auto"} or float, optional
        Kernel coefficient for the RBF kernel. Controls how far the influence
        of a single training example reaches. Default is "scale".
    """

    def __init__(self, C=1.0, gamma="scale"):
        # Underlying scikit-learn SVC model with RBF kernel
        self.model = SVC(kernel="rbf", C=C, gamma=gamma)

    def train(self, train_x, train_y):
        """
        Train the HOG+SVM classifier on the given training data.

        Parameters
        ----------
        train_x : np.ndarray
            Training images as flattened vectors with shape (n_samples, 784).
        train_y : np.ndarray
            Ground-truth labels for the training data. Expected to be a 1D array.
        """
        # Extract HOG features from flattened training images
        train_hog = compute_hog_features(train_x)

        # Fit the SVM model on HOG features
        self.model.fit(train_hog, train_y)

    def evaluate(self, test_x, test_y):
        """
        Evaluate the trained model on a test set.

        Parameters
        ----------
        test_x : np.ndarray
            Test images as flattened vectors with shape (n_samples, 784).
        test_y : np.ndarray
            Ground-truth labels for the test data.

        Returns
        -------
        dict
            A dictionary containing the following metrics:
            - "accuracy": float
            - "precision": float
            - "recall": float
            - "f1": float
        """
        # Compute HOG features for the test images
        test_hog = compute_hog_features(test_x)

        # Predict labels using the trained SVM
        pred = self.model.predict(test_hog)

        # Compute standard binary classification metrics
        return {
            "accuracy": accuracy_score(test_y, pred),
            "precision": precision_score(test_y, pred),
            "recall": recall_score(test_y, pred),
            "f1": f1_score(test_y, pred),
        }