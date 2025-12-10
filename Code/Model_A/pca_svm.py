import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PCASVMModel:
    def __init__(self, n_components=100, C=1.0, gamma="scale"):
        self.pca = PCA(n_components=n_components)
        self.model = SVC(kernel="rbf", C=C, gamma=gamma)

    def train(self, train_x, train_y):
        # PCA fit + transform
        train_x = self.pca.fit_transform(train_x)
        self.model.fit(train_x, train_y)

    def evaluate(self, test_x, test_y):
        # apply PCA on test set
        test_x = self.pca.transform(test_x)
        pred = self.model.predict(test_x)

        return {
            "accuracy": accuracy_score(test_y, pred),
            "precision": precision_score(test_y, pred),
            "recall": recall_score(test_y, pred),
            "f1": f1_score(test_y, pred),
        }