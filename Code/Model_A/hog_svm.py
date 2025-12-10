import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_hog_features(x_flat):
    n_samples = x_flat.shape[0]
    images = x_flat.reshape(n_samples, 28, 28)

    features = []
    for img in images:
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        features.append(feat)

    return np.array(features, dtype=np.float32)


class HOGSVMModel:
    def __init__(self, C=1.0, gamma="scale"):
        self.model = SVC(kernel="rbf", C=C, gamma=gamma)

    def train(self, train_x, train_y):
        train_hog = compute_hog_features(train_x)
        self.model.fit(train_hog, train_y)

    def evaluate(self, test_x, test_y):
        test_hog = compute_hog_features(test_x)
        pred = self.model.predict(test_hog)

        return {
            "accuracy": accuracy_score(test_y, pred),
            "precision": precision_score(test_y, pred),
            "recall": recall_score(test_y, pred),
            "f1": f1_score(test_y, pred),
        }