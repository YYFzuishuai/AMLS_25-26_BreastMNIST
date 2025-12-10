import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SVMModel:
    def __init__(self):
        self.model = SVC(kernel='rbf')

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def evaluate(self, test_x, test_y):
        pred = self.model.predict(test_x)
        return {
            'accuracy': accuracy_score(test_y, pred),
            'precision': precision_score(test_y, pred),
            'recall': recall_score(test_y, pred),
            'f1': f1_score(test_y, pred)
        }