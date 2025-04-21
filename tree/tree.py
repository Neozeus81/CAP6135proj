import sys
import os
from sklearn.tree import DecisionTreeClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')

sys.path.append(file_dir)

from data import Data

class myTree:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, trainData, trainLabels):
        self.model.fit(trainData, trainLabels)

    def predict(self, testData):
        return self.model.predict(testData)
