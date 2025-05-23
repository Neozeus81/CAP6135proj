import sys
import os
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')

sys.path.append(file_dir)

from data import Data
""" kernals linear ploy rbf sigmoid  """
class mySVM:
    def __init__(self, current):
        self.model = SVC(kernel=current, gamma='scale', max_iter = 1000)

    def train(self, trainData, trainLabels):
        self.model.fit(trainData, trainLabels)

    def predict(self, testData):
        return self.model.predict(testData)

