import sys
import os
from sklearn.neural_network import MLPClassifier, MLPRegressor


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')

sys.path.append(file_dir)

from data import Data

train = Data("KDD_TRAIN", ['protocol_type', 'service', 'flag'])
test = Data("KDD_TEST_PLUS", ['protocol_type', 'service', 'flag'])
col_diffs = list(set(train.data.columns.values) - set(test.data.columns.values))
train.data = train.data.drop(columns=col_diffs, axis=1)
print(train.data.head())

class myMLP:
    def __init__(self, x, y, act, epochs):
        self.model = MLPClassifier(hidden_layer_sizes=(x,y), activation=act, solver='adam', max_iter=epochs, random_state=42)

    def train(trainData, trainLabels):
        self.model.fit(trainData, trainLabels)

    def predict(self, testData):
        return self.model.predict(testData)



