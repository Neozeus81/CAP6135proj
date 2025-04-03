import sys
import os
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')

sys.path.append(file_dir)

from data import Data

class myMLP:
    def __init__(self, x, act, learn, epochs):
        self.model = MLPClassifier(hidden_layer_sizes=(x,), activation=act, solver='adam', learning_rate_init=learn, max_iter=epochs, random_state=42)

    def train(self, trainData, trainLabels):
        self.model.fit(trainData, trainLabels)

    def predict(self, testData):
        return self.model.predict(testData)
"""
train = Data("KDD_TRAIN", ['protocol_type', 'service', 'flag'])
test = Data("KDD_TEST_PLUS", ['protocol_type', 'service', 'flag'])
col_diffs = list(set(train.data.columns.values) - set(test.data.columns.values))
train.data = train.data.drop(columns=col_diffs, axis=1)
mlp = myMLP(80,10, 'logistic', 100)
mlp.train(train.get_train_data(), train.get_label_data(True))
y_pred = mlp.predict(test.get_train_data())
print(accuracy_score(test.get_label_data(True), y_pred))
"""

