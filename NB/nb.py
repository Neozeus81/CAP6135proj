import sys
import os
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')

sys.path.append(file_dir)

from data import Data

types = ["GaussianNB", "BernoulliNB", "CatergoricalNB", "ComplementNB", "MultinomialNB"]

class myNB:
    def __init__(self, cur):
        if cur == 0:
            print("gaussian")
            self.model = GaussianNB()
        if cur == 1:
            print("BernoulliNB")
            self.model = BernoulliNB()
        if cur == 3:
            print("ComplementNB")
            self.model = ComplementNB()
        if cur == 4:
            print("Multinominal")
            self.model = MultinomialNB()

    def train(self, trainData, trainLabels):
        self.model.fit(trainData, trainLabels)

    def predict(self, testData):
        return self.model.predict(testData)
