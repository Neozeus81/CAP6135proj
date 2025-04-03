import sys
import os
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CatergoricalNB, ComplementNB, MultinominalNB
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')

sys.path.append(file_dir)

from data import Data

types = ["GaussianNB", "BernoulliNB", "CatergoricalNB", "ComplementNB", "MultinomialNB"]

class myNB:
    def __init__(self, what_type):
        self.types = ["GaussianNB", "BernoulliNB", "CatergoricalNB", "ComplementNB", "MultinomialNB"]
        if what_type in self.types:
            match what_type:
                case self.types[0]:
                    self.model = GaussianNB()
                case self.types[1]:
                    self.model = BernoulliNB()
                case self.types[2]:
                    self.model = CatergoicalNB()
                case self.types[3]:
                    self.model = ComplementNB()
                case self.types[4]:
                    self.model = MultinominalNB()
        else:
            print("Invalid type value")

    def train(self, trainData, trainLabels):
        self.model.fit(trainData, trainLabels)

    def predict(self, testData):
        return self.model.predict(testData)
