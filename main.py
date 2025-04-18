#!/usr/bin/python3

import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import accuracy_score

load_dotenv()
from MLP.mlp import myMLP
from NB.nb import myNB
from SVM.svm import mySVM
from forest.forest import myRFC

from data import Data
from mongoInterface import myConnection, message

dbconn = myConnection('local', 'Term_proj',"mongodb://cap.aflohr.com:27017/")

train = Data("KDD_TRAIN", ['protocol_type', 'service', 'flag'])
test_plus = Data("KDD_TEST_PLUS", ['protocol_type', 'service', 'flag'])
test_minus =  Data("KDD_TEST_MINUS", ['protocol_type', 'service', 'flag'])

col_diffs = list(set(train.data.columns.values) - set(test_plus.data.columns.values))

print(col_diffs)
nodes = [60, 80, 120,160]
lrs = [.01, .1, .5, .8]
"""
for node in nodes:
    for lr in lrs:
        # create model
        mlp = myMLP(node, 'logistic', lr, 100)
        # binary classification
        mlp.train(train.get_train_data(col_diffs), train.get_label_data(True))
        b_train_pred = mlp.predict(train.get_train_data(col_diffs)) 
        b_train_acc = accuracy_score(b_train_pred, train.get_label_data(True))
        b_test_pred = mlp.predict(test_plus.get_train_data([]))
        b_test_acc = accuracy_score(b_test_pred, test_plus.get_label_data(True))
        msg = message("mlp", "plus", 'logistic', 100, node, lr, False, b_train_acc, b_test_acc, 0)
        dbconn.insert(msg.buildMsg())

        # multi class
        mlp.train(train.get_train_data(col_diffs), train.get_label_data(False))
        m_train_pred = mlp.predict(train.get_train_data(col_diffs))
        m_train_acc = accuracy_score(m_train_pred, train.get_label_data(False))
        m_test_pred = mlp.predict(test_plus.get_train_data([]))
        m_test_acc = accuracy_score(m_test_pred, test_plus.get_label_data(False))

        msg = message("mlp", "plus", 'logistic', 100, node, lr, True, m_train_acc, m_test_acc, 0)
        dbconn.insert(msg.buildMsg())
"""
types = ["GaussianNB", "BernoulliNB", "CatergoricalNB", "ComplementNB", "MultinomialNB"]

for i in range(0,5):
    print(i)
    if i == 2:
        continue
    nb = myNB(i)
    nb.train(train.get_train_data(col_diffs), train.get_label_data(True))
    b_train_pred = nb.predict(train.get_train_data(col_diffs))
    b_train_acc = accuracy_score(b_train_pred, train.get_label_data(True))
    b_test_pred = nb.predict(test_plus.get_train_data([]))
    b_test_acc = accuracy_score(b_test_pred, test_plus.get_label_data(True))
    msg = message("nb", "plus", types[i], "none", "none", "none", False, b_train_acc, b_test_acc, 0 )
    dbconn.insert(msg.buildMsg())

    """# multi class
    nb.train(train.get_train_data(col_diffs), train.get_label_data(False))
    m_train_pred = nb.predict(train.get_train_data(col_diffs))
    m_train_acc = accuracy_score(m_train_pred, train.get_label_data(False))
    m_test_pred = nb.predict(test_plus.get_train_data([]))
    m_test_acc = accuracy_score(m_test_pred, test_plus.get_label_data(False))

    msg = message("nb", "plus", types[i], "none", "none",  True, m_train_acc, m_test_acc, 0)
    dbconn.insert(msg.buildMsg())
        """

svm = mySVM('poly')
svm.train(train.get_train_data(col_diffs), train.get_label_data(True))
b_train_pred = svm.predict(train.get_train_data(col_diffs))
print(accuracy_score(train.get_label_data(True), b_train_pred))


forest = myRFC("Hi")
