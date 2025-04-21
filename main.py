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
from tree.tree import myTree

from data import Data
from mongoInterface import myConnection, message

dbconn = myConnection('local', 'Term_proj',"mongodb://cap.aflohr.com:27017/")

train = Data("KDD_TRAIN", ['protocol_type', 'service', 'flag'])
test_plus = Data("KDD_TEST_PLUS", ['protocol_type', 'service', 'flag'])
test_minus =  Data("KDD_TEST_MINUS", ['protocol_type', 'service', 'flag'])

col_diffs_plus = list(set(train.data.columns.values) - set(test_plus.data.columns.values))
col_diffs_minus = list(set(train.data.columns.values) - set(test_minus.data.columns.values))

nodes = [60, 80, 120,160]
lrs = [.01, .1, .5, .8]
datasets = [test_plus, test_minus]
print("mlp")
for test_dataset in datasets:
    for node in nodes:
        for lr in lrs:
            # create model
            if test_dataset.name == "KDD_TEST_PLUS":
                col_diffs = col_diffs_plus
            else:
                col_diffs = col_diffs_minus

            mlp = myMLP(node, 'logistic', lr, 100)
            # binary classification
            mlp.train(train.get_train_data(col_diffs), train.get_label_data(True))
            b_train_pred = mlp.predict(train.get_train_data(col_diffs)) 
            b_train_acc = accuracy_score(b_train_pred, train.get_label_data(True))
            b_test_pred = mlp.predict(test_dataset.get_train_data([]))
            b_test_acc = accuracy_score(b_test_pred, test_dataset.get_label_data(True))
            msg = message("mlp", test_dataset.name, 'logistic', 100, node, lr, False, b_train_acc, b_test_acc, 0)
            dbconn.insert(msg.buildMsg())

            # multi class
            mlp.train(train.get_train_data(col_diffs), train.get_label_data(False))
            m_train_pred = mlp.predict(train.get_train_data(col_diffs))
            m_train_acc = accuracy_score(m_train_pred, train.get_label_data(False))
            m_test_pred = mlp.predict(test_dataset.get_train_data([]))
            m_test_acc = accuracy_score(m_test_pred, test_dataset.get_label_data(False))

            msg = message("mlp", test_dataset.name, 'logistic', 100, node, lr, True, m_train_acc, m_test_acc, 0)
            dbconn.insert(msg.buildMsg())

types = ["GaussianNB", "BernoulliNB", "CatergoricalNB", "ComplementNB", "MultinomialNB"]

print("NB")
for test_dataset in datasets:
    for i in range(0,5):
        print(i)
        if test_dataset.name == "KDD_TEST_PLUS":
            col_diffs = col_diffs_plus
        else:
            col_diffs = col_diffs_minus
        if i == 2:
            continue

        nb = myNB(i)
        nb.train(train.get_train_data(col_diffs), train.get_label_data(True))
        b_train_pred = nb.predict(train.get_train_data(col_diffs))
        b_train_acc = accuracy_score(b_train_pred, train.get_label_data(True))
        b_test_pred = nb.predict(test_dataset.get_train_data([]))
        b_test_acc = accuracy_score(b_test_pred, test_dataset.get_label_data(True))
        msg = message("nb", test_dataset.name, types[i], "none", "none", "none", False, b_train_acc, b_test_acc, 0)
        dbconn.insert(msg.buildMsg())

        # multi class
        nb.train(train.get_train_data(col_diffs), train.get_label_data(False))
        m_train_pred = nb.predict(train.get_train_data(col_diffs))
        m_train_acc = accuracy_score(m_train_pred, train.get_label_data(False))
        m_test_pred = nb.predict(test_dataset.get_train_data([]))
        m_test_acc = accuracy_score(m_test_pred, test_dataset.get_label_data(False))

        msg = message("nb", test_dataset.name, types[i], "none", "none", "none", True, m_train_acc, m_test_acc, 0)
        dbconn.insert(msg.buildMsg())
print("SVMS")
kernals = ['linear', 'poly', 'rbf', 'sigmoid']
for test_dataset in datasets:
    for kernal in kernals:
        if test_dataset.name == "KDD_TEST_PLUS":
            col_diffs = col_diffs_plus
        else:
            col_diffs = col_diffs_minus

        print(kernal)
        svm = mySVM(kernal)
        svm.train(train.get_train_data(col_diffs), train.get_label_data(True))
        b_train_pred = svm.predict(train.get_train_data(col_diffs))
        b_test_pred = svm.predict(test_dataset.get_train_data([]))
        b_train_acc = accuracy_score(train.get_label_data(True), b_train_pred)
        b_test_acc = accuracy_score(test_dataset.get_label_data(True), b_test_pred)
        print("binary classification train:", b_train_acc, "test " , b_test_acc)
        msg = message("svm", test_dataset.name, kernal, 1000, "N/A", "N/A", False, b_train_acc, b_test_acc, 0)
        dbconn.insert(msg.buildMsg())
        svm.train(train.get_train_data(col_diffs), train.get_label_data(False))
        m_train_pred = svm.predict(train.get_train_data(col_diffs))
        m_test_pred = svm.predict(test_dataset.get_train_data([]))
        m_train_acc = accuracy_score(train.get_label_data(False), m_train_pred)
        m_test_acc = accuracy_score(test_dataset.get_label_data(False), m_test_pred)
        print("Multi classification train:", m_train_acc, "test " , m_test_acc)
        msg = message("svm", test_dataset.name, kernal, 1000, "N/A", "N/A", True, m_train_acc, m_test_acc, 0)

        dbconn.insert(msg.buildMsg())

criterions = ["gini", "entropy"]
for test_dataset in datasets:
    for crit in criterions:
        if test_dataset.name == "KDD_TEST_PLUS":
            col_diffs = col_diffs_plus
        else:
            col_diffs = col_diffs_minus
        forest = myRFC(crit)
        forest.train(train.get_train_data(col_diffs), train.get_label_data(True))
        b_train_pred = forest.predict(train.get_train_data(col_diffs))
        b_test_pred = forest.predict(test_dataset.get_train_data([]))
        b_train_acc = accuracy_score(train.get_label_data(True), b_train_pred)
        b_test_acc = accuracy_score(test_dataset.get_label_data(True), b_test_pred)
        print("binary classification train:", b_train_acc, "test " , b_test_acc)
        msg = message("Forest", test_dataset.name, "N/A", 0, "N/A", "N/A", False, b_train_acc, b_test_acc, 0)
        dbconn.insert(msg.buildMsg())
        forest.train(train.get_train_data(col_diffs), train.get_label_data(False))
        m_train_pred = forest.predict(train.get_train_data(col_diffs))
        m_test_pred = forest.predict(test_dataset.get_train_data([]))
        m_train_acc = accuracy_score(train.get_label_data(False), m_train_pred)
        m_test_acc = accuracy_score(test_dataset.get_label_data(False),m_test_pred)
        print("Multi classification train:", m_train_acc, "test " , m_test_acc)
        msg = message("Forest", test_dataset.name, "N/A", 0, "N/A", "N/A", True, m_train_acc, m_test_acc, 0)
        dbconn.insert(msg.buildMsg())
for test_dataset in datasets:
    if test_dataset.name == "KDD_TEST_PLUS":
        col_diffs = col_diffs_plus
    else:
        col_diffs = col_diffs_minus


    tree = myTree()
    tree.train(train.get_train_data(col_diffs), train.get_label_data(True))
    b_train_pred = tree.predict(train.get_train_data(col_diffs))
    b_test_pred = tree.predict(test_dataset.get_train_data([]))
    b_train_acc = accuracy_score(train.get_label_data(True), b_train_pred)
    b_test_acc = accuracy_score(test_dataset.get_label_data(True), b_test_pred)
    print("binary classification train:", b_train_acc, "test " , b_test_acc)
    msg = message("tree", test_dataset.name, "N/A", 0, "N/A", "N/A", False, b_train_acc, b_test_acc, 0)
    dbconn.insert(msg.buildMsg())
    tree.train(train.get_train_data(col_diffs), train.get_label_data(False))
    m_train_pred = tree.predict(train.get_train_data(col_diffs))
    m_test_pred = tree.predict(test_dataset.get_train_data([]))
    m_train_acc = accuracy_score(train.get_label_data(False), m_train_pred)
    m_test_acc = accuracy_score(test_dataset.get_label_data(False),m_test_pred)
    print("Multi classification train:", m_train_acc, "test " , m_test_acc)
    msg = message("tree", test_dataset.name, "N/A", 0, "N/A", "N/A", True, m_train_acc, m_test_acc, 0)
    dbconn.insert(msg.buildMsg())

