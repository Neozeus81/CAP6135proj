import sys
import os
import time
from dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,  confusion_matrix

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')
sys.path.append(file_dir)

from data import Data
from mongoInterface import myConnection, message
def predict(self, x, y):
    y_pred = np.argmax(self.model.predict(x), axis=1)
    print(accuracy_score(y,y_pred))
    return y_pred

def getConfusionMatrix(self, x, y_labels):
    y_pred = self.predict(x, y_labels)
    return confusion_matrix(y_labels, y_pred)

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, x, x_test, y_train, y_test, isMulti, dataset, lr, nodes, mongoConn):
        self.x = x
        self.dataset = dataset
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.isMulti = isMulti
        self.lr = lr
        self.num_nodes = nodes
        self.time = 0
        self.start_time = 0
        self.end_time = 0
        self.conn = mongoConn

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.perf_counter()
        
    def on_epoch_end(self, epoch, logs=None):
        self.end_time = time.perf_counter()
        self.time += self.end_time - self.start_time
        print(" Time elapsed {}", self.time) 
        if not self.isMulti:
            x_pred = (self.model.predict(self.x) > 0.5).astype(int) 
            train_acc = accuracy_score(self.y_train, x_pred)
            x_pred_test = (self.model.predict(self.x_test) > 0.5).astype(int)
            test_acc = accuracy_score(self.y_test, x_pred_test)
        else:
            x_pred = np.argmax(self.model.predict(x), axis=1)
            train_acc = classification_report(self.y_train, x_pred, output_dict = True).get("accuracy")
            x_pred_test = np.argmax(self.model.predict(x_test), axis=1)
            test_acc = classification_report(self.y_test, x_pred_test, output_dict = True).get("accuracy")

        print(confusion_matrix(self.y_test, x_pred_test))

        mes = message("RNN", self.dataset, "sigmoid", epoch, self.num_nodes, self.lr, self.isMulti, train_acc, test_acc, self.time)  

        self.conn.insert(mes.sendCM(confusion_matrix(self.y_test, x_pred_test)))

dbconn = myConnection('local', 'Term_proj', 'mongodb://localhost:27017')

train = Data("KDD_TRAIN", ['protocol_type', 'service', 'flag']) 
test = Data("KDD_TEST_PLUS", ['protocol_type', 'service', 'flag'])
test_minus = Data("KDD_TEST_MINUS", ['protocol_type', 'service', 'flag'])
"""

y_binary = train.get_label_data(True)
y_binary_test = test_minus.get_label_data(True)
y_multi = train.get_label_data(False)
y_multi_test = test_minus.get_label_data(False)
y_test_minus = test_minus.get_label_data(True)
y_test_minus_multi = test_minus.get_label_data(False)

col_diffs = list(set(train.data.columns.values) - set(test_minus.data.columns.values))

x = StandardScaler().fit_transform(train.get_train_data(col_diffs))
x = x.reshape(x.shape[0], 1, x.shape[1])

x_test = StandardScaler().fit_transform(test_minus.get_train_data([]))
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

col_diffs = list(set(train.data.columns.values) - set(test_minus.data.columns.values))

x_test_minus = StandardScaler().fit_transform(test_minus.get_train_data([]))
x_test_minus = x_test_minus.reshape(x_test_minus.shape[0], 1, x_test_minus.shape[1])
"""


lrs = [.01, .1, .5, .8]
nodes = [20, 60, 80, 120, 240]
datasets = [test_minus]
for test_dataset in datasets:

    col_diffs = list(set(train.data.columns.values) - set(test_dataset.data.columns.values))

    x = StandardScaler().fit_transform(train.get_train_data(col_diffs))
    x = x.reshape(x.shape[0], 1, x.shape[1])

    x_test = StandardScaler().fit_transform(test_dataset.get_train_data([]))
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])


    y_binary = train.get_label_data(True)
    y_binary_test = test_dataset.get_label_data(True)
    y_multi = train.get_label_data(False)
    y_multi_test = test_dataset.get_label_data(False)

    for lr in lrs:
        for num_nodes in nodes:
            binary_model = keras.Sequential()

            binary_model.add(layers.SimpleRNN(num_nodes, input_shape=(x.shape[1], x.shape[2]),activation='sigmoid', return_sequences=False))

            binary_model.add(layers.Dense(1, activation='sigmoid'))


            opt = keras.optimizers.Adam(learning_rate=lr)
            binary_model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer = opt)

            binary_model.summary()

            cb = CustomCallback(x, x_test, y_binary.to_numpy(), y_binary_test.to_numpy(), False, test_dataset.name, lr, num_nodes, dbconn)
            binary_model.fit(x, y_binary, epochs=100, callbacks=[cb])

    for lr in lrs:
        for num_nodes in nodes:
            multi_model = keras.Sequential()

            multi_model.add(layers.SimpleRNN(num_nodes, input_shape=(x.shape[1], x.shape[2]),activation='sigmoid', return_sequences=False))

            multi_model.add(layers.Dense(5, activation='softmax'))


            opt = keras.optimizers.Adam(learning_rate=lr)
            multi_model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer = opt)

            multi_model.summary()
            print(y_multi)
            print(y_multi_test)

            cb = CustomCallback(x, x_test, np.argmax(y_multi,axis=1), np.argmax(y_multi_test, axis=1), True, test_dataset.name, lr, num_nodes, dbconn)
            multi_model.fit(x, y_multi, epochs=100, callbacks=[cb])


