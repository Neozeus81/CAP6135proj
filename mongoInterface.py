#!/bin/python3

import pymongo
import json
import datetime



class myConnection:
    def __init__(self, dbName, collectionName, url):
        self.client = pymongo.MongoClient(url)
        self.db = self.client[dbName]
        self.col = self.db[collectionName]
        print("connected")

    def insert(self, msg):
        self.col.insert_one(msg)



class message:
    def __init__(self, model, dataset, act_func, num_nodes, lr, b_train_acc, b_test_acc, m_train_acc, m_test_acc):
        self.model = model
        self.dataset = dataset
        self.act_func = act_func
        self.num_nodes = num_nodes
        self.lr = lr
        self.b_train_acc = b_train_acc
        self.b_test_acc = b_test_acc
        self.m_train_acc = m_train_acc
        self.m_test_acc = m_test_acc

    def buildMsg(self):
        message = {
            "model" : self.model,
            "dataset" : self.dataset,
            "act_func": self.act_func,
            "num_nodes" : self.num_nodes,
            "lr" : self.lr,
            "binary_train_acc" : self.b_train_acc,
            "binary_test_acc" : self.b_test_acc,
            "multi_train_acc" : self.m_train_acc,
            "multi_test_acc" : self.m_test_acc,
            "timestamp" : datetime.datetime.now().timestamp()
        }
        print(json.dumps(message))
        return message
