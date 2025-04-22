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
    def __init__(self, model, dataset, act_func, epochs, num_nodes, lr, isBinary, train_acc, test_acc, time):
        self.model = model
        self.dataset = dataset
        self.act_func = act_func
        self.epochs = epochs
        self.num_nodes = num_nodes
        self.lr = lr
        self.isBinary = isBinary
        self.train_acc = train_acc
        self.test_acc = test_acc
        self.train_time = time

    def buildMsg(self):
        message = {
            "model" : self.model,
            "dataset" : self.dataset,
            "act_func": self.act_func,
            "epochs" : self.epochs,
            "num_nodes" : self.num_nodes,
            "lr" : self.lr,
            "isMulti" : self.isBinary,
            "train_acc" : self.train_acc,
            "test_acc" : self.test_acc,
            "train_time" : self.train_time,
            "timestamp" : datetime.datetime.now().timestamp()
        }
        print(json.dumps(message))
        return message
    def sendCM(self, cm):
        message = {
            "model" : self.model,
            "dataset" : self.dataset,
            "act_func": self.act_func,
            "epochs" : self.epochs,
            "num_nodes" : self.num_nodes,
            "lr" : self.lr,
            "isMulti" : self.isBinary,
            "train_acc" : self.train_acc,
            "test_acc" : self.test_acc,
            "train_time" : self.train_time,
            "Confusion" : json.dumps(cm.tolist()), 
            "timestamp" : datetime.datetime.now().timestamp()
        }
        print(json.dumps(message))
        return message
