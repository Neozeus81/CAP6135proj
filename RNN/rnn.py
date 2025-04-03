import sys
import os
from dataset import Dataset

import numpy as np
import tensorflow as tf
import keras

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')
sys.path.append(file_dir)

from data import Data

train = Data("KDD_TRAIN", ['protocol_type', 'service', 'flag']) 


from keras import layers

model = keras.Sequential()

model.add(layers.Embedding(input_dim=122, output_dim=30))

model.add(layers.SimpleRNN(80))

model.add(layers.Dense(2))

model.summary()

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer = 'Adam')
print(type(train.get_train_data()))
model.fit(train.get_train_data().to_numpy(), train.get_label_data(True).to_numpy())
