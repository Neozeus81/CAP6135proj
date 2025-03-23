import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
from MLP.mlp import myMLP

from data import Data

train = Data("KDD_TRAIN", ['protocol_type', 'service', 'flag'])
test = Data("KDD_TEST_PLUS", ['protocol_type', 'service', 'flag'])

col_diffs = list(set(train.data.columns.values) - set(test.data.columns.values))

mlp = myMLP(80, 40, 'logistic', 100)
mlp.train(train.get_train_data(), train.get_label_data(True))



