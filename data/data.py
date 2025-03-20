import os
from dotenv import load_dotenv
import pandas as pd 
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


class data: 
    def __init__(self, name, columns):
        load_dotenv("../.env")
        traindf = pd.DataFrame(arff.loadarff(os.getenv(name))[0])
        encoded = pd.get_dummies(traindf, columns=columns)
        num_cols = encoded.select_dtypes(include='number').columns
        ct = ColumnTransformer([('scaler', MinMaxScaler(), num_cols)], remainder='passthrough')
        norm_data = ct.fit_transform(encoded)
        self.data = pd.DataFrame(norm_data, columns=encoded.columns)

    def print_data(self):
        print(self.data)

""" example usage
myData = data("KDD_TRAIN", ['protocol_type', 'service', 'flag'])
myData.print_data()
"""
