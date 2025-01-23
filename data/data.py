import os
from dotenv import load_dotenv
import pandas as pd 
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


load_dotenv()

KDD_TRAIN = os.getenv("KDD_TRAIN")
KDD_TEST_PLUS = os.getenv("KDD_TEST_PLUS")

traindf = pd.DataFrame(arff.loadarff(KDD_TRAIN)[0])
print(traindf)
print(traindf.columns.tolist())

encoddf= pd.get_dummies(traindf, columns=['protocol_type', 'service', 'flag'])

print(encoddf)

numerical_cols = encoddf.select_dtypes(include='number').columns
print(numerical_cols)
ct = ColumnTransformer([('scaler', MinMaxScaler(), numerical_cols)], remainder='passthrough')

# Fit and transform the data
df_normalized = ct.fit_transform(encoddf)

# Convert the result back to a DataFrame
df_normalized = pd.DataFrame(df_normalized, columns=encoddf.columns)

print(df_normalized)


